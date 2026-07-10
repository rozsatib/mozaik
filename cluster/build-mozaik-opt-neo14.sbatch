#!/bin/bash
# Background build of the updated MOZAIK container (neo 0.14.4 / numpy<2 / CSNG-MFF imagen / PyNN pin).
# Submit from the mozaik repo root:   sbatch cluster/build-mozaik-opt-neo14.sbatch
# Runs on the SAME arch as the sim run nodes (sapphirerapids) because the def compiles NEST with
# -march=native. Non-destructive: refuses to overwrite an existing output sif.
#
# CAVEATS (read before submitting):
#  * INTERNET: the build git-clones imagen/PyNN, wgets NEST, and pip-installs. If medium96s compute
#    nodes have no outbound network here, run the build on a login/build node instead (see the nohup
#    fallback at the bottom of this file) or via a proxy.
#  * FAKEROOT: the def uses apt-get, which needs root inside the build -> --fakeroot (requires your
#    subuid/subgid to be configured). If --fakeroot is rejected, ask admins for the build path used
#    for the original mozaik-opt.sif.

#SBATCH --job-name=mozaik-sif-build
#SBATCH --partition=medium96s
#SBATCH --constraint=sapphirerapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=mozaik-sif-build-%j.out
#SBATCH --error=mozaik-sif-build-%j.err

set -euo pipefail

module load apptainer 2>/dev/null || true
command -v apptainer >/dev/null || { echo "ERROR: apptainer not on PATH"; exit 1; }

# Resolve to the mozaik repo root (this script lives in cluster/)
cd "$(dirname "$0")/.."
MOZAIK_ROOT="$PWD"

DEF="mozaik-opt-neo14_2026-07.def"
OUT="../mozaik-sif/mozaik-opt_2026-07-10.sif"

[ -f "$DEF" ] || { echo "ERROR: $DEF not found in $MOZAIK_ROOT"; exit 1; }
# Non-destructive guard: never clobber an existing sif (old or new).
if [ -e "$OUT" ]; then
  echo "REFUSE: $OUT already exists — remove or rename it first (non-destructive policy)."; exit 1
fi

# Build scratch on NODE-LOCAL NVMe (fast for the metadata/small-file-heavy work: rootfs extract,
# apt, and the NEST `make -j` compile). SLURM sets $LOCAL_TMPDIR to a job-scoped dir on /local
# (1.7T NVMe, auto-created, auto-cleaned). Do NOT use lustre (/mnt/lustre-grete) or the 4G tmpfs
# /tmp here — lustre is slow for small files and would bottleneck the build; /tmp overflows.
BUILD_TMP="${LOCAL_TMPDIR:-/local/jobs/${SLURM_JOB_ID:-$$}}/apptainer-build"
export APPTAINER_TMPDIR="$BUILD_TMP"
export APPTAINER_CACHEDIR="$BUILD_TMP/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
echo "build scratch (node-local NVMe): $BUILD_TMP"
df -hT "$BUILD_TMP" 2>/dev/null | tail -1

echo "=== build start $(date) on $(hostname) ==="
echo "DEF=$DEF  OUT=$OUT  TMPDIR=$APPTAINER_TMPDIR"

apptainer build --fakeroot "$OUT" "$DEF"

echo "=== build done; capturing new-env manifest ==="
apptainer exec "$OUT" bash -lc '. /opt/mozaik/bin/activate 2>/dev/null; \
  python -c "import numpy,neo,pyNN,imagen; print(\"numpy\",numpy.__version__,\"neo\",neo.__version__,\"pyNN\",pyNN.__version__)"; \
  pip freeze' > ../docs/plan/updating-mozaik/new-env.pip-freeze.txt 2>&1 || true

echo "=== BUILD COMPLETE: $OUT ($(du -h "$OUT" | cut -f1)) $(date) ==="

# --------------------------------------------------------------------------------------------------
# Login/build-node fallback (if compute nodes lack internet). Run from the mozaik repo root.
# Login /tmp is a 4G tmpfs (too small); use node-local NVMe /local (1.7T) for build scratch:
#
#   export APPTAINER_TMPDIR=/local/$USER/apptainer-build
#   export APPTAINER_CACHEDIR=$APPTAINER_TMPDIR/cache
#   mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
#   module load apptainer
#   nohup apptainer build --fakeroot ../mozaik-sif/mozaik-opt_2026-07-10.sif \
#         mozaik-opt-neo14_2026-07.def > mozaik-sif-build.out 2>&1 &
#   tail -f mozaik-sif-build.out
#   # clean up afterwards:  rm -rf /local/$USER/apptainer-build
# --------------------------------------------------------------------------------------------------
