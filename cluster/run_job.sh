#!/bin/bash
# run_job.sh — single bodiless MOZAIK job script (submitted by cluster/submit.sh).
#
# No #SBATCH directives here: they arrive as `sbatch` CLI flags from submit.sh. This script sources
# the experiment config ($MOZAIK_CONF), runs the shared launch boilerplate, and dispatches to the
# right compose script by WORKLOAD. Replaces the former per-scenario run-*.sbatch launchers.
MOZAIK_ROOT=/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

: "${MOZAIK_CONF:?run_job.sh requires MOZAIK_CONF — submit via cluster/submit.sh}"
# shellcheck disable=SC1090
source "$MOZAIK_CONF"

# Proxies are needed for `module load` but break the `parameters` library, so set then unset (matches
# the old launchers exactly).
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"
module load apptainer
cd "$MOZAIK_ROOT"

# Threading (compose scripts also default these to 4; set explicitly to match the old launchers).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_DYNAMIC=FALSE

echo "Config: $MOZAIK_CONF | Workload: $WORKLOAD"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-<none>} | Node: ${SLURM_NODELIST:-?} | PWD: $PWD"
echo "Unsetting proxies to avoid 'parameters' library bug..."
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY

mkdir -p "$(dirname "$OUTPUT")"

case "$WORKLOAD" in
  sim)
    N_CHUNKS="${N_CHUNKS:-12}"
    export TRIAL=$(( ${SLURM_ARRAY_TASK_ID:-0} / N_CHUNKS ))
    export CHUNK=$(( ${SLURM_ARRAY_TASK_ID:-0} % N_CHUNKS ))
    export CHUNK_DIR   # empty -> compose default /data/mozaik_chunk
    export SIF_IMAGE PARAM_FILE   # empty -> compose/runner defaults (old sif, param/defaults)
    echo "Sim: TRIAL=$TRIAL CHUNK=$CHUNK (N_CHUNKS=$N_CHUNKS) CHUNK_DIR=${CHUNK_DIR:-<default>} SIF=${SIF_IMAGE:-<default>} PARAM=${PARAM_FILE:-<default>}"
    bash cluster/apptainer-compose-array.sh
    ;;
  export)
    export TRIAL="${SLURM_ARRAY_TASK_ID:-1}"
    # Any of these left unset by the config export as empty -> compose applies its own defaults.
    export N_CHUNKS CHUNK_START CHUNK_END BATCH_SIZE EXPORT_MODE CHUNK_DIR OUTPUT_PREFIX DATASTORE_PREFIX MODALITY_FILTER SIF_IMAGE
    echo "Export: TRIAL=$TRIAL"
    bash cluster/apptainer-compose-export.sh
    ;;
  *)
    echo "run_job.sh: unknown WORKLOAD='$WORKLOAD' (expected sim|export)" >&2
    exit 2
    ;;
esac
