#!/bin/bash
# submit.sh — submit a MOZAIK cluster job from a human-readable experiment config.
#
# The config (cluster/experiments/*.conf) is the single source of truth for BOTH the scheduler
# directives and the runtime env. This wrapper translates the scheduler keys into `sbatch` CLI flags
# (which #SBATCH directives cannot be set from a sourced file) and submits the bodiless run_job.sh.
#
# Usage:
#   ./cluster/submit.sh cluster/experiments/sim-test3.conf              # submit
#   ./cluster/submit.sh cluster/experiments/sim-test3.conf --dry-run    # print the sbatch command only
#   ./cluster/submit.sh <conf> [extra sbatch args...]                   # pass-through extra flags
set -euo pipefail

CONF="${1:?usage: submit.sh <conf> [--dry-run] [extra sbatch args]}"; shift || true
DRY=0; EXTRA=()
for a in "$@"; do
  if [ "$a" = "--dry-run" ]; then DRY=1; else EXTRA+=("$a"); fi
done
[ -f "$CONF" ] || { echo "submit.sh: config not found: $CONF" >&2; exit 1; }
CONF_ABS="$(cd "$(dirname "$CONF")" && pwd)/$(basename "$CONF")"
# shellcheck disable=SC1090
source "$CONF_ABS"

: "${WORKLOAD:?config must set WORKLOAD}"
: "${JOB_NAME:?config must set JOB_NAME}"; : "${PARTITION:?}"; : "${CONSTRAINT:?}"
: "${NTASKS:?}"; : "${MEM:?}"; : "${TIME:?}"; : "${ARRAY:?}"; : "${OUTPUT:?}"; : "${ERROR:?}"

# Constant directives across all current experiments (override in a config if ever needed).
ACCOUNT="${ACCOUNT:-nix00014}"; NODES="${NODES:-1}"; CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
HINT="${HINT:-nomultithread}"; MAIL_TYPE="${MAIL_TYPE:-ALL}"
MAIL_USER="${MAIL_USER:-goirik.chakrabarty@uni-goettingen.de}"

FLAGS=(
  --account="$ACCOUNT" --job-name="$JOB_NAME" --nodes="$NODES" --ntasks="$NTASKS"
  --cpus-per-task="$CPUS_PER_TASK" --hint="$HINT" --mem="$MEM" --time="$TIME"
  --array="$ARRAY" --partition="$PARTITION" --constraint="$CONSTRAINT"
  --output="$OUTPUT" --error="$ERROR" --mail-type="$MAIL_TYPE" --mail-user="$MAIL_USER"
)
CMD=(sbatch "${FLAGS[@]}" --export=ALL,MOZAIK_CONF="$CONF_ABS" cluster/run_job.sh)

if [ "$DRY" = 1 ]; then
  printf '%s ' "${CMD[@]}"; echo
else
  exec "${CMD[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
fi
