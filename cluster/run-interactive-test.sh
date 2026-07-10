#!/bin/bash
# Quick interactive test — no SLURM, runs directly in Apptainer container
# Usage: bash cluster/run-interactive-test.sh [N_TASKS] [NOISE_SEED] [RUN_NAME]
#   e.g. bash cluster/run-interactive-test.sh 4 0 test_interactive

cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

NTASKS=${1:-4}
NOISE_SEED=${2:-0}
RUN_NAME=${3:-test_interactive}

PROJECT_ROOT="$PWD/../mozaik-models/experanto"
SIF_IMAGE="$PWD/../mozaik-sif/mozaik-opt.sif"
MOZAIK_ROOT="$PWD"
EXPERANTO_ROOT="$PWD/../../experanto"
DATA_ROOT="/mnt/vast-react/projects/neural_foundation_model"

echo "=== Interactive Mozaik Test ==="
echo "MPI tasks:  $NTASKS"
echo "Noise seed: $NOISE_SEED"
echo "Run name:   $RUN_NAME"
echo "SIF image:  $SIF_IMAGE"
echo "Project:    $PROJECT_ROOT"
echo "=============================="

apptainer exec \
  --cleanenv \
  --env OMPI_MCA_orte_tmpdir_base=/tmp \
  --env PYTHONPATH="/mozaik" \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --bind "$PROJECT_ROOT:/project" \
  --bind "$MOZAIK_ROOT:/mozaik" \
  --bind "$EXPERANTO_ROOT:/experanto" \
  --bind "$DATA_ROOT:/data" \
  "$SIF_IMAGE" \
  bash -c "cd /project && mpirun -n $NTASKS -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS -x PYTHONPATH python -u run.py nest $NTASKS param/defaults noise_seed $NOISE_SEED $RUN_NAME"
