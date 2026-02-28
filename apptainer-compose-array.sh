#!/bin/bash

# Define variables

# PROJECT_ROOT="/mnt/vast-nhr/projects/nix00014/goirik/mozaik-models/Rozsa_Cagnol2024" 
PROJECT_ROOT="$PWD/../mozaik-models/experanto" 
SIF_IMAGE="$PWD/../mozaik-sif/mozaik-opt.sif"
ENV_FILE=".env"
MOZAIK_ROOT="$PWD"
EXPERANTO_ROOT="$PWD/../../experanto"
# DATA_ROOT="$PWD/../data"
DATA_ROOT="/mnt/vast-react/projects/neural_foundation_model"
# DATA_ROOT="/mnt/vast-react/projects/neural_foundation_model"

echo PROJECT_ROOT: $PROJECT_ROOT        
echo SIF_IMAGE: $SIF_IMAGE
echo ENV_FILE: $ENV_FILE
echo MOZAIK_ROOT: $MOZAIK_ROOT
echo EXPERANTO_ROOT: $EXPERANTO_ROOT

# Inherit the thread counts populated by your SLURM script
# Fallback to 8 if not running under SLURM
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${OMP_NUM_THREADS:-4}
export VECLIB_MAXIMUM_THREADS=${OMP_NUM_THREADS:-4}

# Capture SLURM tasks, default to 4
export NTASKS=${SLURM_NTASKS:-12}

echo "Starting Mozaik Container..."
apptainer exec \
 --cleanenv \
 --env OMPI_MCA_orte_tmpdir_base=/tmp \
 --env PYTHONPATH="/mozaik:$PYTHONPATH" \
 --env STIM_OFFSET="$STIM_OFFSET" \
 --env STIM_WINDOW="$STIM_WINDOW" \
 --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
 --env MKL_NUM_THREADS=$MKL_NUM_THREADS \
 --env OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS \
 --env NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS \
 --env NTASKS=$NTASKS \
 --bind "$PROJECT_ROOT:/project" \
 --bind "$MOZAIK_ROOT:/mozaik" \
 --bind "$EXPERANTO_ROOT:/experanto" \
 --bind "$DATA_ROOT:/data" \
 "$SIF_IMAGE" \
 bash apptainer-runners/mozaik-simulation-array.sh