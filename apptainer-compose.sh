#!/bin/bash

# Define variables

# PROJECT_ROOT="/mnt/vast-nhr/projects/nix00014/goirik/mozaik-models/Rozsa_Cagnol2024" 
PROJECT_ROOT="$PWD/../mozaik-models/experanto" 
SIF_IMAGE="$PWD/../mozaik-sif/mozaik-opt.sif"
ENV_FILE=".env"
MOZAIK_ROOT="$PWD"
EXPERANTO_ROOT="$PWD/../../experanto"
DATA_ROOT="$PWD/../data"
# DATA_ROOT="/mnt/vast-react/projects/neural_foundation_model"

echo PROJECT_ROOT: $PROJECT_ROOT        
echo SIF_IMAGE: $SIF_IMAGE
echo ENV_FILE: $ENV_FILE
echo MOZAIK_ROOT: $MOZAIK_ROOT

# # Export environment variables from .env file
# set -o allexport
# source "$ENV_FILE"
# set +o allexport

# Run the container (Internal MPI Mode)
# We export the threading limits here so Apptainer picks them up
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Starting Mozaik Container..."
apptainer exec \
 --cleanenv \
 --env OMPI_MCA_orte_tmpdir_base=/tmp \
 --env PYTHONPATH="/mozaik:$PYTHONPATH" \
 --env OMP_NUM_THREADS=1 \
 --env MKL_NUM_THREADS=1 \
 --env OPENBLAS_NUM_THREADS=1 \
 --env NUMEXPR_NUM_THREADS=1 \
 --bind "$PROJECT_ROOT:/project" \
 --bind "$MOZAIK_ROOT:/mozaik" \
 --bind "$EXPERANTO_ROOT:/experanto" \
 --bind "$DATA_ROOT:/data" \
 "$SIF_IMAGE" \
 bash apptainer-runners/mozaik-simulation-mpi.sh