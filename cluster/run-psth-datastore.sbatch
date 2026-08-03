#!/bin/bash

#SBATCH --account=nix00014
#SBATCH --job-name=psth-datastore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=./slurm_test33/psth-datastore-%j.out
#SBATCH --error=./slurm_test33/psth-datastore-%j.err
#SBATCH -p standard96
#SBATCH -C cascadelake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de

export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_DYNAMIC=FALSE

mkdir -p ./slurm_test33

echo "Job ID: ${SLURM_JOB_ID}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Unsetting proxies to avoid 'parameters' library bug..."
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY

cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

PROJECT_ROOT="$PWD/../mozaik-models/experanto"
SIF_IMAGE="$PWD/../mozaik-sif/mozaik-opt.sif"
MOZAIK_ROOT="$PWD"
DATA_ROOT="/mnt/vast-react/projects/neural_foundation_model"

echo "Starting PSTH DataStore computation..."
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "SIF_IMAGE: $SIF_IMAGE"

apptainer exec \
 --cleanenv \
 --env PYTHONPATH="/mozaik:$PYTHONPATH" \
 --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
 --env MKL_NUM_THREADS=$MKL_NUM_THREADS \
 --env OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS \
 --bind "$PROJECT_ROOT:/project" \
 --bind "$MOZAIK_ROOT:/mozaik" \
 --bind "$DATA_ROOT:/data" \
 "$SIF_IMAGE" \
 bash -c "cd /project && python -u analysis/compute_psth_datastore.py"
