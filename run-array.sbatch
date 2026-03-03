#!/bin/bash

#SBATCH --account=nix00014
#SBATCH --job-name=mozaik-simulation
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=250G
#SBATCH --array=0-10
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_array/mozaik-opt-%j.out
#SBATCH --error=./slurm_array/mozaik-opt-%j.err
#SBATCH -p medium96s # standard96 or medium96s
#SBATCH -C sapphirerapids # cascadelake or sapphirerapids
#SBATCH --mail-type=ALL
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de

export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer

cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

# Threading environment
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
# export NUMEXPR_NUM_THREADS=1 # Added
# export VECLIB_MAXIMUM_THREADS=1 # Added
export MKL_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE

# Calculate offset
STIMULUS_PER_JOB=50
# Set SLURM_ARRAY_TASK_ID to 1 for testing outside of SLURM
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "SLURM_ARRAY_TASK_ID not set. Defaulting to 1 for testing."
    export SLURM_ARRAY_TASK_ID=1
fi

STIM_OFFSET=$((SLURM_ARRAY_TASK_ID * STIMULUS_PER_JOB))
STIM_WINDOW=$STIMULUS_PER_JOB
export STIM_OFFSET
export STIM_WINDOW

echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Stimulus per job: ${STIM_WINDOW}"
echo "Computed Offset: ${STIM_OFFSET}"

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Unsetting proxies to avoid 'parameters' library bug..."
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY

bash apptainer-compose-array.sh