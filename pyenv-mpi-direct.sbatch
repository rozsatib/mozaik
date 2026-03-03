#!/bin/bash

#SBATCH --job-name=mozaik-mpi
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

# 1. PATH CONFIGURATION
#    Verify these paths match your cluster setup (taken from run.sbatch)
WORKING_DIR="/home/goirik/mozaik-models/experanto"
cd "$WORKING_DIR" || exit 1

# 2. ENVIRONMENT CONFIGURATION
#    Set the path to your Python executable (from pyenv-compose.sh)
PYTHON_EXEC="$HOME/mozaik-env/bin/python"

#    Add current directory to PYTHONPATH so python can find the 'mozaik' library
# export PYTHONPATH="$WORKING_DIR:$PYTHONPATH"

# 3. THREADING LIMITS (Critical for MPI)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Submitting job from: ${SLURM_SUBMIT_DIR}"
echo "Python executable:   ${PYTHON_EXEC}"
echo "PYTHONPATH:          ${PYTHONPATH}"

rm -rf SelfSustainedPushPull_test_____/

# 4. RUN SIMULATION
#    Using srun to launch the MPI tasks. 
#    Check if 'param_MSA/defaults' exists relative to where you are running.
#    Note: If --mpi=pmix_v5 fails, try removing the flag or using --mpi=pmi2

srun --mpi=pmix_v5 "$PYTHON_EXEC" -u run.py nest 32 param_MSA/defaults test