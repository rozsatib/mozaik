#!/bin/bash

# ==============================================================================
# PyEnv Composition Script
# Analogous to apptainer-compose.sh, but for local/pyenv environments.
# ==============================================================================

# 1. PATH CONFIGURATION
#    Define the locations of your code and data on the Host
PROJECT_ROOT="$PWD/../mozaik-models/experanto" 
MOZAIK_ROOT="$PWD"
EXPERANTO_ROOT="$PWD/../experanto"
DATA_ROOT="$PWD/../data"
ENV_FILE=".env"

# 2. PYTHON & MPI CONFIGURATION
#    Set this to the path of your Local/PyEnv Python executable
#    Example: "$HOME/.pyenv/versions/3.10.13/bin/python" or "$HOME/mozaik-env/bin/python"
PYTHON_EXEC="$HOME/mozaik-env/bin/python"
# PYTHON_EXEC="/home/goirik/miniconda3/envs/mozaik/bin/python"

#    Set this to your OpenMPI executable to avoid the "Intel Hydra" crash
#    If you load modules, this might just be "mpirun"
MPI_EXEC="/usr/bin/mpirun"
# MPI_EXEC="/home/goirik/miniconda3/envs/mozaik/bin/mpirun"

echo "--- PyEnv Simulation Launch ---"
echo "Project Root: $PROJECT_ROOT"
echo "Mozaik Root:  $MOZAIK_ROOT"
echo "Python Exec:  $PYTHON_EXEC"
echo "MPI Exec:     $MPI_EXEC"

# 3. EXPORT ENVIRONMENT VARIABLES
#    Source .env if it exists
if [ -f "$ENV_FILE" ]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

#    CRITICAL: Threading Limits for MPI
#    We must force these on the Host just like we did in the container
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

#    CRITICAL: Add Mozaik to PYTHONPATH
#    This replaces the "--bind /mozaik" step from Apptainer.
#    It forces Python to import the local library version instead of the installed one.
export PYTHONPATH="$MOZAIK_ROOT:$PYTHONPATH"

# 4. EXECUTION
#    Navigate to the project directory (replaces 'cd /project' in the runner)
cd "$PROJECT_ROOT" || exit 1

#    Safety Check for run.py
if [ ! -f "run.py" ]; then
    echo "WARNING: run.py not found in $PROJECT_ROOT"
    if [ -d "examples/VogelsAbbott2005" ]; then
        echo "Found example directory, switching..."
        cd "examples/VogelsAbbott2005"
    else
        echo "ERROR: Could not locate run.py"
        exit 1
    fi
fi

rm -rf SelfSustainedPushPull_test:pyenv_ntasks32_____/

# 5. RUN SIMULATION
#    We execute MPI directly on the host.
#    --bind-to core: Pins processes to cores for speed
#    -x VAR: Exports variables to all workers

$MPI_EXEC --mca pmix pmix_v5 --bind-to core \
    -n 32 \
    -x OMP_NUM_THREADS=1 \
    -x OPENBLAS_NUM_THREADS=1 \
    -x MKL_NUM_THREADS=1 \
    -x NUMEXPR_NUM_THREADS=1 \
    -x VECLIB_MAXIMUM_THREADS=1 \
    -x PYTHONPATH \
    "$PYTHON_EXEC" -u run.py nest 32 param_MSA/defaults 'test:pyenv_ntasks32'

# $MPI_EXEC -n 32 \
#     "$PYTHON_EXEC" -u run.py nest 32 param_MSA/defaults 'test:pyenv_ntasks32'

echo "--- Simulation Finished with Code $? ---"