#!/bin/bash

# 1. Navigate to the correct directory
cd /project

# 2. Safety Check: Find run.py
if [ ! -f "run.py" ]; then
    echo "WARNING: run.py not found in /project. Checking subdirectories..."
    if [ -d "examples/VogelsAbbott2005" ]; then
        cd examples/VogelsAbbott2005
        echo "Changed directory to: $(pwd)"
    else
        echo "ERROR: Could not find run.py. Current contents of $(pwd):"
        ls -F
        exit 1
    fi
fi

# 3. Clean up previous results
rm -rf SelfSustainedPushPull_test:optntasks32_____

echo "--- Starting Simulation (Internal MPI + Source Python) ---"

# 4. DEFINITIONS (CRITICAL FIXES)

# Path to the custom Python we compiled (Source Build)
CONTAINER_PYTHON="/usr/local/mozaik-python/bin/python"

# Path to the System OpenMPI (Bypasses the Intel MPI/Hydra that caused the error)
# On Ubuntu containers, OpenMPI is strictly located at /usr/bin/mpirun
MPI_EXEC="/usr/bin/mpirun"

# 5. Run with Explicit Paths
echo "Using MPI: $MPI_EXEC"
echo "Using Python: $CONTAINER_PYTHON"

$MPI_EXEC --bind-to core \
    -n 32 \
    -x OMP_NUM_THREADS=1 \
    -x MKL_NUM_THREADS=1 \
    -x OPENBLAS_NUM_THREADS=1 \
    -x PYTHONPATH \
    $CONTAINER_PYTHON -u run.py nest 32 param_MSA/defaults 'test:optntasks32'

echo "--- Simulation Finished with Code $? ---"