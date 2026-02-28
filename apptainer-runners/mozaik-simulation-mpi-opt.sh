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

# 3. Clean up previous results (Updated output name)
rm -rf SelfSustainedPushPull_test:test${NTASKS}_____

echo "--- Starting Simulation (Internal MPI) ---"
echo "Host Thread Limit Check: OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Running with $NTASKS MPI Tasks"

# 4. Run the Simulation
# IMPORTANT: '--bind-to core' is REMOVED so OpenMP can use multiple cores per task.
# We pass through the environment variables using -x without an = sign, 
# which tells OpenMPI to pass the existing container values to the workers.

mpirun \
    -n $NTASKS \
    -x OMP_NUM_THREADS \
    -x MKL_NUM_THREADS \
    -x OPENBLAS_NUM_THREADS \
    -x PYTHONPATH \
    python -u run.py nest $NTASKS param_MSA/defaults "test:test${NTASKS}"

echo "--- Simulation Finished with Code $? ---"