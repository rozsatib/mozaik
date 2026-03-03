#!/bin/bash

# 1. Navigate to the correct directory
# Based on your previous logs, the code seems to be in /project
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
rm -rf SelfSustainedPushPull_test:test32_____

echo "--- Starting Simulation (Internal MPI) ---"
echo "Host Thread Limit Check: OMP_NUM_THREADS=$OMP_NUM_THREADS"

# 4. Run the Simulation
# We use -x to FORCE the environment variables into every worker process.
# We remove '--bind-to core' temporarily to let the OS scheduler handle the load 
# (this prevents the 'floating process' thrashing if the container can't see the hardware topology).

mpirun --bind-to core \
    -n 32 \
    -x OMP_NUM_THREADS=1 \
    -x MKL_NUM_THREADS=1 \
    -x OPENBLAS_NUM_THREADS=1 \
    -x PYTHONPATH \
    python -u run.py nest 32 param_MSA/defaults 'test:test32'

echo "--- Simulation Finished with Code $? ---"