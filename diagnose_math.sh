#!/bin/bash

# ==============================================================================
# Mozaik Performance Reproduction Script
# This script mimics the behavior of your 'diagnose_env.py' benchmarks.
# ==============================================================================

PYTHON_EXEC="/home/goirik/mozaik-env/bin/python"
BENCH_SCRIPT="repro_bench.py"

# Create the python benchmark snippet
cat <<EOF > \$BENCH_SCRIPT
import numpy as np
import time
import os

size = 2000
print(f"\n--- Rank {os.environ.get('OMPI_COMM_WORLD_RANK', '0')} Performance Test ---")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not Set')}")
print(f"OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not Set')}")

A = np.random.random((size, size)).astype(np.float64)
B = np.random.random((size, size)).astype(np.float64)

start = time.time()
C = np.dot(A, B)
end = time.time()

print(f"Dot Product (2000x2000) took: {end - start:.4f} seconds")
EOF

echo "=========================================================="
echo "STEP 1: REPRODUCING INITIAL BASELINE (0.015s)"
echo "Description: Multi-threaded OpenBLAS (Default behavior)"
echo "=========================================================="
# Unset limits to allow OpenBLAS to use its default internal threading
unset OMP_NUM_THREADS
unset OPENBLAS_NUM_THREADS
$PYTHON_EXEC \$BENCH_SCRIPT

echo -e "\n"
echo "=========================================================="
echo "STEP 2: REPRODUCING CURRENT SLOWDOWN (0.27s)"
echo "Description: MPI Rank with OMP/OpenBLAS limits set to 1"
echo "=========================================================="
# Force the limits observed in your 3x slowdown case
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpirun -n 2 \
    -x OMP_NUM_THREADS=1 \
    -x OPENBLAS_NUM_THREADS=1 \
    $PYTHON_EXEC \$BENCH_SCRIPT

# Cleanup
rm \$BENCH_SCRIPT