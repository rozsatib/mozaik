import time
import numpy as np
import nest
import os

def benchmark_numpy():
    print(f"--- NumPy Benchmark ({np.__version__}) ---")
    np.show_config()
    
    N = 5000
    print(f"\nRunning {N}x{N} Matrix Multiplication...")
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    
    t0 = time.time()
    C = np.dot(A, B)
    t1 = time.time()
    print(f"NumPy Time: {t1-t0:.4f} seconds")
    return t1-t0

def benchmark_nest():
    print(f"\n--- NEST Benchmark ({nest.__version__}) ---")
    nest.ResetKernel()
    nest.SetKernelStatus({"local_num_threads": 1})
    
    print("Creating 10,000 neurons...")
    pop = nest.Create("iaf_psc_alpha", 10000)
    nest.Connect(pop, pop, "one_to_one")
    
    print("Simulating 1000ms...")
    t0 = time.time()
    nest.Simulate(1000.0)
    t1 = time.time()
    print(f"NEST Time:  {t1-t0:.4f} seconds")
    return t1-t0

if __name__ == "__main__":
    print(f"Environment: {os.environ.get('MOZAIK_HOME', 'Host/PyEnv')}")
    print(f"Threads: OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS')} "
          f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    
    t_numpy = benchmark_numpy()
    t_nest = benchmark_nest()
    
    print("\n--- SUMMARY ---")
    print(f"Math (Matrix Mul): {t_numpy:.4f}s")
    print(f"Sim  (NEST Kernel): {t_nest:.4f}s")