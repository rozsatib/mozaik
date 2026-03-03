import sys
import os
import platform
import time
import importlib.util

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_package(name):
    try:
        pkg = importlib.import_module(name)
        version = getattr(pkg, '__version__', 'Unknown')
        path = getattr(pkg, '__file__', 'Unknown')
        print(f"[{name}]")
        print(f"  Version: {version}")
        print(f"  Path:    {path}")
        return pkg
    except ImportError:
        print(f"[{name}] NOT FOUND")
        return None

# ==========================================
# 1. SYSTEM & PYTHON INTERPRETER
# ==========================================
print_header("SYSTEM INFORMATION")
print(f"OS:           {platform.system()} {platform.release()}")
print(f"Machine:      {platform.machine()}")
print(f"Processor:    {platform.processor()}")
print(f"Python Ver:   {sys.version.split()[0]}")
print(f"Python Path:  {sys.executable}")
print(f"Prefix:       {sys.prefix}")

# Check if we are running the specific custom python we built
if "/usr/local/mozaik-python" in sys.executable:
    print("STATUS:  [OK] Running Isolated Source-Built Python")
elif "conda" in sys.executable:
    print("STATUS:  [OK] Running Conda Python")
elif "/opt/python" in sys.executable or "/usr/bin/python" in sys.executable:
    print("STATUS:  [WARNING] Running Host/System Python (Possible Hijack!)")

# ==========================================
# 2. ENVIRONMENT VARIABLES
# ==========================================
print_header("CRITICAL ENVIRONMENT VARIABLES")
env_vars = [
    "OMP_NUM_THREADS", 
    "MKL_NUM_THREADS", 
    "OPENBLAS_NUM_THREADS", 
    "NUMEXPR_NUM_THREADS", 
    "PYTHONPATH", 
    "LD_LIBRARY_PATH",
    "MOZAIK_HOME"
]
for var in env_vars:
    print(f"{var:<22}: {os.environ.get(var, 'Not Set')}")

# ==========================================
# 3. NUMPY & MATH KERNEL (BLAS/LAPACK)
# ==========================================
print_header("NUMPY & LINEAR ALGEBRA")
np = check_package("numpy")
if np:
    print("\n--- NumPy Configuration Details ---")
    try:
        # NumPy 1.26+ prefer show_config()
        np.show_config()
    except AttributeError:
        print("Could not run np.show_config()")

    # Quick Math Benchmark
    print("\n--- Quick Math Performance Test ---")
    N = 2000
    print(f"Creating {N}x{N} random matrix...", end="", flush=True)
    try:
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        print(" Done.")
        
        t0 = time.time()
        print(f"Multiplying (Dot Product)...", end="", flush=True)
        C = np.dot(A, B)
        t1 = time.time()
        duration = t1 - t0
        print(f" Done in {duration:.4f} seconds.")
        
        if duration < 1.5:
            print("Performance: [EXCELLENT] Likely using MKL/AVX.")
        elif duration < 4.0:
            print("Performance: [GOOD] Likely using OpenBLAS.")
        else:
            print("Performance: [SLOW] Likely using generic BLAS or unoptimized code.")
    except Exception as e:
        print(f"\nMath test failed: {e}")

# ==========================================
# 4. MPI SUPPORT
# ==========================================
print_header("MPI4PY CONFIGURATION")
mpi = check_package("mpi4py")
if mpi:
    try:
        from mpi4py import MPI
        print(f"MPI Vendor:   {MPI.get_vendor()}")
        print(f"Library Ver:  {MPI.Get_library_version()}")
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"MPI Status:   Running as Rank {rank} of {size}")
        
        if size == 1:
            print("WARNING: Running on 1 rank. Use 'mpirun -n X' to test parallel setup.")
    except Exception as e:
        print(f"MPI Error: {e}")

# ==========================================
# 5. NEUROSCIENCE STACK (NEST / PyNN / Mozaik)
# ==========================================
print_header("NEUROSCIENCE STACK")

# NEST
nest = check_package("nest")
if nest:
    try:
        print(f"  NEST Ver: {nest.__version__}")
        nest.ResetKernel()
        # Check threads reported by NEST
        status = nest.GetKernelStatus()
        print(f"  Local Threads: {status.get('local_num_threads', '?')}")
        print(f"  Resolution:    {status.get('resolution', '?')} ms")
        
        # Check for loaded extensions (like StepCurrent)
        print(f"  Loaded Models (Sample): {nest.Models()[:5]} ...")
    except Exception as e:
        print(f"  Error checking NEST internals: {e}")

# PyNN
pynn = check_package("pyNN")
if pynn:
    try:
        from pyNN import nest as pynn_nest
        print(f"  PyNN Backend: {pynn_nest.__name__}")
    except ImportError:
        print("  Could not import pyNN.nest")

# Mozaik
mozaik = check_package("mozaik")

# ==========================================
# 6. DIRECTORY CHECKS
# ==========================================
print_header("FILESYSTEM CHECKS")
dirs_to_check = ["/project", "/mozaik", "/data", "/experanto"]
for d in dirs_to_check:
    exists = os.path.exists(d)
    content_count = len(os.listdir(d)) if exists else 0
    print(f"{d:<15}: {'EXISTS' if exists else 'MISSING'} ({content_count} items)")

print("\nDiagnostics Complete.")