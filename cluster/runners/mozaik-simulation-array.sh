cd /project

# 1. Set defaults if not provided (safe fallback)
TRIAL=${TRIAL:-0}
CHUNK=${CHUNK:-0}

# 2. Create a unique run name using trial and chunk
RUN_NAME="trial${TRIAL}_chunk${CHUNK}"

# 3. Construct the expected output directory name
# Mozaik appends modified parameters: ModelName_RunName_____key:value
# noise_seed is passed as a modified parameter, so it appears in the dir name.
NOISE_SEED=$(( TRIAL * 1000 + CHUNK ))
DIR_NAME="SelfSustainedPushPull_${RUN_NAME}_____noise_seed:${NOISE_SEED}"

echo "Running simulation with name: $RUN_NAME"
echo "Cleaning up directory: $DIR_NAME"

# 4. Remove the specific directory for THIS job only
rm -rf "$DIR_NAME"

echo "--- Starting Simulation (Internal MPI) ---"
echo "Host Thread Limit Check: OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Running with $NTASKS MPI Tasks"
echo "Trial: $TRIAL, Chunk: $CHUNK"

# 5. noise_seed was computed in step 3 above
echo "Per-trial noise_seed=$NOISE_SEED (pynn_seed=5, mozaik_seed=1023 fixed)"

# 6. Run the python script passing the UNIQUE Run Name
#    Modified parameters (key value pairs) go between param_file and run_name.
mpirun \
    -n $NTASKS \
    -x OMP_NUM_THREADS \
    -x MKL_NUM_THREADS \
    -x OPENBLAS_NUM_THREADS \
    -x PYTHONPATH \
    python -u run.py nest $NTASKS param/defaults noise_seed $NOISE_SEED "$RUN_NAME"