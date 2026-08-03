cd /project

# 1. Set defaults if not provided (safe fallback)
TRIAL=${TRIAL:-0}
CHUNK=${CHUNK:-0}

# 2. Create a unique run name using trial and chunk
RUN_NAME="trial${TRIAL}_chunk${CHUNK}"

# 3. Construct the expected output directory name.
# Mozaik appends modified parameters as ModelName_RunName_____key:value, but current mozaik
# (result_directory_name) TRUNCATES + hashes keys longer than 24 chars — and
# lgn_stepcurrentsource_noise_seed is 32, so the real dir is ..._____lgn_stepcurre_<sha1>:N.
# Ask mozaik for the exact name instead of reconstructing it, so cleanup matches what run.py creates
# (and does not touch the legacy ..._____noise_seed:N datastores, which hash to a different name).
NOISE_SEED=$(( TRIAL * 1000 + CHUNK ))
DIR_NAME=$(python -c "from mozaik.tools.misc import result_directory_name; print(result_directory_name('${RUN_NAME}','SelfSustainedPushPull',{'lgn_stepcurrentsource_noise_seed':${NOISE_SEED}}))")

echo "Running simulation with name: $RUN_NAME"
echo "Cleaning up directory: $DIR_NAME"

# 4. Remove the specific directory for THIS job only
rm -rf "$DIR_NAME"

echo "--- Starting Simulation (Internal MPI) ---"
echo "Host Thread Limit Check: OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Running with $NTASKS MPI Tasks"
echo "Trial: $TRIAL, Chunk: $CHUNK"

# 5. lgn_stepcurrentsource_noise_seed value was computed in step 3 above
echo "Per-trial lgn_stepcurrentsource_noise_seed=$NOISE_SEED (pynn_seed=5, mozaik_seed=1023 fixed)"

# 6. Run the python script passing the UNIQUE Run Name
#    Modified parameters (key value pairs) go between param_file and run_name.
mpirun \
    -n $NTASKS \
    -x OMP_NUM_THREADS \
    -x MKL_NUM_THREADS \
    -x OPENBLAS_NUM_THREADS \
    -x PYTHONPATH \
    python -u run.py nest $NTASKS "${PARAM_FILE:-param/defaults}" lgn_stepcurrentsource_noise_seed $NOISE_SEED "$RUN_NAME"