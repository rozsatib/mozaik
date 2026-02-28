cd /project

# 1. Set default offset if not provided (safe fallback)
if [ -z "$STIM_OFFSET" ]; then
    STIM_OFFSET=0
fi

# 2. Create a unique run name using the offset
RUN_NAME="50stim_15trial_${STIM_OFFSET}"

# 3. Construct the expected output directory name
# Mozaik typically constructs this as: ModelName_RunName_____
DIR_NAME="SelfSustainedPushPull_${RUN_NAME}_____"

echo "Running simulation with name: $RUN_NAME"
echo "Cleaning up directory: $DIR_NAME"

# 4. Remove the specific directory for THIS job only
rm -rf "$DIR_NAME"

echo "--- Starting Simulation (Internal MPI) ---"
echo "Host Thread Limit Check: OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Running with $NTASKS MPI Tasks"
echo "Stimulus Offset: $STIM_OFFSET, Stimulus Window: $STIM_WINDOW"

# 5. Run the python script passing the UNIQUE Run Name
mpirun \
    -n $NTASKS \
    -x OMP_NUM_THREADS \
    -x MKL_NUM_THREADS \
    -x OPENBLAS_NUM_THREADS \
    -x PYTHONPATH \
    python -u run.py nest $NTASKS param_MSA/defaults "$RUN_NAME"