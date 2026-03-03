cd /project
rm -r SelfSustainedPushPull_test:ntasks32_____

# python run_parameter_search.py run_spont.py nest param_MSA/defaults 
# python run_spont.py nest 2 param_MSA/defaults 'test'
# echo "Waiting for debugger attach..."
# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client run.py nest 2 param_MSA/defaults 'test:3'
# python -u run.py nest 32 param_MSA/defaults 'test:testntasks32'

# Check where processes are running (run this with mpirun if possible, or just run it once)
mpirun -n 32 bash -c "echo 'Rank \${PMIX_RANK} running on core:' \$(taskset -c -p \$\$)"
# mpirun --bind-to core -n 32 python -u run.py nest 32 param_MSA/defaults 'test:ntasks32' &
python -u run.py nest 32 param_MSA/defaults 'test:ntasks32' &
PID=$!

# Monitor loop (prints usage every 10 seconds)
for i in {1..10}; do
    sleep 10
    echo "--- System Status ---"
    uptime
    # Show top 5 CPU consuming processes
    ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -n 6
done

wait $PID