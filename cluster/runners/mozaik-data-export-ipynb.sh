cd /project

jupyter lab --allow-root --ip=0.0.0.0 --no-browser --port=8888 --NotebookApp.token='1234' --notebook-dir='/project'
# python export.py
# python compare_stimulus_AB.py
# python compute_psth.py
# python plot_psth_comparison.py
# echo "Running data export and comparison script..."
# python compare_test3_trials.py