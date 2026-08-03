# `cluster/` — launching MOZAIK jobs

One launcher, config-driven. Each experiment is a human-readable file in `experiments/*.conf`; the
`submit.sh` wrapper reads it and submits the bodiless `run_job.sh` with the right `sbatch` directives.

## Launch

```bash
cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

./cluster/submit.sh cluster/experiments/sim-test3.conf        # 3-trial sim smoke test
./cluster/submit.sh cluster/experiments/sim-test33.conf       # 31-trial sim
./cluster/submit.sh cluster/experiments/sim-prod.conf         # production sim
./cluster/submit.sh cluster/experiments/export-test3.conf     # export test3 datastores
./cluster/submit.sh cluster/experiments/export-test33.conf    # export test33
./cluster/submit.sh cluster/experiments/export-prod.conf      # production export (screen-only)

./cluster/submit.sh cluster/experiments/sim-test3.conf --dry-run   # print the sbatch command, don't submit
```

Add a new experiment = copy the closest `.conf`, edit the values, submit. No new launcher script needed.

## Config layout (rule)
The six standard pipeline configs (`sim-{prod,test3,test33}.conf`, `export-{…}.conf`) live **loose**
in `experiments/` — their paths are pinned by the golden gate and this README, so don't move them.
**Any config set created for a specific experiment/study goes in its own subfolder**
`experiments/<experiment-name>/` (e.g. `experiments/ntasks-scaling/sim-scale-nt*.conf`) so the top
level stays uncluttered. `submit.sh` takes a path, so nested confs just work:
`./cluster/submit.sh cluster/experiments/ntasks-scaling/sim-scale-nt12.conf`.

## Config keys
- **Scheduler:** `WORKLOAD` (sim|export), `PARTITION CONSTRAINT NTASKS MEM TIME ARRAY JOB_NAME OUTPUT ERROR`.
  Constant defaults (`ACCOUNT nix00014`, `NODES 1`, `CPUS_PER_TASK 4`, `HINT nomultithread`, mail) live
  in `submit.sh`; override in a config only if an experiment needs to differ.
- **Runtime (sim):** `N_CHUNKS` (TRIAL = array_id / N_CHUNKS, CHUNK = array_id % N_CHUNKS), optional `CHUNK_DIR`.
- **Runtime (export):** `CHUNK_START CHUNK_END BATCH_SIZE N_CHUNKS CHUNK_DIR OUTPUT_PREFIX DATASTORE_PREFIX
  EXPORT_MODE` — any omitted key falls back to the compose-script default.

## Call chain (unchanged below submit.sh)
`submit.sh <conf>` → `sbatch … run_job.sh` → `apptainer-compose-{array,export}.sh` →
`runners/mozaik-{simulation,export}-array.sh` → `run.py` / `export.py`.

## Gate (behavior-preserving check)
The consolidation is verified byte-identical to the old `run-*.sbatch` launchers by
`_gate/capture_gate.py` against `docs/plan/audit/golden/P1_launch.json` (scheduler directives + the
composed `apptainer exec` command, captured with a stub apptainer — never runs the real pipeline).
Re-run after any change here:
```bash
python3 cluster/_gate/capture_gate.py --mode new --cmp ../../docs/plan/audit/golden/P1_launch.json
```

## Legacy
`legacy/` holds retired launchers (interactive-Jupyter `run*.sbatch` via the old `compose.sh`, and the
non-apptainer `pyenv-mpi-direct.sbatch`). Kept for reference; not part of the batch pipeline.

## PSTH
`run-psth-datastore.sbatch` is a standalone one-off (its own `apptainer exec`, no compose) and is left as-is.
