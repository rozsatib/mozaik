#!/usr/bin/env python3
"""REFACTOR-02 gate — capture the *effective launch behavior* of the cluster launchers WITHOUT running.

Two axes per scenario (see docs/plan/PLAN_REFACTOR_sbatch_consolidation.md):
  (A) scheduler directives — normalized key=value set of the #SBATCH lines (old launcher) or of the
      `sbatch ...` flags that `submit.sh --dry-run` would pass (new config+wrapper).
  (B) composed apptainer command — the full `apptainer exec ...` argv (env + binds + SIF + inner cmd),
      captured by putting a stub `apptainer` (and no-op `module`) on PATH and executing the launcher body.

Because old and new both dispatch through the SAME unchanged compose-*.sh, axis B is identical iff the
env the launcher exports before dispatch is identical — so capturing at the apptainer boundary is exact.

Usage:
  capture_gate.py --mode old --out  <golden.json>     # capture from the current run-*.sbatch launchers
  capture_gate.py --mode new --cmp  <golden.json>     # capture via submit.sh/run_job.sh, diff vs golden
Exit 0 iff (new) matches golden on every scenario.
"""
import argparse, json, os, re, shutil, stat, subprocess, sys, tempfile

MOZAIK = "/mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik"
CL = os.path.join(MOZAIK, "cluster")

# Representative array ids chosen to exercise the TRIAL/CHUNK arithmetic (prod: id/12, id%12).
SCENARIOS = [
    # label            workload  old_launcher              conf                       array_id ntasks
    ("sim-prod",       "sim",    "run-array.sbatch",       "sim-prod.conf",           13,      12),
    ("sim-test3",      "sim",    "run-test3.sbatch",       "sim-test3.conf",          1,       12),
    ("sim-test33",     "sim",    "run-test33.sbatch",      "sim-test33.conf",         35,      12),
    ("export-prod",    "export", "run-export.sbatch",      "export-prod.conf",        1,       1),
    ("export-test3",   "export", "run-export-test3.sbatch","export-test3.conf",       1,       1),
    ("export-test33",  "export", "run-export-test33.sbatch","export-test33.conf",     35,      1),
]

SHORT = {"-p": "partition", "-C": "constraint", "-N": "nodes", "-n": "ntasks", "-c": "cpus-per-task",
         "-t": "time", "-a": "array", "-A": "account", "-J": "job-name", "-o": "output", "-e": "error",
         "-G": "gres"}


def _strip_comment(s):
    # drop a trailing ' # ...' comment (SBATCH lines often annotate); keep '#' inside quotes untouched
    return re.sub(r"\s+#.*$", "", s).strip()


def _norm_tokens(tokens):
    """tokens: flat list like ['--array=0-2','-p','medium96s','--mem','250G'] -> {canonical: value}."""
    out = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            if "=" in t:
                k, v = t[2:].split("=", 1)
            else:
                k = t[2:]
                v = tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith("-") else ""
                if v:
                    i += 1
            out[k] = v.strip().strip('"').strip("'")
        elif t.startswith("-") and t in SHORT:
            k = SHORT[t]
            v = tokens[i + 1] if i + 1 < len(tokens) else ""
            i += 1
            out[k] = v.strip().strip('"').strip("'")
        i += 1
    return out


def directives_from_sbatch_file(path):
    toks = []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#SBATCH"):
            body = _strip_comment(line[len("#SBATCH"):].strip())
            toks += body.split()
    return _norm_tokens(toks)


def directives_from_dryrun(dry_line):
    # dry_line: the full 'sbatch <flags> ... cluster/run_job.sh' command
    toks = dry_line.split()
    assert toks and toks[0] == "sbatch", f"unexpected dry-run: {dry_line!r}"
    # drop the trailing script path and any --export=... (not a scheduler directive)
    flags = [t for t in toks[1:] if not t.endswith("run_job.sh") and not t.startswith("--export")]
    return _norm_tokens(flags)


def capture_apptainer(cmd, extra_env, array_id, ntasks):
    """Run `cmd` (a launcher body) with a stub apptainer and capture its exec argv WITHOUT running it.

    SAFETY: apptainer is already on PATH (the alloc's sbatch `module load apptainer`d it), and the
    launchers re-run `module load apptainer` which re-prepends the real binary — so a PATH-based stub
    loses and the REAL pipeline runs (the sim runner does `rm -rf $DIR_NAME` → data loss; this already
    cost trial1). Fix: stub `apptainer` as an exported bash *function*. Bash resolves functions before
    PATH, so the stub wins even after `module load apptainer`. `mpirun`/`srun`/`nest` are no-op
    functions too (defense in depth). A guard aborts unless the function is provably in effect, in the
    parent AND in a child shell (the launcher runs as `bash <file>`, a child).
    """
    fd, cap = tempfile.mkstemp(prefix="gate_cap_", suffix=".txt")
    os.close(fd)
    open(cap, "w").close()
    try:
        env = dict(os.environ)
        env["SLURM_ARRAY_TASK_ID"] = str(array_id)
        env["SLURM_NTASKS"] = str(ntasks)
        env["SLURM_JOB_ID"] = "GATE"
        env["SLURM_NODELIST"] = "gate-node"
        env["SLURM_SUBMIT_DIR"] = MOZAIK
        env["GATE_CAP"] = cap
        env.update(extra_env)
        prelude = (
            'apptainer() { { echo "=== APPTAINER ==="; for a in "$@"; do printf "%s\\n" "$a"; done; } >> "$GATE_CAP"; }; '
            'mpirun() { echo "STUB mpirun blocked" >&2; }; srun() { echo "STUB srun blocked" >&2; }; nest() { :; }; '
            'export -f apptainer mpirun srun nest; '
            # guard: function must win in THIS shell and in a child (launchers run as `bash <file>`)
            'if [ "$(type -t apptainer)" != "function" ]; then echo GUARD_FAIL_PARENT >&2; exit 77; fi; '
            'if [ "$(bash -c "type -t apptainer")" != "function" ]; then echo GUARD_FAIL_CHILD >&2; exit 77; fi; '
        )
        r = subprocess.run(["bash", "-c", prelude + cmd], env=env, cwd=MOZAIK,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
        if r.returncode == 77:
            sys.exit(f"ABORT: apptainer stub function not in effect — refusing to run (real-pipeline risk).\n{r.stderr.decode()}")
        argv = [l.rstrip("\n") for l in open(cap)]
        return argv or None
    finally:
        os.unlink(cap)


def capture_old(sc):
    label, workload, launcher, conf, aid, nt = sc
    path = os.path.join(CL, launcher)
    directives = directives_from_sbatch_file(path)
    argv = capture_apptainer(f"bash {path}", {}, aid, nt)
    return {"directives": directives, "apptainer_argv": argv}


def capture_new(sc):
    label, workload, launcher, conf, aid, nt = sc
    confpath = os.path.join(CL, "experiments", conf)
    dry = subprocess.run(["bash", os.path.join(CL, "submit.sh"), confpath, "--dry-run"],
                         capture_output=True, text=True, cwd=MOZAIK)
    dry_line = dry.stdout.strip().splitlines()[-1] if dry.stdout.strip() else ""
    directives = directives_from_dryrun(dry_line)
    argv = capture_apptainer(f"MOZAIK_CONF={confpath} bash {os.path.join(CL,'run_job.sh')}", {}, aid, nt)
    return {"directives": directives, "apptainer_argv": argv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "new"], required=True)
    ap.add_argument("--out")
    ap.add_argument("--cmp")
    args = ap.parse_args()

    result = {}
    for sc in SCENARIOS:
        result[sc[0]] = capture_old(sc) if args.mode == "old" else capture_new(sc)

    if args.out:
        json.dump(result, open(args.out, "w"), indent=2, sort_keys=True)
        print(f"wrote {args.out} ({len(result)} scenarios)")
        for lbl, r in result.items():
            n = len(r["apptainer_argv"]) if r["apptainer_argv"] else 0
            print(f"  {lbl:15} directives={len(r['directives'])} apptainer_argv_lines={n}")
        return 0

    if args.cmp:
        golden = json.load(open(args.cmp))
        ok = True
        for lbl in golden:
            g, n = golden[lbl], result.get(lbl)
            for axis in ("directives", "apptainer_argv"):
                if not n or g[axis] != n[axis]:
                    ok = False
                    print(f"DIFFER [{lbl}] {axis}:")
                    print(f"  golden: {json.dumps(g[axis])}")
                    print(f"  new   : {json.dumps(n[axis]) if n else None}")
            if n and g == n:
                print(f"MATCH   {lbl}")
        print("\nGATE:", "PASS — all scenarios byte-identical" if ok else "FAIL — launch behavior changed")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
