# ksdetect

Knowledge Sufficiency Detection — internal-representation-based hallucination
/ knowledge-sufficiency detection research codebase, rebuilt per the project
implementation plan (`研究项目重建实施计划.md`). Read that document first for
the research process (stages 0-6); this README covers the engineering side
only.

If anything below is ambiguous or you're not sure a step applies to your
situation, STOP and ask before proceeding (project plan §2.1) — do not
guess and continue.

---

## 1. Environment setup

**Server:** AutoDL, project directory `/root/autodl-tmp/ksdetect`.
**Model:** LLaMA-3.1-8B-Instruct, fixed local path
`/autodl-fs/data/Llama-3.1-8B-Instruct` (do not change without adding a new
entry to `docs/decisions.md`).
**Connection:** PyCharm SSH Remote Interpreter, pointed at the conda
environment created below (not a local venv, not the system/base conda env).

### 1.1 Create the conda environment (on the server, via SSH terminal)

```bash
cd /root/autodl-tmp/ksdetect
conda env create -f environment.yml
conda activate ksdetect
```

If this is a re-run (environment already exists and you need to update it):

```bash
conda env update -f environment.yml --prune
```

**If `torch==2.4.1` (CUDA 12.1 build) fails to install or fails at runtime
with a CUDA version mismatch:** run `nvidia-smi` on the server, check the
CUDA version in the top-right corner, then find the matching torch wheel at
https://pytorch.org/get-started/previous-versions/ and install that
specific version instead — do NOT loosen the pin in `environment.yml` to a
version range (`>=`); replace it with the specific correct version and note
the change in `docs/decisions.md`.

### 1.2 Point PyCharm at this environment

1. PyCharm → Settings → Project → Python Interpreter → Add Interpreter →
   On SSH.
2. Enter the AutoDL SSH connection details (host/port/user — from the
   AutoDL instance page).
3. When asked for the interpreter path, use the conda env's python:
   `/root/miniconda3/envs/ksdetect/bin/python` (adjust the miniconda/
   anaconda path prefix if your AutoDL image differs — check with
   `which python` after `conda activate ksdetect`).
4. Set the remote project path to `/root/autodl-tmp/ksdetect`, sync.

### 1.3 Verify the model path

```bash
ls -la /autodl-fs/data/Llama-3.1-8B-Instruct
```

Should list the model's config/weight files. If this path doesn't exist or
is empty, stop here and resolve that before running anything — every
experiment config assumes this path is already correct and populated.

---

## 2. Directory structure

```
ksdetect/
├── core/                  # STABLE LAYER — see rule below
│   ├── extraction/        # model loading, forward hooks, batch extraction
│   ├── labeling/          # correctness judgment, hard/soft label construction
│   ├── stats/              # nested CV, multi-seed pooling, CI, significance tests
│   └── viz/                 # publication-grade plotting (English-only)
├── experiments/            # EXPLORATION LAYER — one folder per experiment
│   ├── _template/           # copy this to start a new experiment
│   └── exp_00_sanity_check/ # infrastructure validation, run this first
├── docs/
│   ├── known_issues.md      # bug log — check before debugging something familiar
│   └── decisions.md         # decisions that shouldn't be silently re-litigated
├── environment.yml
└── README.md                 # this file
```

**Non-negotiable rule:** any experiment script under `experiments/` may
ONLY call into `core/*` for cross-validation, feature selection,
significance testing, labeling, model loading, hook registration, or
plotting. If you find yourself about to write a `for` loop implementing
CV, or a plotting call with a Chinese string in it, or a raw
`model.forward()` with manual hook code, STOP — that logic belongs in
`core/`, not in an experiment script. This is what prevents the same bug
from needing to be fixed in five different places later.

---

## 3. Running an experiment

### 3.1 First run ever: the sanity check

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect

# Part 1 — no GPU needed, run this first
python experiments/exp_00_sanity_check/run_stats_sanity_check.py

# Part 2 — needs the GPU + real model
python experiments/exp_00_sanity_check/run_extraction_smoke_test.py
```

See `experiments/exp_00_sanity_check/README.md` for what each check means
and what to send back for review. **Do not proceed to any other experiment
until both parts report PASS.**

### 3.2 Any other experiment

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_<name>/run.py --config experiments/exp_<name>/config.yaml
```

Every experiment writes:
- `experiments/exp_<name>/run_log.txt` — full execution log
- `experiments/exp_<name>/results/summary.json` — machine-readable results
- `experiments/exp_<name>/results/*.pdf` — figures (English-only, per §4.5
  of the project plan)

---

## 4. Creating a new experiment

1. `cp -r experiments/_template experiments/exp_YYYYMMDD_short_name`
2. Fill in every `REPLACE_ME` field in the new `config.yaml` — the runner
   refuses to start if any are left in place.
3. Fill in the numbered `TODO` sections in `run.py` — these correspond to:
   dataset loading, model loading (via `core.extraction`, don't
   reimplement), activation extraction + shard saving, label construction
   (via `core.labeling`), statistics (via `core.stats`), plotting (via
   `core.viz`), and finally writing `results/summary.json` +
   `results/run_report.md`.
4. Before running: re-read the config against whatever was most recently
   discussed/decided (project plan §2.1's "确认最新要求已经完整对齐") — a
   stale config that technically runs without error is still wrong.
5. Run on a SMALL scale first (`n_examples` set to something small in
   config.yaml) before the full run (project plan §2.2).
6. Once the small run looks right, scale up. Before increasing batch
   size/concurrency, watch GPU memory / throughput on the small run first
   (project plan §2.2, concurrency caution).

---

## 5. Reporting results back for review

After any run — successful, partially successful, or failed — send back:

1. **Experiment path**, e.g. `experiments/exp_20260101_layer_scan/`
2. **Config used** (the file itself, or a diff if it's a small change from
   a previous run)
3. **Run status**: counts of completed / still-running / errored /
   not-yet-run tasks. If anything errored: the error message AND whether
   it's been fixed + re-run yet (project plan §2.4 — don't just report the
   count, report what's being done about it)
4. **Core results**: key numbers with their std/CI, not just point
   estimates
5. **Figure paths**
6. **Anything new and unexpected**, and whether it's been added to
   `docs/known_issues.md`

If a result looks surprising (e.g. an unexpectedly high/low AUC, an
unstable-looking curve), say so explicitly rather than only reporting the
final number — an anomaly should be investigated as a process question
first ("did something leak / did a fold get corrupted / is the sample size
too small"), not immediately attributed to "randomness" or "model
behavior" (project plan §2.4).

---

## 6. When something breaks

1. Don't just note that it failed and move on — find out why.
2. Check `docs/known_issues.md` first; the failure may already be
   documented with a known fix.
3. Once fixed: re-run whatever failed (don't leave gaps in a batch job),
   and if it's a new failure mode, add an entry to `docs/known_issues.md`
   in the same format as the existing entries.

---

## 7. Known issues / decisions

See `docs/known_issues.md` and `docs/decisions.md`. Skim both before
starting new work — they capture context that isn't repeated elsewhere.
