# exp_00_sanity_check

Two independent checks, run separately, per project plan §4.7 ("退出标准").

## Part 1 — Stats sanity check (run this first, works anywhere, no GPU)

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_00_sanity_check/run_stats_sanity_check.py
```

What it checks, and why each one matters:

| Check | What it proves | Where it's used later |
|---|---|---|
| (a) strong signal, CI-lower > 0.75 | core.stats correctly detects a real signal | every AUC comparison in later experiments |
| (b) pure noise, AUC CI covers 0.5 | no systematic upward bias when there's nothing to find | rules out false positives in the wide scan |
| (c) selected(L1) vs random(k), paired Wilcoxon p<0.05 | selection actually finds something beyond chance | the "necessity of selection" ablation required by the project plan before trusting any sparsity claim |
| (d) leaky vs proper, leaky is inflated | this pipeline WOULD catch the exact feature-selection-leakage bug from the earlier project (docs/known_issues.md #3) if it reappeared | regression protection — run this after any change to core/stats |
| (e) labeling word-set-overlap cases | core.labeling.is_correct behaves as documented, not as strict substring match | every hard/soft label in every experiment |

**Expected result:** `overall_PASS: true` in `results/summary.json`, exit code 0.
If it fails, DO NOT proceed to any other experiment — fix core/stats or
core/labeling first, since everything downstream depends on them (project
plan §2.5, "已经实现过的功能应先找到原实现并复用" — this is the "原实现"
everything else calls).

Outputs: `results/summary.json`, `results/fig_leakage_check.pdf`,
`results/fig_selected_vs_random.pdf`, `run_log_stats.txt`.

## Part 2 — Extraction smoke test (server only, needs the real model)

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_00_sanity_check/run_extraction_smoke_test.py
```

Runs on the 3 prompts in `config.yaml`'s `smoke_test_prompts`, and checks:
  - the model loads from `MODEL_PATH` without hitting the network,
  - hooks fire on all four module types (embedding, attention, ffn_module,
    ffn_neuron, residual) and produce tensors of the expected shape,
  - `extract_batch` pooling ("last") gives one vector per prompt per
    layer/module with no NaNs.

Run this BEFORE any large-scale extraction job (project plan §2.2, "先在
少量案例上验证实现和实际行为,确认正确后再扩大规模"). If this fails, the
problem is almost certainly in `core/extraction/hooks.py`'s assumptions
about the transformers module structure (see that file's docstring) — check
the installed `transformers` version against the one pinned in
`environment.yml` first.

**When you run this, send me:**
1. The full console output (or `run_log_extraction.txt`)
2. Whether it printed `EXTRACTION SMOKE TEST: PASS` or `FAIL`
3. If FAIL: the full traceback

I'll confirm the shapes/values look right before you move on to Stage 1
(wide scan).
