# Known Issues Log

Append-only log of bugs, gotchas, and their fixes. Per project plan §2.4
("每次犯过的错误都应记录下来"), check this file before debugging something
that looks familiar, and add a new numbered entry whenever you fix a
non-trivial bug — even if it seems "obvious" in hindsight.

Format per entry: what happened, how it was found, the fix, and which
file(s) now encode the fix so it can't silently regress.

---

### #1 — `datasets` library breaking change broke `hotpot_qa` loading
**Inherited from the earlier project iteration** (recorded here as
background, not yet re-verified against this codebase — re-verify the first
time `hotpot_qa` is actually loaded in `ksdetect`).
`datasets>=4.0.0` introduced a change that broke the `hotpot_qa` loading
script used previously. Fix: `environment.yml` pins `datasets==2.21.0`
explicitly rather than `datasets>=X`.
**Encoded in:** `environment.yml`.

### #2 — Single-seed CV gives noisy, unstable conclusions
**Inherited from the earlier project iteration.** A single 5-fold CV split
produces only 5 AUC values; at small k / small sample sizes, "which k first
crosses a threshold" was effectively determined by which fold seed was
used, not by the underlying signal. Fix: always pool fold-level AUCs across
many seeds (`core.stats.repeated_cv_auc` / `matched_repeated_cv`), and judge
threshold-crossing by the CI LOWER bound, not the point estimate
(`core.stats.ci_lower_bound_exceeds`).
**Encoded in:** `core/stats/multiseed.py`, `core/stats/ci.py`.

### #3 — Feature selection performed on the full dataset before CV split (leakage)
**Inherited from the earlier project iteration.** An earlier version
selected the "top-k" neurons using ALL samples (including ones that ended
up in the test fold), which produced impossibly high AUC (k=512 -> AUC
0.99+) that vanished once selection was correctly confined to each fold's
training portion only. Fix: `core.stats.nested_cv.select_features()` only
ever receives the training-fold data; a deliberately-leaky version
(`cross_validated_auc_LEAKY_FOR_TESTING_ONLY`) is kept ONLY as a regression
test fixture in `experiments/exp_00_sanity_check`, never called from real
experiment code.
**Encoded in:** `core/stats/nested_cv.py`; regression-tested in
`experiments/exp_00_sanity_check/run_stats_sanity_check.py` check (d).

### #4 — HuggingFace endpoint / network configuration issues on AutoDL
**Inherited from the earlier project iteration.** Loading the model via the
HF Hub API on the AutoDL instance ran into endpoint/network configuration
problems. Fix: always load from the local path
(`/autodl-fs/data/Llama-3.1-8B-Instruct`) with `local_files_only=True`,
never fall back to a Hub download.
**Encoded in:** `core/extraction/model_loader.py`.

### #5 — Naive substring matching under/over-counts correct answers
**Inherited from the earlier project iteration.** Strict `reference in
prediction` matching missed correct answers phrased differently and
falsely matched unrelated short references. Fix: word-set overlap matching
in `core.labeling.correctness.is_correct`, with explicit non-goals
documented (no numeric normalization, no synonym resolution) so the
matching logic doesn't silently grow scope creep.
**Encoded in:** `core/labeling/correctness.py`.

---

### #6 — （沙箱环境限制，非真实bug）core/data/loaders.py 尚未在真实数据上跑通
2026-08-11。编写`core/data`时沙箱环境无法访问huggingface.co,所有5个数据
集加载器的字段名是对照HF数据集页面的文档/预览手动核实的,但没有实际执行过
`datasets.load_dataset(...)`。已建立`experiments/exp_01_data_check`作为服
务器端的验证步骤,必须先跑通并确认字段内容合理,再在此基础上写完整的抽取
流水线。若跑的时候CoQA的`trust_remote_code=True`触发了非预期行为或报错,
在这里补充一条新记录说明具体报错和解决方式。

<!-- New entries go below this line. Use the same format: what happened,
how found, the fix, and which file encodes it. -->
