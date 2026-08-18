"""
multiseed.py — repeated CV with multiple seeds, pooled (not averaged) into a
single distribution of fold-level AUCs.

Background (see docs/known_issues.md #5): a single 5-fold CV split gives only
5 AUC values, and at small sample sizes / small k this is noisy enough that
"which k first crosses the 90%-of-full-AUC threshold" is essentially random
depending on the fold seed. The fix used here — run N different seeds' worth
of 5-fold CV and pool all N*5 fold-level AUCs together — is exactly the
approach validated in the earlier project (Exp2.5 rework). Pooling raw
fold-level AUCs (not seed-level means) is what shrinks the confidence
interval; averaging into per-seed means first and then computing statistics
over those N means would throw away most of the added information.
"""

from __future__ import annotations

import numpy as np

from .nested_cv import cross_validated_auc, cross_validated_auc_multi_k, make_stratified_folds
from .ci import bootstrap_ci


def repeated_cv_auc_multi_k(
    X: np.ndarray,
    y: np.ndarray,
    k_values: list,
    selection_method: str = "l1",
    n_splits: int = 5,
    n_seeds: int = 10,
    base_seed: int = 0,
    on_seed_done=None,
) -> dict:
    """
    Like repeated_cv_auc(), but for MULTIPLE k values at once, built on
    cross_validated_auc_multi_k() so the (expensive, for high-dim modules)
    feature ranking is computed once per fold and shared across all
    requested k values instead of being recomputed per k — see
    docs/known_issues.md #9.

    on_seed_done: optional callback `on_seed_done(seed_idx, n_seeds,
    elapsed_seconds)` invoked after each seed's n_splits folds finish, for
    progress/heartbeat logging on cells that take a while (this is what
    exp_02b_analysis uses so a slow cell doesn't look like a silent hang).

    Returns {k: {"pooled_aucs":..., "mean":..., "std":..., "ci_lower":...,
    "ci_upper":..., "n_seeds":..., "n_splits":...} for k in k_values}.
    """
    import time
    pooled_by_k = {k: [] for k in k_values}

    for seed_offset in range(n_seeds):
        t0 = time.time()
        seed = base_seed + seed_offset
        per_k_fold_aucs = cross_validated_auc_multi_k(
            X, y, k_values=k_values, selection_method=selection_method,
            n_splits=n_splits, random_state=seed,
        )
        for k in k_values:
            pooled_by_k[k].append(per_k_fold_aucs[k])

        if on_seed_done is not None:
            on_seed_done(seed_offset + 1, n_seeds, time.time() - t0)

    output = {}
    for k in k_values:
        pooled = np.concatenate(pooled_by_k[k])
        ci_lower, ci_upper = bootstrap_ci(pooled)
        output[k] = {
            "pooled_aucs": pooled,
            "mean": float(pooled.mean()),
            "std": float(pooled.std(ddof=1)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_seeds": n_seeds,
            "n_splits": n_splits,
        }
    return output


def repeated_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    k: int | None = None,
    selection_method: str = "l1",
    n_splits: int = 5,
    n_seeds: int = 10,
    base_seed: int = 0,
) -> dict:
    """
    Run cross_validated_auc() once per seed in
    [base_seed, base_seed+1, ..., base_seed+n_seeds-1], each time with a fresh
    stratified fold split (different random_state -> different fold
    assignment), and pool all n_seeds * n_splits fold-level AUC values.

    Returns a dict:
      {
        "pooled_aucs": np.ndarray of shape (n_seeds * n_splits,),
        "mean": float,
        "std": float,
        "ci_lower": float,   # 95% bootstrap CI lower bound over pooled_aucs
        "ci_upper": float,
        "n_seeds": int,
        "n_splits": int,
      }
    """
    all_aucs = []
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        fold_aucs = cross_validated_auc(
            X, y, k=k, selection_method=selection_method,
            n_splits=n_splits, random_state=seed,
        )
        all_aucs.append(fold_aucs)

    pooled = np.concatenate(all_aucs)
    ci_lower, ci_upper = bootstrap_ci(pooled)

    return {
        "pooled_aucs": pooled,
        "mean": float(pooled.mean()),
        "std": float(pooled.std(ddof=1)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_seeds": n_seeds,
        "n_splits": n_splits,
    }


def matched_repeated_cv(
    feature_sets: dict[str, np.ndarray],
    y: np.ndarray,
    k: int | None = None,
    selection_method: str = "l1",
    n_splits: int = 5,
    n_seeds: int = 10,
    base_seed: int = 0,
) -> dict:
    """
    Like repeated_cv_auc(), but evaluates MULTIPLE named feature sets (e.g.
    {"ffn": X_ffn, "attention": X_attn, "residual": X_resid}) using the
    IDENTICAL fold assignment (same seed -> same StratifiedKFold split, since
    the split only depends on y and the seed, not on X) for every named set at
    every seed.

    This is what makes a paired significance test (paired_wilcoxon /
    paired_ttest in significance.py) valid afterward: pooled_aucs["ffn"][i]
    and pooled_aucs["residual"][i] come from evaluating both feature sets on
    the exact same held-out test samples, for every i. Comparing feature sets
    evaluated on DIFFERENT fold splits is a common mistake that silently
    invalidates a paired test — this function exists specifically to prevent
    that mistake structurally.

    Returns:
      {
        "pooled_aucs": {name: np.ndarray of shape (n_seeds * n_splits,) for name in feature_sets},
        "summary": {name: {"mean":..., "std":..., "ci_lower":..., "ci_upper":...} for name in feature_sets},
        "n_seeds": int,
        "n_splits": int,
      }
    """
    names = list(feature_sets.keys())
    pooled_by_name = {name: [] for name in names}

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        folds = make_stratified_folds(y, n_splits=n_splits, random_state=seed)
        for name in names:
            X = feature_sets[name]
            fold_aucs = cross_validated_auc(
                X, y, k=k, selection_method=selection_method,
                n_splits=n_splits, random_state=seed, folds=folds,
            )
            pooled_by_name[name].append(fold_aucs)

    pooled_aucs = {name: np.concatenate(vals) for name, vals in pooled_by_name.items()}
    summary = {}
    for name, arr in pooled_aucs.items():
        ci_lower, ci_upper = bootstrap_ci(arr)
        summary[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return {
        "pooled_aucs": pooled_aucs,
        "summary": summary,
        "n_seeds": n_seeds,
        "n_splits": n_splits,
    }
