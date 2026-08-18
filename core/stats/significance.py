"""
significance.py — paired significance tests for comparing two conditions
(e.g. FFN vs Residual, selected-k vs random-k) that were evaluated on
IDENTICAL held-out folds.

IMPORTANT: only pass in arrays produced by matched_repeated_cv() (or two
cross_validated_auc() calls that you manually verified used the exact same
`folds` argument). Passing in independently-generated AUC arrays (different
fold splits) makes the pairing meaningless and silently inflates or deflates
the test's power — see the docstring in multiseed.matched_repeated_cv for why.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon, ttest_rel


def paired_wilcoxon(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank test on paired (values_a[i], values_b[i]).
    Preferred over the paired t-test when normality of the differences is not
    established, which is usually the case for pooled AUC values from a
    handful of folds/seeds.

    Returns {"statistic": float, "p_value": float, "n_pairs": int,
             "median_diff": float}
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    if len(values_a) != len(values_b):
        raise ValueError(
            f"paired_wilcoxon requires equal-length arrays from matched folds, "
            f"got {len(values_a)} vs {len(values_b)}"
        )
    diff = values_a - values_b
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": len(diff),
                 "median_diff": 0.0}
    stat, p = wilcoxon(values_a, values_b)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_pairs": len(diff),
        "median_diff": float(np.median(diff)),
    }


def paired_ttest(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    """
    Paired t-test, provided as a secondary/sanity-check statistic alongside
    paired_wilcoxon (default). Report both if results disagree qualitatively.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    if len(values_a) != len(values_b):
        raise ValueError(
            f"paired_ttest requires equal-length arrays from matched folds, "
            f"got {len(values_a)} vs {len(values_b)}"
        )
    stat, p = ttest_rel(values_a, values_b)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_pairs": len(values_a),
        "mean_diff": float(np.mean(values_a - values_b)),
    }
