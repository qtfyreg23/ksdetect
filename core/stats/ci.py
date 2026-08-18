"""
ci.py — confidence intervals, and the "CI-lower-bound vs point-estimate"
decision rule.

Background (see docs/known_issues.md #5): deciding "has sparsity level k
reached 90% of full-representation AUC" by comparing a point-estimate AUC
against the 90% threshold is fragile — a point estimate that clears the
threshold by an amount smaller than its own standard deviation is
indistinguishable from noise. The rule enforced here is: only accept "k has
reached the threshold" when the CI LOWER BOUND clears the threshold, not the
point estimate.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_boot: int = 10000,
    random_state: int = 0,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for the mean of `values`.

    Returns (lower, upper). With small `values` (e.g. n < 10) this CI will be
    wide, which is the correct (honest) behavior — do not shrink n_boot to
    make intervals look tighter.
    """
    rng = np.random.RandomState(random_state)
    values = np.asarray(values)
    n = len(values)
    if n < 2:
        raise ValueError("bootstrap_ci requires at least 2 values")

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def ci_lower_bound_exceeds(
    values: np.ndarray,
    threshold: float,
    confidence: float = 0.95,
    n_boot: int = 10000,
    random_state: int = 0,
) -> dict:
    """
    Implements the project's required decision rule: a quantity is only
    considered to have "reached" `threshold` if the CI LOWER bound exceeds
    it — not the point estimate (mean).

    Returns:
      {
        "passed": bool,          # True iff ci_lower > threshold
        "mean": float,
        "ci_lower": float,
        "ci_upper": float,
        "margin_in_stds": float, # (mean - threshold) / std — report this
                                  # alongside "passed" so a near-miss is
                                  # visible even when passed=False, and a
                                  # narrow pass is visible even when
                                  # passed=True.
      }
    """
    values = np.asarray(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    ci_lower, ci_upper = bootstrap_ci(
        values, confidence=confidence, n_boot=n_boot, random_state=random_state,
    )
    margin_in_stds = (mean - threshold) / std if std and std > 0 else float("nan")
    return {
        "passed": ci_lower > threshold,
        "mean": mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin_in_stds": margin_in_stds,
    }
