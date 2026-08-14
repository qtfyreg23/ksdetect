"""
core.stats — the ONLY place in this codebase allowed to implement cross-validation,
feature selection, or significance-testing logic.

Rule (see project plan §2.3 "强制规则"): every experiment script under experiments/
must call these functions instead of writing its own CV loop. This is the single
biggest defense against repeating the feature-selection leakage bug and the
single-seed noise problem documented in docs/known_issues.md (#2, #3).
"""

from .nested_cv import (
    make_stratified_folds,
    select_features,
    select_features_ranking,
    cross_validated_auc,
    cross_validated_auc_multi_k,
)
from .multiseed import (
    repeated_cv_auc,
    repeated_cv_auc_multi_k,
    matched_repeated_cv,
)
from .ci import (
    bootstrap_ci,
    ci_lower_bound_exceeds,
)
from .significance import (
    paired_wilcoxon,
    paired_ttest,
)

__all__ = [
    "make_stratified_folds",
    "select_features",
    "select_features_ranking",
    "cross_validated_auc",
    "cross_validated_auc_multi_k",
    "repeated_cv_auc",
    "repeated_cv_auc_multi_k",
    "matched_repeated_cv",
    "bootstrap_ci",
    "ci_lower_bound_exceeds",
    "paired_wilcoxon",
    "paired_ttest",
]
