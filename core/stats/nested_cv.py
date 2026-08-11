"""
nested_cv.py — leakage-safe feature selection + evaluation.

Background (see docs/known_issues.md #3): an earlier version of this project
selected the "top-k" FFN neurons using the FULL dataset (including samples that
were later used as the test fold), which produced an impossibly high AUC
(k=512 -> AUC 0.99+) that did not survive a proper re-implementation. The fix
enforced here is structural, not a matter of "being careful": feature selection
is a function argument that only ever receives the TRAINING portion of a fold,
never the full dataset and never the test portion.

Terminology mapping to the project's Chinese-language docs:
  - "嵌套交叉验证" -> the fact that select_features() is called separately inside
    each fold, using only that fold's training indices.
  - within a single call to cross_validated_auc(), there is no separate "inner"
    CV loop for hyperparameter tuning (kept deliberately simple for v0). If a
    future experiment needs inner-loop hyperparameter search, add it as a new
    keyword-argument-driven behavior here — do NOT reimplement CV elsewhere.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def make_stratified_folds(y: np.ndarray, n_splits: int = 5, random_state: int = 0):
    """
    Return a list of (train_idx, test_idx) tuples from a single StratifiedKFold
    split. Given the SAME y, n_splits, and random_state, this always returns the
    identical split — this determinism is what matched_repeated_cv() in
    multiseed.py relies on to produce paired samples for significance testing
    across different feature sets (e.g. FFN vs Residual) on IDENTICAL folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros_like(y), y))


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    method: str = "l1",
    random_state: int = 0,
) -> np.ndarray:
    """
    Select k feature indices using ONLY X_train / y_train.

    method:
      - "l1": fit an L1-penalized logistic regression on the training fold,
        rank features by |coefficient|, take the top-k. This is the default
        used for the sparsity-scan experiments (k=32/64/128/... style curves).
      - "variance": rank features by variance within X_train, take the top-k.
        Cheap heuristic baseline for the "necessity of selection" ablation.
      - "magnitude": rank features by mean absolute activation within X_train,
        take the top-k. Another cheap heuristic baseline.
      - "random": draw k indices uniformly at random (seeded by random_state).
        This is the REQUIRED comparison arm for the selected-vs-random
        ablation (project plan §7.2, Experiment 1 in the earlier research
        notes) — it must go through this same function so that it is subject
        to the exact same downstream evaluation code as every other method.

    Returns an array of k feature-column indices into X_train's second axis.
    """
    n_features = X_train.shape[1]
    k = min(k, n_features)
    rng = np.random.RandomState(random_state)

    if method == "random":
        return rng.choice(n_features, size=k, replace=False)

    if method == "variance":
        scores = X_train.var(axis=0)
        return np.argsort(scores)[::-1][:k]

    if method == "magnitude":
        scores = np.abs(X_train).mean(axis=0)
        return np.argsort(scores)[::-1][:k]

    if method == "l1":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        # C is intentionally on the smaller side to encourage sparsity in the
        # ranking; this is a RANKING device, not the final classifier used for
        # evaluation (that happens in cross_validated_auc with an unpenalized
        # or lightly-penalized model on the selected subset).
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5, random_state=random_state,
            max_iter=2000,
        )
        clf.fit(X_scaled, y_train)
        coefs = np.abs(clf.coef_).ravel()
        return np.argsort(coefs)[::-1][:k]

    raise ValueError(f"Unknown selection method: {method!r}")


def cross_validated_auc(
    X: np.ndarray,
    y: np.ndarray,
    k: int | None = None,
    selection_method: str = "l1",
    n_splits: int = 5,
    random_state: int = 0,
    folds: list | None = None,
) -> np.ndarray:
    """
    Run one stratified K-fold CV pass. For each fold:
      1. Split into train/test (uses `folds` if given, else derives them via
         make_stratified_folds(y, n_splits, random_state) internally).
      2. If k is not None: call select_features() using ONLY the training
         portion, obtain a feature subset.
      3. Standardize features (fit on train only) and fit a plain logistic
         regression classifier on the (possibly-subsetted) training portion.
      4. Predict probabilities on the held-out test portion, compute AUC.

    Returns a 1-D numpy array of length n_splits containing one AUC per fold.
    This is a SINGLE random split's worth of AUCs — for the multi-seed pooling
    described in the project plan (needed because single-split fold variance
    can be large at small k / small sample sizes), use repeated_cv_auc() in
    multiseed.py instead of calling this function directly in an experiment
    script.
    """
    if folds is None:
        folds = make_stratified_folds(y, n_splits=n_splits, random_state=random_state)

    fold_aucs = np.zeros(len(folds))
    for i, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if k is not None:
            feat_idx = select_features(
                X_train, y_train, k=k, method=selection_method,
                random_state=random_state,
            )
            X_train = X_train[:, feat_idx]
            X_test = X_test[:, feat_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, random_state=random_state)
        clf.fit(X_train_scaled, y_train)
        probs = clf.predict_proba(X_test_scaled)[:, 1]
        fold_aucs[i] = roc_auc_score(y_test, probs)

    return fold_aucs


def cross_validated_auc_LEAKY_FOR_TESTING_ONLY(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    selection_method: str = "l1",
    n_splits: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    """
    DELIBERATELY LEAKY implementation kept ONLY as a regression-test fixture
    (see experiments/exp_00_sanity_check) that reproduces the exact bug
    described in docs/known_issues.md #3: feature selection is performed on
    the FULL dataset (X, y) before the train/test split, so the test fold's
    labels influence which features are chosen.

    DO NOT call this from any experiment script. Its only purpose is to prove,
    once, that our sanity-check pipeline is capable of detecting this failure
    mode when it happens — i.e. this function is used to test the tests.
    """
    feat_idx = select_features(X, y, k=k, method=selection_method, random_state=random_state)
    X_sub = X[:, feat_idx]
    return cross_validated_auc(
        X_sub, y, k=None, n_splits=n_splits, random_state=random_state,
    )
