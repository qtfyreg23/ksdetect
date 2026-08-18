"""
correctness.py — deciding whether a model's generated answer is "correct"
given one or more reference answers.

Background (see docs/known_issues.md #1): a strict substring match
(reference in prediction) systematically under-counts correct answers when
the model phrases things differently ("1,200" vs "1200", "New York City" vs
"NYC" is out of scope but "New York City" vs "new york" should count), and
over-counts when a short reference is a substring of an unrelated longer
prediction. The word-set matching approach here is the fix that was already
validated in the earlier project iteration — it is being reimplemented from
scratch here (per the "rebuild the code, not the conclusions" decision) but
the *approach* (word-set overlap, not the old code) is being reused
deliberately.
"""

from __future__ import annotations

import re
import string


_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer_tokens(text: str) -> set[str]:
    """
    Lowercase, strip punctuation, remove English articles, split on
    whitespace, and return the resulting token set.

    This is intentionally simple (no stemming, no synonym resolution) —
    anything more aggressive risks silently inflating the correctness rate.
    If a dataset needs numeric normalization (e.g. "1,200" == "1200"),
    handle it explicitly per-dataset in the calling code, not by adding
    hidden magic here.
    """
    text = text.lower().translate(_PUNCT_TABLE)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _ARTICLES]
    return set(tokens)


def is_correct(
    prediction: str,
    references: list[str],
    truncate_at_newline: bool = True,
    min_overlap_ratio: float = 0.5,
) -> bool:
    """
    Returns True iff `prediction` is judged correct against ANY of
    `references` (multi-reference datasets like TriviaQA / HotpotQA supply
    several acceptable answer strings).

    Matching rule (word-set overlap, not substring):
      1. If truncate_at_newline: cut `prediction` at the first newline, to
         avoid the model's continuation after the answer (e.g. rambling
         into unrelated hallucinated text) polluting the token set.
      2. Tokenize both prediction and each reference via
         normalize_answer_tokens().
      3. A reference is considered matched if:
           - the reference's token set is a subset of the prediction's token
             set (handles the model producing the correct answer plus extra
             wording), OR
           - the Jaccard-style overlap |ref ∩ pred| / |ref| is >=
             min_overlap_ratio AND |ref| > 0 (handles partial phrasing
             differences without requiring every reference token to appear).
      4. Returns True if any reference matches under rule 3.

    This function does NOT do any dataset-specific normalization (numbers,
    dates, aliases). If a dataset needs that, normalize `references` before
    calling this function and document the normalization in that
    experiment's config/README, not here.
    """
    if truncate_at_newline:
        prediction = prediction.split("\n")[0]

    pred_tokens = normalize_answer_tokens(prediction)
    if not pred_tokens:
        return False

    for ref in references:
        ref_tokens = normalize_answer_tokens(ref)
        if not ref_tokens:
            continue
        if ref_tokens.issubset(pred_tokens):
            return True
        overlap = len(ref_tokens & pred_tokens) / len(ref_tokens)
        if overlap >= min_overlap_ratio:
            return True

    return False
