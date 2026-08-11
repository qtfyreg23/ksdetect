"""
soft_label.py — turning a set of sampled generations (or a single greedy
generation) into a label usable by core.stats.

Two label types, matching the "标签类型" variable in the wide-scan design
(project plan §5.1):
  - hard_label_from_greedy: 0/1, based on a single greedy-decoded generation.
  - soft_label_from_samples: a float in [0, 1], the fraction of N
    temperature-sampled generations judged INCORRECT (i.e. higher = more
    "the model doesn't reliably know this"). This is deliberately defined as
    an error RATE, not a correctness rate, so that "higher soft label = more
    hallucination-prone" reads naturally in downstream analysis — keep this
    convention consistent everywhere the soft label is used.
"""

from __future__ import annotations

from typing import Callable

from .correctness import is_correct

# Signature every correctness function passed as `correctness_fn` must match:
# (prediction: str, references: list[str]) -> bool
CorrectnessFn = Callable[[str, list], bool]


def hard_label_from_greedy(
    prediction: str,
    references: list[str],
    correctness_fn: CorrectnessFn = is_correct,
) -> int:
    """
    1 if the greedy generation is INCORRECT (i.e. this is a hallucination /
    "doesn't know the answer" label), 0 if correct. Same "higher = more
    hallucination-prone" convention as soft_label_from_samples, so hard and
    soft labels can be compared/mixed without a sign flip.

    correctness_fn: defaults to core.labeling.correctness.is_correct
    (word-set matching). For GSM8K, pass a wrapper around
    core.labeling.gsm8k_is_correct instead — see
    core/data/loaders.py's GSM8K loader for the exact call pattern. Do NOT
    silently use word-set matching for GSM8K; it will not judge "42" ==
    "forty-two" correctly and was explicitly scoped to exclude numeric
    normalization (see correctness.py's docstring).
    """
    return int(not correctness_fn(prediction, references))


def soft_label_from_samples(
    predictions: list[str],
    references: list[str],
    correctness_fn: CorrectnessFn = is_correct,
) -> dict:
    """
    predictions: list of N temperature-sampled generations for the same
    question.
    references: the reference answer(s) for that question.

    Returns:
      {
        "soft_label": float,      # fraction of the N samples judged incorrect
        "n_samples": int,
        "n_correct": int,
        "n_incorrect": int,
        "per_sample_correct": list[bool],  # kept for auditing / spot checks
      }
    """
    if not predictions:
        raise ValueError("soft_label_from_samples requires at least 1 prediction")

    per_sample_correct = [correctness_fn(p, references) for p in predictions]
    n_correct = sum(per_sample_correct)
    n_total = len(predictions)
    n_incorrect = n_total - n_correct

    return {
        "soft_label": n_incorrect / n_total,
        "n_samples": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "per_sample_correct": per_sample_correct,
    }
