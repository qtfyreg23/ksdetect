"""
core.labeling — correctness judgment and label construction.

Rule: any experiment that needs to decide "was this generation correct" or
needs a soft/hard label must call these functions, not re-implement string
matching inline. See docs/known_issues.md #1 for why naive substring matching
is unsafe.
"""

from .correctness import is_correct, normalize_answer_tokens
from .soft_label import soft_label_from_samples, hard_label_from_greedy
from .gsm8k_correctness import (
    gsm8k_is_correct,
    extract_gsm8k_reference_number,
    extract_first_number,
)

__all__ = [
    "is_correct",
    "normalize_answer_tokens",
    "soft_label_from_samples",
    "hard_label_from_greedy",
    "gsm8k_is_correct",
    "extract_gsm8k_reference_number",
    "extract_first_number",
]
