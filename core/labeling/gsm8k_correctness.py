"""
gsm8k_correctness.py — numeric exact-match for GSM8K, kept separate from
correctness.py's word-set matching on purpose.

Background: correctness.py's is_correct() explicitly does NOT do numeric
normalization ("forty-two" != "42") — that was a deliberate scope decision
so word-set matching doesn't grow silent numeric-parsing magic. GSM8K
answers are always a single final number (per the dataset's own "####
<number>" format), so it needs its own matcher, not a bolt-on to
correctness.py.
"""

from __future__ import annotations

import re


def extract_gsm8k_reference_number(answer_field: str) -> float | None:
    """
    GSM8K's `answer` field looks like:
        "Natalia sold ... <reasoning> ... #### 72"
    Returns the number after the LAST "####" marker as a float, or None if
    not found (should not happen on well-formed GSM8K data — if it does,
    treat it as a data-quality issue to investigate, not silently skip).
    """
    matches = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def extract_first_number(text: str) -> float | None:
    """
    Extracts the LAST number-looking token in `text` (models often restate
    intermediate numbers before the final answer; the final one is usually
    the model's actual answer). Returns None if no number found.
    Handles commas in numbers (e.g. "1,200") and simple decimals.
    """
    matches = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def gsm8k_is_correct(prediction: str, reference_answer_field: str, tol: float = 1e-4) -> bool:
    """
    prediction: the model's generated response text.
    reference_answer_field: GSM8K's raw `answer` column value (containing
    the "#### <number>" marker), NOT a pre-extracted number — this function
    does the extraction itself so the caller doesn't need a separate
    preprocessing step that could silently drift from this logic.

    Returns True iff the last number found in `prediction` matches the
    reference's "####"-marked number within `tol`. Returns False (not an
    exception) if either side has no extractable number — an unparseable
    prediction is a wrong answer, not a pipeline error, but IS worth
    tracking: if this happens often for a given model/prompt, that is a
    signal to look at raw generations, not a reason to change this
    function's behavior.
    """
    ref_num = extract_gsm8k_reference_number(reference_answer_field)
    if ref_num is None:
        raise ValueError(
            f"Could not find a '#### <number>' marker in reference answer: "
            f"{reference_answer_field!r} — this indicates malformed GSM8K "
            f"data, not a prediction-side issue."
        )
    pred_num = extract_first_number(prediction)
    if pred_num is None:
        return False
    return abs(pred_num - ref_num) < tol
