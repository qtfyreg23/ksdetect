"""
core.extraction — model loading and activation extraction.

Rule: any experiment that needs hidden states / MLP / attention activations
must go through load_model() + register_hooks() + extract_batch() here. Do
NOT write ad-hoc forward-hook code inside an experiments/ script — if the
hook logic needs to change (e.g. to add a new signal source), change it here
so every experiment benefits and stays consistent.

NOTE ON TESTABILITY: this module requires torch + transformers + the actual
model weights at MODEL_PATH, none of which are available in the sandbox this
code was authored in. It has been reviewed for correctness against the
transformers LlamaForCausalLM module structure but has NOT been executed.
The first thing to run on the AutoDL server is
experiments/exp_00_sanity_check/run_extraction_smoke_test.py (see that
experiment's README) — a tiny smoke test on 2-3 examples — BEFORE any
large-scale extraction, per project plan §2.2 ("先在少量案例上验证").
"""

from .model_loader import load_model_and_tokenizer, MODEL_PATH
from .hooks import ActivationCollector, SUPPORTED_MODULES
from .extract import extract_batch, save_activation_shard, load_activation, compute_prompt_token_lengths
from .generate import generate_greedy_batch, generate_samples_batch, DEFAULT_MAX_NEW_TOKENS

__all__ = [
    "load_model_and_tokenizer",
    "MODEL_PATH",
    "ActivationCollector",
    "SUPPORTED_MODULES",
    "extract_batch",
    "save_activation_shard",
    "load_activation",
    "compute_prompt_token_lengths",
    "generate_greedy_batch",
    "generate_samples_batch",
    "DEFAULT_MAX_NEW_TOKENS",
]
