#!/usr/bin/env bash
# Convenience wrapper for README.md §1.1. Run from the project root:
#   bash scripts/setup_env.sh
set -euo pipefail

ENV_NAME="ksdetect"
MODEL_PATH="/autodl-fs/data/Llama-3.1-8B-Instruct"

echo "=== Checking model path ==="
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: $MODEL_PATH does not exist or is not a directory."
    echo "Fix this before continuing — every experiment config assumes it's populated."
    exit 1
fi
echo "OK: $MODEL_PATH exists."

echo "=== Creating/updating conda environment '$ENV_NAME' ==="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists, updating..."
    conda env update -f environment.yml --prune
else
    conda env create -f environment.yml
fi

echo "=== Done. Activate with: conda activate $ENV_NAME ==="
echo "Next: run experiments/exp_00_sanity_check per README.md §3.1"
