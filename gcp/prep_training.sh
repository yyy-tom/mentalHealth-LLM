#!/bin/bash
# ============================================================
# GCP: Pre-training Preparation
# ============================================================
#
# Downloads Qwen2.5-7B model weights and pre-tokenizes the dataset.
# Run this AFTER download_data.sh and BEFORE train_7b_fullft.sh.
#
# Usage:
#    bash gcp/prep_training.sh
#
# ============================================================

set -euo pipefail

echo "========================================"
echo "Pre-training Preparation"
echo "Start time: $(date)"
echo "========================================"

PROJECT_DIR="$HOME/LLM"
CONFIG="configs/qwen_7b_fullft_h100_8gpu.json"

cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# ---------- Step 1: Download model weights ----------
echo ""
echo "[1/2] Downloading Qwen2.5-7B-Instruct model weights..."

python3 -c "
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_name = 'Qwen/Qwen2.5-7B-Instruct'
print(f'Downloading {model_name}...')

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f'Tokenizer vocab size: {len(tokenizer):,}')

print('Downloading model weights (this may take a while)...')
snapshot_download(
    repo_id=model_name,
    local_dir=None,
    ignore_patterns=['*.md', '*.txt'],
)
print('Model weights cached successfully!')
"

# ---------- Step 2: Pre-tokenize dataset ----------
echo ""
echo "[2/2] Pre-tokenizing dataset..."

DATASET_PATH="datasets/all_mental_health_combined"
TOKENIZED_PATH="${DATASET_PATH}_tokenized"

if [ -d "$TOKENIZED_PATH" ]; then
    echo "Pre-tokenized dataset already exists at $TOKENIZED_PATH, skipping."
else
    if [ -d "$DATASET_PATH" ]; then
        echo "y" | python3 scripts/pretokenize_dataset.py --config "$CONFIG"
    else
        echo "WARNING: Dataset not found at $DATASET_PATH"
        echo "Run gcp/download_data.sh first."
        exit 1
    fi
fi

echo ""
echo "========================================"
echo "Preparation complete!"
echo "End time: $(date)"
echo "========================================"
echo ""
echo "Next: bash gcp/train_7b_fullft.sh"
