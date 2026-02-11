#!/bin/bash
# ============================================================
# GCP: Download & Process Datasets
# ============================================================
#
# Downloads datasets from HuggingFace, processes and combines them.
# Run this AFTER setup_remote.sh and BEFORE prep_training.sh.
#
# Usage:
#    bash gcp/download_data.sh
#
# ============================================================

set -euo pipefail

echo "========================================"
echo "Dataset Download & Processing"
echo "Start time: $(date)"
echo "========================================"

PROJECT_DIR="$HOME/LLM"
cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# ---------- Download & process ----------
echo ""
echo "Running dataset pipeline..."

# Option A (default): Use synced Suicide Watch CSV
if [ -f "datasets/kaggle_suicide_watch/Suicide_Detection.csv" ]; then
    python3 scripts/download_and_process_datasets.py \
        --suicide_csv datasets/kaggle_suicide_watch/Suicide_Detection.csv
else
    # Option B: Skip Kaggle datasets
    echo "Suicide Watch CSV not found, skipping Kaggle datasets."
    python3 scripts/download_and_process_datasets.py --skip_kaggle
fi

echo ""
echo "========================================"
echo "Dataset download complete!"
echo "End time: $(date)"
echo "========================================"
echo ""
echo "Next: bash gcp/prep_training.sh"
