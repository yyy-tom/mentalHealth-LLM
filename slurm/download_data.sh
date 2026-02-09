#!/bin/bash
#SBATCH --job-name=data-prep
#SBATCH --output=logs/data_prep_%j.out
#SBATCH --error=logs/data_prep_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 8
#SBATCH --time=01:00:00

# ============================================
# Dataset Download & Processing Job
#
# Downloads datasets from HuggingFace / Kaggle,
# processes them, combines, and pre-tokenizes.
#
# Run this BEFORE prep_training.sh
# ============================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "========================================"

# ------- Project setup -------
PROJECT_DIR="/research/d7/fyp25/yyyu2/LLM"
cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" logs

# ------- Run the pipeline -------
# Option A: Full pipeline (HuggingFace + Kaggle + combine + tokenize)
#   Requires: ~/.kaggle/kaggle.json
python3 scripts/download_and_process_datasets.py \
    --tokenize \
    --config configs/qwen_7b_8x2080ti.json

# Option B: Skip Kaggle download (if CSV was transferred manually)
# python3 scripts/download_and_process_datasets.py \
#     --suicide_csv datasets/kaggle_suicide_watch/Suicide_Detection.csv \
#     --tokenize \
#     --config configs/qwen_7b_8x2080ti.json

# Option C: Skip Kaggle entirely
# python3 scripts/download_and_process_datasets.py \
#     --skip_kaggle \
#     --tokenize \
#     --config configs/qwen_7b_8x2080ti.json

echo ""
echo "========================================"
echo "Data preparation complete!"
echo "End: $(date)"
echo "========================================"
echo ""
echo "Next step:"
echo "  sbatch slurm/prep_training.sh    # download model weights"
echo "  sbatch slurm/train_7b_8x2080ti.sh"
