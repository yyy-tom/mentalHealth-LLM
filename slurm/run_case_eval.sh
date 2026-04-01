#!/bin/bash
#SBATCH --job-name=case-eval
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00

# ============================================
# Case-Based Evaluation: 6 models x 10 cases
# Loads models one at a time, scores via DeepSeek
# ============================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Set project directory
PROJECT_DIR="/research/d7/fyp25/yyyu2/LLM"
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
source .venv/bin/activate

# Cache on research partition — NOT home dir (quota limit)
export HF_HOME="/research/d7/fyp25/yyyu2/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Print GPU info
echo "========================================"
nvidia-smi
echo "========================================"

# Create output directories
mkdir -p logs evaluation/case_results

# Run case evaluation (--resume skips completed models/cases)
echo "Starting case-based evaluation..."
python scripts/evaluation/run_case_eval.py \
    --cases evaluation/cases.json \
    --output-dir evaluation/case_results \
    --models qwen-ft qwen-base gemma-ft gemma-base mistral-ft mistral-base \
    --judge deepseek \
    --resume

echo "========================================"
echo "Evaluation completed at: $(date)"
echo "========================================"
