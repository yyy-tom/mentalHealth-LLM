#!/bin/bash
#SBATCH --job-name=tg-bot-counselor
#SBATCH --output=logs/slurm_bot_%j.out
#SBATCH --error=logs/slurm_bot_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00

# ============================================
# Telegram Bot — Mental Health Counselor
# ============================================
#
# Runs the Telegram bot as a long-running SLURM job.
# Only needs 1 GPU (inference only, no training).
#
# IMPORTANT: Compute nodes may not have outbound internet.
#   Test first: srun --gres=gpu:rtx2080:1 --time=00:05:00 \
#       curl -s https://api.telegram.org
#
# Usage:
#   export TELEGRAM_BOT_TOKEN="your-token-here"
#   sbatch slurm/telegram_bot.sh
#
#   # Or pass token inline:
#   TELEGRAM_BOT_TOKEN="xxx" sbatch --export=ALL slurm/telegram_bot.sh
#
# Monitor:
#   tail -f logs/slurm_bot_<jobid>.out
#   squeue -u $USER
#
# Stop:
#   scancel <jobid>
#
# ============================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# ---------- Validate token ----------
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "ERROR: TELEGRAM_BOT_TOKEN is not set."
    echo "Usage: TELEGRAM_BOT_TOKEN=\"xxx\" sbatch --export=ALL slurm/telegram_bot.sh"
    exit 1
fi

# ---------- Project setup ----------
PROJECT_DIR="/research/d7/fyp25/yyyu2/LLM"
cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

# Cache on research partition
export HF_HOME="/research/d7/fyp25/yyyu2/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" logs

# Single GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ---------- Network check ----------
echo ""
echo "Checking outbound internet connectivity..."
if curl -s --connect-timeout 10 https://api.telegram.org > /dev/null 2>&1; then
    echo "OK: Can reach api.telegram.org"
else
    echo "ERROR: Cannot reach api.telegram.org from this compute node."
    echo "HPC compute nodes typically block outbound internet."
    echo "Consider running on a GCP VM or login node instead."
    exit 1
fi

# ---------- GPU info ----------
echo ""
nvidia-smi
echo ""

# ---------- Model path ----------
MODEL_PATH="models/qwen2.5-7b-mental-health-fullft-a100"
BASE_MODEL="$MODEL_PATH"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# ---------- Run bot ----------
echo "Starting Telegram bot..."
echo "Model: $MODEL_PATH"
echo "To stop: scancel $SLURM_JOB_ID"
echo "========================================"

python scripts/telegram_bot.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL"

echo "========================================"
echo "Bot stopped at: $(date)"
echo "========================================"
