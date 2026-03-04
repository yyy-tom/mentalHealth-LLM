#!/bin/bash
#SBATCH --job-name=skill-lora
#SBATCH --output=logs/slurm_skills_%j.out
#SBATCH --error=logs/slurm_skills_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:rtx2080:7
#SBATCH --cpus-per-task=25
#SBATCH --time=72:00:00

# ============================================
# SLURM: Train Skill-Specific LoRA Adapters
# ============================================
#
# Trains all 6 skill adapters on 8x RTX 2080 Ti.
#
# Usage:
#    sbatch slurm/train_skills.sh
#
# ============================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
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

# CUDA settings — auto-detect from SLURM allocation (don't hardcode count)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Print GPU info
echo "========================================"
nvidia-smi
echo "========================================"

# Create logs directory
mkdir -p logs

# Run skill adapter training
echo "Starting skill adapter training..."
bash scripts/train_all_skills.sh --hardware 8x2080ti

echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
