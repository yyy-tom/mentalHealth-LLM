#!/bin/bash
#SBATCH --job-name=qwen7b-mental-health
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:rtx2080:8
#SBATCH --cpus-per-task=30
#SBATCH --time=72:00:00

# ============================================
# Qwen2.5-7B Training on 8x RTX 2080 Ti
# ============================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "========================================"

# Set project directory (adjust this path)
PROJECT_DIR="/research/d7/fyp25/yyyu2/LLM"
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
source .venv/bin/activate

# Cache on research partition — NOT home dir (quota limit)
export HF_HOME="/research/d7/fyp25/yyyu2/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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

# Run multi-GPU training with torchrun
echo "Starting training..."
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/train_qwen_counsel_multi_gpu.py \
    --config configs/qwen_7b_8x2080ti.json

echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
