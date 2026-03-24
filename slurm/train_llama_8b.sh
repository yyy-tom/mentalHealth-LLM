#!/bin/bash
#SBATCH --job-name=llama8b-fullft
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00

# ============================================
# Llama-3.1-8B Training on 1x RTX 4090 (24GB)
# Uses QLoRA (4-bit quantization + LoRA)
# ============================================

echo "========================================"
echo "Llama-3.1-8B QLoRA Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

# Memory optimization for 24GB GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Llama is a gated model — ensure HF token is set
if [ -z "$HF_TOKEN" ] && [ ! -f "$HF_HOME/token" ]; then
    echo "ERROR: Llama is a gated model. Run 'huggingface-cli login' first."
    exit 1
fi

# Print GPU info
echo "========================================"
nvidia-smi
echo "========================================"

# Create logs directory
mkdir -p logs

# Run single-GPU QLoRA training
echo "Starting Llama-3.1-8B QLoRA training..."
python scripts/train_qwen_counsel.py \
    --config configs/llama_8b_4090.json

echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
