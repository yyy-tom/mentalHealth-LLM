#!/bin/bash
# ============================================================
# GCP: Llama-3.1-8B Full Fine-Tune on A100/H100 80GB
# ============================================================
#
# This script runs the actual training.
# Run gcp/setup_remote.sh and gcp/prep_training.sh FIRST.
#
# Usage (inside tmux on the VM):
#    bash gcp/train_llama_8b_fullft.sh
#
# For multi-GPU:
#    NUM_GPUS=8 CONFIG=configs/llama_8b_fullft_h100_8gpu.json bash gcp/train_llama_8b_fullft.sh
#
# ============================================================

set -euo pipefail

echo "========================================"
echo "Llama-3.1-8B Full Fine-Tune"
echo "Start time: $(date)"
echo "========================================"

# ---------- Configuration ----------
PROJECT_DIR="${PROJECT_DIR:-/workspace/LLM}"
CONFIG="${CONFIG:-configs/llama_8b_fullft_a100_1gpu_fast.json}"
NUM_GPUS="${NUM_GPUS:-1}"

cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }
source .venv/bin/activate

# ---------- Llama access check ----------
echo ""
echo "[0/3] Checking Llama model access..."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HF_HOME/token" ]; then
    echo "ERROR: Llama is a gated model. Run 'huggingface-cli login' first."
    echo "       Or set HF_TOKEN environment variable."
    exit 1
fi
echo "HF authentication: OK"

# ---------- GPU check ----------
echo ""
echo "[1/3] Checking GPUs..."
nvidia-smi
echo ""

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo "WARNING: Expected $NUM_GPUS GPUs but found $GPU_COUNT. Adjusting."
    NUM_GPUS=$GPU_COUNT
fi

# ---------- Environment ----------
echo ""
echo "[2/3] Setting environment..."

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" logs

# CUDA / NCCL settings for multi-GPU
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG"

# ---------- Verify data ready ----------
DATASET_PATH="datasets/all_mental_health_combined"
TOKENIZED_PATH="${DATASET_PATH}_tokenized"

if [ ! -d "$DATASET_PATH" ] && [ ! -d "$TOKENIZED_PATH" ]; then
    echo "ERROR: Dataset not found. Run gcp/prep_training.sh first."
    exit 1
fi

# ---------- Train ----------
echo ""
echo "[3/3] Starting training with torchrun ($NUM_GPUS GPUs)..."
echo ""

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29500 \
    scripts/train_qwen_fullft.py \
    --config "$CONFIG"

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Download model:  gcloud compute scp --recurse VM_NAME:~/LLM/models/llama-3.1-8b-mental-health-fullft-a100 ./models/ --zone=ZONE"
echo "  2. Stop VM:         gcloud compute instances stop VM_NAME --zone=ZONE"
