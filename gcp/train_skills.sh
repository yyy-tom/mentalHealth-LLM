#!/bin/bash
# ============================================================
# GCP: Train All Skill-Specific LoRA Adapters
# ============================================================
#
# Wrapper for training skill LoRA adapters on a GCP VM.
# Run gcp/setup_remote.sh and gcp/prep_training.sh FIRST.
#
# Usage (inside tmux on the VM):
#    bash gcp/train_skills.sh
#    bash gcp/train_skills.sh --skills "psychoeducation cbt-therapy"
#
# ============================================================

set -euo pipefail

echo "========================================"
echo "Skill LoRA Training — GCP VM"
echo "Start time: $(date)"
echo "========================================"

# ---------- Configuration ----------
PROJECT_DIR="/workspace/LLM"
cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }
source .venv/bin/activate

# ---------- GPU check ----------
echo ""
echo "[1/3] Checking GPUs..."
nvidia-smi
echo ""

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $GPU_COUNT GPU(s)"

# ---------- Environment ----------
echo ""
echo "[2/3] Setting environment..."

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" logs

# CUDA / NCCL settings
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---------- Verify base model ----------
BASE_MODEL="models/qwen2.5-7b-mental-health-fullft-a100"
if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Base model not found at $BASE_MODEL"
    echo "Please ensure the full-FT model is available."
    exit 1
fi

# ---------- Train ----------
echo ""
echo "[3/3] Starting skill adapter training..."
echo ""

bash scripts/train_all_skills.sh --hardware a100 "$@"

echo ""
echo "========================================"
echo "All skill adapters trained at: $(date)"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Download adapters: gcloud compute scp --recurse VM_NAME:~/LLM/adapters/ ./adapters/ --zone=ZONE"
echo "  2. Test: python scripts/skill_inference.py --interactive"
echo "  3. Stop VM: gcloud compute instances stop VM_NAME --zone=ZONE"
