#!/bin/bash
#SBATCH --job-name=skill-lora
#SBATCH --output=logs/slurm_skill_%j.out
#SBATCH --error=logs/slurm_skill_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:rtx2080:7
#SBATCH --cpus-per-task=25
#SBATCH --time=72:00:00

# ============================================
# SLURM: Train ONE Skill-Specific LoRA Adapter
# ============================================
#
# Trains a single skill adapter per job to avoid
# CUDA memory fragmentation between sequential skills.
#
# Usage (single skill):
#   sbatch slurm/train_one_skill.sh crisis-intervention
#
# Usage (submit all failed skills):
#   for skill in crisis-intervention cbt-therapy empathetic-listening psychoeducation; do
#       sbatch slurm/train_one_skill.sh "$skill"
#   done
#
# ============================================

SKILL="${1:?Usage: sbatch slurm/train_one_skill.sh <skill-name>}"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Skill: $SKILL"
echo "Start time: $(date)"
echo "========================================"

# ---------- Project setup ----------
PROJECT_DIR="/research/d7/fyp25/yyyu2/LLM"
cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

# Cache on research partition
export HF_HOME="/research/d7/fyp25/yyyu2/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" logs

# CUDA settings
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
nvidia-smi
echo "========================================"

# ---------- Clean emergency checkpoints from prior failed runs ----------
if [ -d "adapters/$SKILL/emergency_checkpoint" ]; then
    echo "Removing emergency checkpoint from prior failed run..."
    rm -rf "adapters/$SKILL/emergency_checkpoint"
fi

# ---------- Config ----------
CONFIG="configs/skills/${SKILL}.json"
if [ ! -f "$CONFIG" ]; then
    echo "WARNING: Config not found: $CONFIG"
    CONFIG=""
fi

# ---------- Train ----------
echo "Training skill: $SKILL"
NUM_GPUS=$GPU_COUNT
echo "Detected $NUM_GPUS GPUs"

CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 scripts/train_skill_lora.py --skill $SKILL --use-4bit --max-length 256 --lora-r 16 --batch-size 1 --grad-accum 32"
if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
fi

echo "Command: $CMD"
if $CMD; then
    echo "========================================"
    echo "SUCCESS: $SKILL completed at $(date)"
    echo "========================================"
    ls -la "adapters/$SKILL/"
else
    echo "========================================"
    echo "FAILED: $SKILL at $(date)"
    echo "========================================"
    exit 1
fi
