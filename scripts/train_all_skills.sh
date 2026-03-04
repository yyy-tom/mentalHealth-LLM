#!/bin/bash
# ============================================================
# Train All 6 Skill-Specific LoRA Adapters
# ============================================================
#
# Trains each skill adapter sequentially on top of the frozen
# full-FT base model. Prepares skill datasets first if needed.
#
# Usage:
#    bash scripts/train_all_skills.sh
#    bash scripts/train_all_skills.sh --hardware a100
#    bash scripts/train_all_skills.sh --hardware 4090
#    bash scripts/train_all_skills.sh --hardware 8x2080ti
#    bash scripts/train_all_skills.sh --skills "psychoeducation cbt-therapy"
#
# ============================================================

set -euo pipefail

echo "========================================"
echo "Skill-Specific LoRA Adapter Training"
echo "Start time: $(date)"
echo "========================================"

# ---------- Parse arguments ----------
HARDWARE="a100"
SKILLS="crisis-intervention general-support cbt-therapy empathetic-listening psychoeducation professional-counseling"
SKIP_DATA_PREP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --hardware)
            HARDWARE="$2"
            shift 2
            ;;
        --skills)
            SKILLS="$2"
            shift 2
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--hardware a100|4090|8x2080ti] [--skills \"skill1 skill2\"] [--skip-data-prep]"
            exit 1
            ;;
    esac
done

# ---------- Determine project directory ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Project: $PROJECT_DIR"
echo "Hardware: $HARDWARE"
echo "Skills: $SKILLS"

# ---------- Activate venv if available ----------
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# ---------- Step 1: Prepare skill datasets ----------
if [ "$SKIP_DATA_PREP" = false ]; then
    NEED_PREP=false
    for SKILL in $SKILLS; do
        if [ ! -d "datasets/skills/$SKILL" ]; then
            NEED_PREP=true
            break
        fi
    done

    if [ "$NEED_PREP" = true ]; then
        echo ""
        echo "[1/2] Preparing skill datasets..."
        python scripts/prepare_skill_datasets.py
    else
        echo ""
        echo "[1/2] Skill datasets already exist, skipping preparation."
    fi
else
    echo ""
    echo "[1/2] Skipping data preparation (--skip-data-prep)"
fi

# ---------- Step 2: Train each skill ----------
echo ""
echo "[2/2] Training adapters..."

TOTAL=0
SUCCESS=0
FAILED=0

for SKILL in $SKILLS; do
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "========================================"
    echo "Training: $SKILL ($TOTAL of $(echo $SKILLS | wc -w | tr -d ' '))"
    echo "Time: $(date)"
    echo "========================================"

    CONFIG="configs/skills/${SKILL}.json"
    if [ ! -f "$CONFIG" ]; then
        echo "WARNING: Config not found: $CONFIG — using defaults"
        CONFIG=""
    fi

    # Build command based on hardware
    CMD="python scripts/train_skill_lora.py --skill $SKILL"
    if [ -n "$CONFIG" ]; then
        CMD="$CMD --config $CONFIG"
    fi

    case $HARDWARE in
        a100|4090)
            # Single GPU — no quantization needed (24-80GB VRAM)
            if eval $CMD; then
                SUCCESS=$((SUCCESS + 1))
                echo "Completed: $SKILL"
            else
                FAILED=$((FAILED + 1))
                echo "FAILED: $SKILL"
            fi
            ;;
        8x2080ti)
            # Multi-GPU via torchrun on 11GB GPUs (10.57 GiB usable):
            #   - 4-bit quantization (7B model = ~3.5GB in NF4)
            #   - batch_size=1 (logits tensor = batch*seq*152K*4B, batch=4 OOMs)
            #   - grad_accum=32 (effective batch = 32 * NUM_GPUS)
            #   - max_length=256 to reduce activation memory
            #   - lora_r=16 to reduce trainable params memory
            # Auto-detect GPU count (SLURM may allocate fewer than 8)
            NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
            NUM_GPUS=${NUM_GPUS:-8}
            echo "Detected $NUM_GPUS GPUs for torchrun"
            TORCHRUN_CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 scripts/train_skill_lora.py --skill $SKILL --use-4bit --max-length 256 --lora-r 16 --batch-size 1 --grad-accum 32"
            if [ -n "$CONFIG" ]; then
                TORCHRUN_CMD="$TORCHRUN_CMD --config $CONFIG"
            fi
            if eval $TORCHRUN_CMD; then
                SUCCESS=$((SUCCESS + 1))
                echo "Completed: $SKILL"
            else
                FAILED=$((FAILED + 1))
                echo "FAILED: $SKILL"
            fi
            ;;
        *)
            echo "Unknown hardware: $HARDWARE"
            exit 1
            ;;
    esac
done

# ---------- Summary ----------
echo ""
echo "========================================"
echo "Training Complete"
echo "End time: $(date)"
echo "========================================"
echo "Total:   $TOTAL"
echo "Success: $SUCCESS"
echo "Failed:  $FAILED"
echo ""
echo "Adapters saved under: adapters/"
ls -la adapters/ 2>/dev/null || echo "(no adapters directory yet)"
