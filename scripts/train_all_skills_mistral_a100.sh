#!/bin/bash
# ============================================================
# Full Pipeline: Download Data → Prepare Skill Datasets → Train (Mistral 7B)
# ============================================================
#
# Self-contained script for 1x A100 80GB. Uses the full-finetuned
# Mistral 7B model as the base and trains all 5 skill LoRA adapters
# sequentially.
#
# Usage:
#    bash scripts/train_all_skills_mistral_a100.sh
#    bash scripts/train_all_skills_mistral_a100.sh --skills "cbt-therapy psychoeducation"
#    bash scripts/train_all_skills_mistral_a100.sh --skip-download   # datasets already exist
#    bash scripts/train_all_skills_mistral_a100.sh --skip-train      # only download + prepare
#    bash scripts/train_all_skills_mistral_a100.sh --base-model /path/to/base/model
#
# ============================================================

set -euo pipefail

# ---------- Parse arguments ----------
SKILLS="crisis-intervention cbt-therapy empathetic-listening psychoeducation professional-counseling"
BASE_MODEL="models/mistral-7b-mental-health-fullft-a100"
ADAPTER_PREFIX="adapters-mistral"
SKIP_DOWNLOAD=false
SKIP_TRAIN=false
CLEAN_FAILED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skills)
            SKILLS="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --clean-failed)
            CLEAN_FAILED=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skills \"skill1 skill2\"] [--base-model PATH] [--skip-download] [--skip-train] [--clean-failed]"
            exit 1
            ;;
    esac
done

# ---------- Setup ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DATASETS_DIR="$PROJECT_DIR/datasets"
mkdir -p "$DATASETS_DIR"
mkdir -p "$ADAPTER_PREFIX"

echo "========================================================"
echo "  Skill LoRA Training Pipeline — Mistral 7B (A100 80GB)"
echo "  Start: $(date)"
echo "========================================================"
echo "  Project:    $PROJECT_DIR"
echo "  Datasets:   $DATASETS_DIR"
echo "  Base model: $BASE_MODEL"
echo "  Adapters:   $ADAPTER_PREFIX/"
echo "  Skills:     $SKILLS"
echo ""

# Resolve base model path (relative → absolute)
if [[ "$BASE_MODEL" != /* ]]; then
    RESOLVED_MODEL="$PROJECT_DIR/$BASE_MODEL"
else
    RESOLVED_MODEL="$BASE_MODEL"
fi

# Activate venv if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# ---------- Clean up failed adapters ----------
if [ "$CLEAN_FAILED" = true ]; then
    echo ""
    echo "[0] Cleaning up failed/partial adapters..."
    for SKILL in $SKILLS; do
        ADAPTER_DIR="$ADAPTER_PREFIX/$SKILL"
        if [ -d "$ADAPTER_DIR" ]; then
            # Keep if it has a final adapter_model file
            if [ -f "$ADAPTER_DIR/adapter_model.safetensors" ] || [ -f "$ADAPTER_DIR/adapter_model.bin" ]; then
                echo "  $SKILL: has final adapter, keeping"
            else
                echo "  $SKILL: removing incomplete adapter dir"
                rm -rf "$ADAPTER_DIR"
            fi
        fi
    done
fi

# ==========================================================
# PHASE 1: Download and process source datasets
# ==========================================================
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "========================================================"
    echo "  PHASE 1: Download & Process Source Datasets"
    echo "========================================================"

    # --- 1a. Cactus (CBT therapy) ---
    if [ ! -d "$DATASETS_DIR/cactus_processed/train" ]; then
        echo ""
        echo "[1/6] Downloading & processing Cactus (CBT)..."
        uv run python3 scripts/data/prepare_cactus_dataset.py \
            --output_dir "$DATASETS_DIR/cactus_processed" \
            || echo "WARNING: Cactus dataset preparation failed"
    else
        echo "[1/6] cactus_processed already exists, skipping"
    fi

    # --- 1b. ESConv (Empathetic listening) ---
    if [ ! -d "$DATASETS_DIR/esconv_processed/train" ]; then
        echo ""
        echo "[2/6] Downloading & processing ESConv (Empathetic)..."
        uv run python3 scripts/data/prepare_esconv_dataset.py \
            --output_dir "$DATASETS_DIR/esconv_processed" \
            || echo "WARNING: ESConv dataset preparation failed"
    else
        echo "[2/6] esconv_processed already exists, skipping"
    fi

    # --- 1c. MentalChat16K (Psychoeducation) ---
    if [ ! -d "$DATASETS_DIR/mentalchat16k_processed/train" ]; then
        echo ""
        echo "[3/6] Downloading & processing MentalChat16K (Psychoeducation)..."
        uv run python3 scripts/data/prepare_mentalchat16k_dataset.py \
            --output_dir "$DATASETS_DIR/mentalchat16k_processed" \
            || echo "WARNING: MentalChat16K dataset preparation failed"
    else
        echo "[3/6] mentalchat16k_processed already exists, skipping"
    fi

    # --- 1d. Counsel Chat (Professional counseling) ---
    if [ ! -d "$DATASETS_DIR/counsel_chat_processed/train" ]; then
        echo ""
        echo "[4/6] Downloading & processing Counsel Chat..."
        uv run python3 scripts/data/download_missing_datasets.py  \
            || echo "WARNING: Counsel Chat download failed"
    else
        echo "[4/6] counsel_chat_processed already exists, skipping"
    fi

    # --- 1e. AMOD (Professional counseling) ---
    if [ ! -d "$DATASETS_DIR/amod_processed/train" ]; then
        echo ""
        echo "[5/6] Downloading & processing AMOD..."
        uv run  python3 scripts/data/prepare_amod_dataset.py \
            --output_dir "$DATASETS_DIR/amod_processed" \
            || echo "WARNING: AMOD dataset preparation failed"
    else
        echo "[5/6] amod_processed already exists, skipping"
    fi

    # --- 1f. Crisis detection (Kaggle suicide watch) ---
    if [ ! -d "$DATASETS_DIR/crisis_detection_processed/train" ]; then
        echo ""
        echo "[6/6] Processing crisis detection dataset..."
        # This needs the kaggle CSV — check if it exists
        if [ -f "$DATASETS_DIR/kaggle_suicide_watch/Suicide_Detection.csv" ]; then
            uv run python3 scripts/data/download_missing_datasets.py \
                || echo "WARNING: Crisis detection processing failed"
        else
            echo "  Kaggle CSV not found, attempting full download..."
            uv run python3 scripts/data/download_missing_datasets.py \
                || echo "WARNING: Crisis detection download failed"
        fi
    else
        echo "[6/6] crisis_detection_processed already exists, skipping"
    fi

    echo ""
    echo "--- Source dataset status ---"
    for DS in cactus_processed esconv_processed mentalchat16k_processed \
              counsel_chat_processed amod_processed crisis_detection_processed; do
        if [ -d "$DATASETS_DIR/$DS/train" ]; then
            echo "  $DS: OK"
        else
            echo "  $DS: MISSING"
        fi
    done
else
    echo ""
    echo "PHASE 1: Skipped (--skip-download)"
fi

# ==========================================================
# PHASE 2: Prepare skill-specific datasets
# ==========================================================
echo ""
echo "========================================================"
echo "  PHASE 2: Prepare Skill-Specific Datasets"
echo "========================================================"

NEED_PREP=false
for SKILL in $SKILLS; do
    if [ ! -d "$DATASETS_DIR/skills/$SKILL/train" ]; then
        NEED_PREP=true
        break
    fi
done

if [ "$NEED_PREP" = true ]; then
    echo "Running prepare_skill_datasets.py..."
    uv run python3 scripts/prepare_skill_datasets.py --output_dir "$DATASETS_DIR/skills"
else
    echo "All skill datasets already exist, skipping."
fi

echo ""
echo "--- Skill dataset status ---"
for SKILL in $SKILLS; do
    if [ -d "$DATASETS_DIR/skills/$SKILL/train" ]; then
        echo "  $SKILL: OK"
    else
        echo "  $SKILL: MISSING (training will fail for this skill)"
    fi
done

# ==========================================================
# PHASE 3: Train LoRA adapters
# ==========================================================
if [ "$SKIP_TRAIN" = true ]; then
    echo ""
    echo "PHASE 3: Skipped (--skip-train)"
    echo ""
    echo "To train manually:"
    echo "  python scripts/train_skill_lora.py --skill <skill_name> --base_model $BASE_MODEL --output_dir $ADAPTER_PREFIX/<skill_name>"
    exit 0
fi

# Pre-flight: verify base model exists (skip check for HuggingFace model IDs)
if [[ "$BASE_MODEL" == */* && "$BASE_MODEL" != /* ]]; then
    # Looks like a HuggingFace model ID (e.g. mistralai/Mistral-7B-Instruct-v0.3)
    # Check if it also exists as a local path; if not, assume HF download
    if [ -d "$RESOLVED_MODEL" ]; then
        echo "Base model (local): $RESOLVED_MODEL"
    else
        echo "Base model (HuggingFace): $BASE_MODEL (will download on first use)"
        RESOLVED_MODEL="$BASE_MODEL"
    fi
elif [ ! -d "$RESOLVED_MODEL" ]; then
    echo ""
    echo "ERROR: Base model not found at: $RESOLVED_MODEL"
    echo ""
    echo "Options:"
    echo "  1. Use a HuggingFace model:"
    echo "       bash $0 --base-model mistralai/Mistral-7B-Instruct-v0.3"
    echo "  2. Copy your fine-tuned model:"
    echo "       scp -r vm:~/LLM/models/mistral-7b-mental-health-fullft-a100 $PROJECT_DIR/models/"
    echo ""
    exit 1
else
    echo "Base model (local): $RESOLVED_MODEL"
fi

echo ""
echo "========================================================"
echo "  PHASE 3: Train LoRA Adapters — Mistral 7B (1x A100 80GB)"
echo "========================================================"
echo ""
echo "  Mistral 7B is memory-efficient. Using standard settings."
echo ""

TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

for SKILL in $SKILLS; do
    TOTAL=$((TOTAL + 1))
    ADAPTER_DIR="$ADAPTER_PREFIX/$SKILL"

    # Skip if already trained
    if [ -f "$ADAPTER_DIR/adapter_model.safetensors" ] || [ -f "$ADAPTER_DIR/adapter_model.bin" ]; then
        echo ""
        echo "[$TOTAL] $SKILL: already trained, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Skip if no skill dataset
    if [ ! -d "$DATASETS_DIR/skills/$SKILL/train" ]; then
        echo ""
        echo "[$TOTAL] $SKILL: no training data, skipping"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo ""
    echo "========================================================"
    echo "  Training: $SKILL ($TOTAL of $(echo $SKILLS | wc -w | tr -d ' '))"
    echo "  Time: $(date)"
    echo "========================================================"

    # Mistral 7B can handle larger batch sizes
    CMD="uv run python3 scripts/train_skill_lora.py \
        --skill $SKILL \
        --base_model $RESOLVED_MODEL \
        --output_dir $ADAPTER_DIR \
        --batch_size 4 \
        --eval_batch_size 4 \
        --max_length 768 \
        --gradient_accumulation_steps 2 \
        --lora_r 32 \
        --lora_alpha 64"

    if eval $CMD; then
        SUCCESS=$((SUCCESS + 1))
        echo "Completed: $SKILL"
    else
        FAILED=$((FAILED + 1))
        echo "FAILED: $SKILL (check logs above)"
    fi
done

# ==========================================================
# Summary
# ==========================================================
echo ""
echo "========================================================"
echo "  Pipeline Complete — Mistral 7B Skill Adapters"
echo "  End: $(date)"
echo "========================================================"
echo "  Total:   $TOTAL"
echo "  Trained: $SUCCESS"
echo "  Skipped: $SKIPPED (already done)"
echo "  Failed:  $FAILED"
echo ""
echo "  Adapter status:"
for SKILL in $SKILLS; do
    DIR="$ADAPTER_PREFIX/$SKILL"
    if [ -f "$DIR/adapter_model.safetensors" ] || [ -f "$DIR/adapter_model.bin" ]; then
        echo "    $SKILL: READY"
    elif [ -d "$DIR/emergency_checkpoint" ]; then
        echo "    $SKILL: FAILED (emergency checkpoint saved)"
    else
        echo "    $SKILL: NOT TRAINED"
    fi
done

# Also check general-support
DIR="$ADAPTER_PREFIX/general-support"
if [ -f "$DIR/adapter_model.safetensors" ] || [ -f "$DIR/adapter_model.bin" ]; then
    echo "    general-support: READY"
else
    echo "    general-support: NOT TRAINED"
fi

echo ""
echo "Adapters saved to: $ADAPTER_PREFIX/"
echo ""
