#!/bin/bash
# ============================================================
# GCP: Download & Process ALL Datasets (individual scripts)
# ============================================================
#
# Downloads and processes each dataset using the individual
# scripts in scripts/data/, then combines them all.
#
# Run this AFTER setup_remote.sh and BEFORE prep_training.sh.
#
# Usage:
#    bash gcp/download_data.sh
#
# ============================================================

set -euo pipefail

echo "========================================"
echo "Dataset Download & Processing (all scripts)"
echo "Start time: $(date)"
echo "========================================"

PROJECT_DIR="/workspace/LLM"
cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

DATASETS_DIR="datasets"
mkdir -p "$DATASETS_DIR"

# Track which datasets were successfully processed
PROCESSED=()

# ---------- 1. Counsel Chat ----------
echo ""
echo "[1/8] Counsel Chat"
COUNSEL_CSV="counsel-chat/data/20200325_counsel_chat.csv"
if [ -f "$COUNSEL_CSV" ]; then
    python3 scripts/data/prepare_counsel_dataset.py \
        --input_csv "$COUNSEL_CSV" \
        --output_dir "$DATASETS_DIR/counsel_chat_processed" \
    && PROCESSED+=("$DATASETS_DIR/counsel_chat_processed")
else
    echo "  Counsel Chat CSV not found at $COUNSEL_CSV, skipping."
    [ -d "$DATASETS_DIR/counsel_chat_processed" ] && PROCESSED+=("$DATASETS_DIR/counsel_chat_processed")
fi

# ---------- 2. Amod Mental Health ----------
echo ""
echo "[2/8] Amod Mental Health Counseling"
python3 scripts/data/prepare_amod_dataset.py \
    --output_dir "$DATASETS_DIR/amod_processed" \
    --cache_dir "$HF_DATASETS_CACHE" \
&& PROCESSED+=("$DATASETS_DIR/amod_processed")

# ---------- 3. ESConv ----------
echo ""
echo "[3/8] ESConv (Empathetic Conversations)"
python3 scripts/data/prepare_esconv_dataset.py \
    --output_dir "$DATASETS_DIR/esconv_processed" \
    --cache_dir "$HF_DATASETS_CACHE" \
&& PROCESSED+=("$DATASETS_DIR/esconv_processed")

# ---------- 4. CACTUS CBT ----------
echo ""
echo "[4/8] CACTUS (CBT)"
python3 scripts/data/prepare_cactus_dataset.py \
    --output_dir "$DATASETS_DIR/cactus_processed" \
    --cache_dir "$HF_DATASETS_CACHE" \
&& PROCESSED+=("$DATASETS_DIR/cactus_processed")

# ---------- 5. MentalChat16K ----------
echo ""
echo "[5/8] MentalChat16K"
python3 scripts/data/prepare_mentalchat16k_dataset.py \
    --output_dir "$DATASETS_DIR/mentalchat16k_processed" \
    --cache_dir "$HF_DATASETS_CACHE" \
&& PROCESSED+=("$DATASETS_DIR/mentalchat16k_processed")

# ---------- 6. PsyDial (Chinese) ----------
echo ""
echo "[6/8] PsyDial (Chinese)"
python3 scripts/data/prepare_psydial_dataset.py \
    --output_dir "$DATASETS_DIR/psydial_processed" \
    --cache_dir "$HF_DATASETS_CACHE" \
&& PROCESSED+=("$DATASETS_DIR/psydial_processed")

# ---------- 7. Kaggle Mental Health (Nguyen) ----------
echo ""
echo "[7/8] Kaggle Mental Health (Nguyen)"
KAGGLE_INPUT_DIR="$DATASETS_DIR/kaggle_mental_health_nguyen"
if [ -d "$KAGGLE_INPUT_DIR" ]; then
    python3 scripts/data/prepare_kaggle_dataset.py \
        --input_dir "$KAGGLE_INPUT_DIR" \
        --output_dir "$DATASETS_DIR/kaggle_mental_health_nguyen_processed_combined" \
    && PROCESSED+=("$DATASETS_DIR/kaggle_mental_health_nguyen_processed_combined")
else
    echo "  Kaggle Nguyen directory not found at $KAGGLE_INPUT_DIR, skipping."
    [ -d "$DATASETS_DIR/kaggle_mental_health_nguyen_processed_combined" ] \
        && PROCESSED+=("$DATASETS_DIR/kaggle_mental_health_nguyen_processed_combined")
fi

# ---------- 8. Suicide Watch / Crisis Detection ----------
echo ""
echo "[8/8] Suicide Watch (Crisis Detection)"
SUICIDE_CSV="$DATASETS_DIR/kaggle_suicide_watch/Suicide_Detection.csv"
if [ -f "$SUICIDE_CSV" ]; then
    python3 scripts/download_and_process_datasets.py \
        --suicide_csv "$SUICIDE_CSV" \
        --skip_downloads
    [ -d "$DATASETS_DIR/crisis_detection_processed" ] \
        && PROCESSED+=("$DATASETS_DIR/crisis_detection_processed")
else
    echo "  Suicide Watch CSV not found at $SUICIDE_CSV, skipping."
    [ -d "$DATASETS_DIR/crisis_detection_processed" ] \
        && PROCESSED+=("$DATASETS_DIR/crisis_detection_processed")
fi

# ---------- Combine all ----------
echo ""
echo "========================================"
echo "Combining ${#PROCESSED[@]} datasets..."
echo "========================================"
for ds in "${PROCESSED[@]}"; do
    echo "  + $(basename "$ds")"
done

python3 scripts/data/combine_all_datasets.py \
    --datasets "/workspace/LLM/datasets/*" \
    --output_dir "/workspace/LLM/datasets/all_mental_health_combined"

echo ""
echo "========================================"
echo "Dataset download & processing complete!"
echo "End time: $(date)"
echo "========================================"
echo ""
echo "Datasets processed: ${#PROCESSED[@]}"
echo "Combined output:    $DATASETS_DIR/all_mental_health_combined"
echo ""
echo "Next: bash gcp/prep_training.sh"
