#!/bin/bash
# ============================================================
# GCP: Gemma-2-9B-it Complete Training Pipeline
# ============================================================
#
# End-to-end workflow:
#   Step 0: Environment setup (venv + dependencies)
#   Step 1: HuggingFace login (Gemma is gated)
#   Step 2: Download & process datasets
#   Step 3: Download Gemma-2-9B model weights
#   Step 4: Pre-tokenize dataset
#   Step 5: Train
#
# Usage (run inside tmux on the VM):
#   bash gcp/gemma2_9b_pipeline.sh
#
# To skip steps that are already done:
#   SKIP_SETUP=1 SKIP_DATA=1 bash gcp/gemma2_9b_pipeline.sh
#
# To use multi-GPU:
#   NUM_GPUS=8 CONFIG=configs/gemma2_9b_fullft_h100_8gpu.json bash gcp/gemma2_9b_pipeline.sh
#
# ============================================================

set -euo pipefail

# ---------- Configuration ----------
PROJECT_DIR="${PROJECT_DIR:-/workspace/LLM}"
CONFIG="${CONFIG:-configs/gemma2_9b_fullft_a100_1gpu_fast.json}"
NUM_GPUS="${NUM_GPUS:-1}"
MODEL_NAME="google/gemma-2-9b-it"
DATASET_PATH="datasets/all_mental_health_combined"
TOKENIZED_PATH="${DATASET_PATH}_tokenized_gemma2"

# Skip flags (set to 1 to skip a step)
SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_TOKENIZE="${SKIP_TOKENIZE:-0}"

echo "============================================================"
echo "  Gemma-2-9B-it Complete Training Pipeline"
echo "============================================================"
echo "  Project dir : $PROJECT_DIR"
echo "  Config      : $CONFIG"
echo "  GPUs        : $NUM_GPUS"
echo "  Model       : $MODEL_NAME"
echo "  Start time  : $(date)"
echo "============================================================"
echo ""

cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

# ============================================================
# Step 0: Environment Setup
# ============================================================
if [ "$SKIP_SETUP" = "0" ]; then
    echo "========================================"
    echo "[0/5] Setting up environment..."
    echo "========================================"

    # Create venv if needed
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate

    # Install dependencies
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install -e ".[cuda]"
    pip install rouge-score

    # Create directories
    mkdir -p logs models datasets .cache/huggingface

    # Verify
    python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}, {props.total_memory / 1024**3:.0f} GB')
import transformers, peft, datasets as ds
print(f'transformers {transformers.__version__}, peft {peft.__version__}, datasets {ds.__version__}')
"
    echo ""
else
    echo "[0/5] Skipping setup (SKIP_SETUP=1)"
    source .venv/bin/activate
fi

# ---------- Set cache paths ----------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# CUDA / NCCL settings
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- Kaggle credentials ----------
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Setting up Kaggle credentials..."
    mkdir -p "$HOME/.kaggle"
    echo '{"username":"yuyanyuk","key":"b2732b941a0d12912406a92478d7f3c2"}' > "$HOME/.kaggle/kaggle.json"
    chmod 600 "$HOME/.kaggle/kaggle.json"
fi

# ============================================================
# Step 1: HuggingFace Login (Gemma is a gated model)
# ============================================================
echo ""
echo "========================================"
echo "[1/5] Checking HuggingFace authentication..."
echo "========================================"

if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is set in environment."
elif [ -f "$HF_HOME/token" ]; then
    echo "HF token found in cache."
else
    echo ""
    echo "Gemma 2 is a gated model. You need to:"
    echo "  1. Accept the license at: https://huggingface.co/google/gemma-2-9b-it"
    echo "  2. Log in with: huggingface-cli login"
    echo ""
    echo "Running huggingface-cli login..."
    huggingface-cli login
fi

# Verify access
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.model_info('$MODEL_NAME')
    print('Gemma 2 model access: OK')
except Exception as e:
    print(f'ERROR: Cannot access $MODEL_NAME')
    print(f'  {e}')
    print('  Accept the license at: https://huggingface.co/google/gemma-2-9b-it')
    exit(1)
"

# ============================================================
# Step 2: Download & Process Datasets
# ============================================================
if [ "$SKIP_DATA" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[2/5] Downloading & processing datasets..."
    echo "========================================"

    if [ -d "$DATASET_PATH" ]; then
        echo "Combined dataset already exists at $DATASET_PATH"
        python3 -c "
from datasets import load_from_disk
ds = load_from_disk('$DATASET_PATH')
print(f'  Train: {len(ds[\"train\"]):,}, Validation: {len(ds[\"validation\"]):,}')
"
    else
        echo "Running dataset pipeline..."
        DATASETS_DIR="datasets"
        mkdir -p "$DATASETS_DIR"
        PROCESSED=()

        # --- Counsel Chat ---
        echo "[2a] Counsel Chat"
        COUNSEL_CSV="counsel-chat/data/20200325_counsel_chat.csv"
        if [ -f "$COUNSEL_CSV" ]; then
            python3 scripts/data/prepare_counsel_dataset.py \
                --input_csv "$COUNSEL_CSV" \
                --output_dir "$DATASETS_DIR/counsel_chat_processed" \
            && PROCESSED+=("$DATASETS_DIR/counsel_chat_processed")
        else
            [ -d "$DATASETS_DIR/counsel_chat_processed" ] && PROCESSED+=("$DATASETS_DIR/counsel_chat_processed")
        fi

        # --- Amod Mental Health ---
        echo "[2b] Amod Mental Health"
        python3 scripts/data/prepare_amod_dataset.py \
            --output_dir "$DATASETS_DIR/amod_processed" \
            --cache_dir "$HF_DATASETS_CACHE" \
        && PROCESSED+=("$DATASETS_DIR/amod_processed")

        # --- ESConv ---
        echo "[2c] ESConv"
        python3 scripts/data/prepare_esconv_dataset.py \
            --output_dir "$DATASETS_DIR/esconv_processed" \
            --cache_dir "$HF_DATASETS_CACHE" \
        && PROCESSED+=("$DATASETS_DIR/esconv_processed")

        # --- CACTUS CBT ---
        echo "[2d] CACTUS (CBT)"
        python3 scripts/data/prepare_cactus_dataset.py \
            --output_dir "$DATASETS_DIR/cactus_processed" \
            --cache_dir "$HF_DATASETS_CACHE" \
        && PROCESSED+=("$DATASETS_DIR/cactus_processed")

        # --- MentalChat16K ---
        echo "[2e] MentalChat16K"
        python3 scripts/data/prepare_mentalchat16k_dataset.py \
            --output_dir "$DATASETS_DIR/mentalchat16k_processed" \
            --cache_dir "$HF_DATASETS_CACHE" \
        && PROCESSED+=("$DATASETS_DIR/mentalchat16k_processed")

        # --- PsyDial (Chinese) ---
        echo "[2f] PsyDial (Chinese)"
        python3 scripts/data/prepare_psydial_dataset.py \
            --output_dir "$DATASETS_DIR/psydial_processed" \
            --cache_dir "$HF_DATASETS_CACHE" \
        && PROCESSED+=("$DATASETS_DIR/psydial_processed")

        # --- Kaggle Nguyen ---
        echo "[2g] Kaggle Mental Health (Nguyen)"
        KAGGLE_INPUT_DIR="$DATASETS_DIR/kaggle_mental_health_nguyen"
        if [ -d "$KAGGLE_INPUT_DIR" ]; then
            python3 scripts/data/prepare_kaggle_dataset.py \
                --input_dir "$KAGGLE_INPUT_DIR" \
                --output_dir "$DATASETS_DIR/kaggle_mental_health_nguyen_processed_combined" \
            && PROCESSED+=("$DATASETS_DIR/kaggle_mental_health_nguyen_processed_combined")
        else
            [ -d "$DATASETS_DIR/kaggle_mental_health_nguyen_processed_combined" ] \
                && PROCESSED+=("$DATASETS_DIR/kaggle_mental_health_nguyen_processed_combined")
        fi

        # --- Suicide Watch ---
        echo "[2h] Suicide Watch (Crisis Detection)"
        SUICIDE_CSV="$DATASETS_DIR/kaggle_suicide_watch/Suicide_Detection.csv"
        if [ -f "$SUICIDE_CSV" ]; then
            python3 scripts/download_and_process_datasets.py \
                --suicide_csv "$SUICIDE_CSV" \
                --skip_downloads
            [ -d "$DATASETS_DIR/crisis_detection_processed" ] \
                && PROCESSED+=("$DATASETS_DIR/crisis_detection_processed")
        else
            [ -d "$DATASETS_DIR/crisis_detection_processed" ] \
                && PROCESSED+=("$DATASETS_DIR/crisis_detection_processed")
        fi

        # --- Combine ---
        echo ""
        echo "Combining ${#PROCESSED[@]} datasets..."
        python3 scripts/data/combine_all_datasets.py \
            --datasets "$PROJECT_DIR/datasets/*" \
            --output_dir "$PROJECT_DIR/$DATASET_PATH"
    fi
    echo ""
else
    echo "[2/5] Skipping data download (SKIP_DATA=1)"
fi

# ============================================================
# Step 3: Download Gemma-2-9B Model Weights
# ============================================================
if [ "$SKIP_MODEL" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[3/5] Downloading Gemma-2-9B model weights..."
    echo "========================================"

    python3 -c "
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_name = '$MODEL_NAME'
print(f'Downloading {model_name}...')

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f'Tokenizer vocab size: {len(tokenizer):,}')

print('Downloading model weights (this may take 15-25 min for 9B)...')
snapshot_download(
    repo_id=model_name,
    local_dir=None,
    ignore_patterns=['*.md', '*.txt'],
)
print('Model weights cached successfully!')
"
    echo ""
else
    echo "[3/5] Skipping model download (SKIP_MODEL=1)"
fi

# ============================================================
# Step 4: Pre-tokenize Dataset for Gemma 2
# ============================================================
if [ "$SKIP_TOKENIZE" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[4/5] Pre-tokenizing dataset for Gemma 2..."
    echo "========================================"

    if [ -d "$TOKENIZED_PATH" ]; then
        echo "Pre-tokenized dataset already exists at $TOKENIZED_PATH, skipping."
    elif [ -d "$DATASET_PATH" ]; then
        # Hide GPU during tokenization to prevent CUDA fork errors
        echo "y" | CUDA_VISIBLE_DEVICES="" python3 scripts/pretokenize_dataset.py \
            --config "$CONFIG" \
            --output_suffix "_tokenized_gemma2"
    else
        echo "ERROR: Dataset not found at $DATASET_PATH"
        echo "Re-run without SKIP_DATA=1"
        exit 1
    fi
    echo ""
else
    echo "[4/5] Skipping pre-tokenization (SKIP_TOKENIZE=1)"
fi

# ============================================================
# Step 5: Train
# ============================================================
echo ""
echo "========================================"
echo "[5/5] Starting Gemma-2-9B training..."
echo "========================================"
echo "  Config : $CONFIG"
echo "  GPUs   : $NUM_GPUS"
echo ""
echo "  NOTE: Gemma 2 9B is memory-tight on A100 80GB."
echo "  If you see OOM, reduce max_length to 384 in the config."
echo ""

nvidia-smi
echo ""

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        scripts/train_qwen_fullft.py \
        --config "$CONFIG"
else
    python3 scripts/train_qwen_fullft.py \
        --config "$CONFIG"
fi

echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "  End time: $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Check training logs in the output directory"
echo "  2. Download model: gcloud compute scp --recurse VM:~/LLM/models/gemma2-9b-mental-health-fullft-a100 ./models/ --zone=ZONE"
echo "  3. Stop VM: gcloud compute instances stop VM --zone=ZONE"
