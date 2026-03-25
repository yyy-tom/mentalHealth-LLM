#!/bin/bash
# ============================================================
# GCP: Mistral-7B-Instruct-v0.3 Complete Training Pipeline
# ============================================================
#
# End-to-end workflow:
#   Step 0: Environment setup (venv + dependencies)
#   Step 1: Download & process datasets (if needed)
#   Step 2: Download Mistral-7B model weights (NOT gated — no login needed)
#   Step 3: Pre-tokenize dataset
#   Step 4: Train
#
# Usage (run inside tmux on the VM):
#   bash gcp/mistral_7b_pipeline.sh
#
# To skip steps that are already done:
#   SKIP_SETUP=1 SKIP_DATA=1 bash gcp/mistral_7b_pipeline.sh
#
# ============================================================

set -euo pipefail

# ---------- Configuration ----------
PROJECT_DIR="${PROJECT_DIR:-/workspace/LLM}"
CONFIG="${CONFIG:-configs/mistral_7b_fullft_a100_1gpu_fast.json}"
NUM_GPUS="${NUM_GPUS:-1}"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH="datasets/all_mental_health_combined"
TOKENIZED_PATH="${DATASET_PATH}_tokenized_mistral"

# Skip flags (set to 1 to skip a step)
SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_TOKENIZE="${SKIP_TOKENIZE:-0}"

echo "============================================================"
echo "  Mistral-7B-Instruct-v0.3 Complete Training Pipeline"
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
    echo "[0/4] Setting up environment..."
    echo "========================================"

    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate

    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -e ".[cuda]"
    pip install rouge-score

    mkdir -p logs models datasets

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
    echo "[0/4] Skipping setup (SKIP_SETUP=1)"
    source .venv/bin/activate
fi

# ---------- Set cache paths ----------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- Kaggle credentials ----------
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Setting up Kaggle credentials..."
    mkdir -p "$HOME/.kaggle"
    echo '{"username":"yuyanyuk","key":"b2732b941a0d12912406a92478d7f3c2"}' > "$HOME/.kaggle/kaggle.json"
    chmod 600 "$HOME/.kaggle/kaggle.json"
fi

# ============================================================
# Step 1: Download & Process Datasets
# ============================================================
if [ "$SKIP_DATA" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[1/4] Checking datasets..."
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

        # Download any missing datasets
        python3 scripts/data/download_missing_datasets.py

        # Combine all
        python3 scripts/data/combine_all_datasets.py \
            --output_dir "$DATASET_PATH"
    fi
    echo ""
else
    echo "[1/4] Skipping data (SKIP_DATA=1)"
fi

# ============================================================
# Step 2: Download Mistral-7B Model Weights (NOT gated)
# ============================================================
if [ "$SKIP_MODEL" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[2/4] Downloading Mistral-7B model weights..."
    echo "       (NOT gated — no login needed)"
    echo "========================================"

    python3 -c "
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_name = '$MODEL_NAME'
print(f'Downloading {model_name}...')

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f'Tokenizer vocab size: {len(tokenizer):,}')

print('Downloading model weights...')
snapshot_download(
    repo_id=model_name,
    local_dir=None,
    ignore_patterns=['*.md', '*.txt'],
)
print('Model weights cached successfully!')
"
    echo ""
else
    echo "[2/4] Skipping model download (SKIP_MODEL=1)"
fi

# ============================================================
# Step 3: Pre-tokenize Dataset for Mistral
# ============================================================
if [ "$SKIP_TOKENIZE" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[3/4] Pre-tokenizing dataset for Mistral..."
    echo "========================================"

    if [ -d "$TOKENIZED_PATH" ]; then
        echo "Pre-tokenized dataset already exists at $TOKENIZED_PATH, skipping."
    elif [ -d "$DATASET_PATH" ]; then
        # Hide GPU during tokenization to prevent CUDA fork errors
        echo "y" | CUDA_VISIBLE_DEVICES="" python3 scripts/pretokenize_dataset.py \
            --config "$CONFIG" \
            --output_suffix "_tokenized_mistral"
    else
        echo "ERROR: Dataset not found at $DATASET_PATH"
        exit 1
    fi
    echo ""
else
    echo "[3/4] Skipping pre-tokenization (SKIP_TOKENIZE=1)"
fi

# ============================================================
# Step 4: Train
# ============================================================
echo ""
echo "========================================"
echo "[4/4] Starting Mistral-7B training..."
echo "========================================"
echo "  Config : $CONFIG"
echo "  GPUs   : $NUM_GPUS"
echo ""

# Check disk space
DISK_FREE=$(df --output=avail -BG /workspace 2>/dev/null | tail -1 | tr -d ' G' || echo "unknown")
echo "  Disk free: ${DISK_FREE}GB"
if [ "$DISK_FREE" != "unknown" ] && [ "$DISK_FREE" -lt 50 ]; then
    echo "  WARNING: Low disk space! Consider cleaning up old checkpoints:"
    echo "    rm -rf models/*/checkpoint-*"
fi
echo ""

nvidia-smi
echo ""

# Use plain python for single GPU (avoids NCCL/torchrun overhead)
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
echo "Model saved to: models/mistral-7b-mental-health-fullft-a100/"
