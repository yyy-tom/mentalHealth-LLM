#!/bin/bash
#SBATCH --job-name=llama8b-pipeline
#SBATCH --output=logs/llama_pipeline_%j.out
#SBATCH --error=logs/llama_pipeline_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00

# ============================================================
# SLURM: Llama-3.1-8B Complete Training Pipeline
# ============================================================
#
# End-to-end workflow in a single SLURM job:
#   Step 0: Environment setup (venv + dependencies)
#   Step 1: HuggingFace authentication check
#   Step 2: Download & process datasets
#   Step 3: Download Llama-3.1-8B model weights
#   Step 4: Pre-tokenize dataset
#   Step 5: Train (QLoRA on 4090 / full fine-tune on A100+)
#
# Usage:
#   sbatch slurm/llama_8b_pipeline.sh
#
# To skip steps that are already done:
#   SKIP_SETUP=1 SKIP_DATA=1 sbatch slurm/llama_8b_pipeline.sh
#
# For full fine-tune on A100 (submit from a node with A100):
#   TRAIN_MODE=fullft sbatch --gres=gpu:a100:1 slurm/llama_8b_pipeline.sh
#
# ============================================================

set -euo pipefail

# ---------- Configuration ----------
PROJECT_DIR="/research/d7/fyp25/yyyu2/LLM"
TRAIN_MODE="${TRAIN_MODE:-qlora}"   # "qlora" (4090/2080) or "fullft" (A100/H100)
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH="datasets/all_mental_health_combined"

# Select config based on training mode
if [ "$TRAIN_MODE" = "fullft" ]; then
    CONFIG="configs/llama_8b_fullft_a100_1gpu_fast.json"
    TOKENIZED_SUFFIX="_tokenized_llama_fullft"
else
    CONFIG="configs/llama_8b_4090.json"
    TOKENIZED_SUFFIX="_tokenized_llama_qlora"
fi

TOKENIZED_PATH="${DATASET_PATH}${TOKENIZED_SUFFIX}"

# Skip flags
SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_TOKENIZE="${SKIP_TOKENIZE:-0}"

echo "============================================================"
echo "  Llama-3.1-8B Training Pipeline (SLURM)"
echo "============================================================"
echo "  Job ID      : ${SLURM_JOB_ID:-local}"
echo "  Node        : ${SLURM_NODELIST:-$(hostname)}"
echo "  Train mode  : $TRAIN_MODE"
echo "  Config      : $CONFIG"
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

    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        if command -v uv &> /dev/null; then
            uv venv
        else
            python3 -m venv .venv
        fi
    fi
    source .venv/bin/activate

    # Install/upgrade dependencies
    echo "Installing dependencies..."
    if command -v uv &> /dev/null; then
        uv pip install --python "$VIRTUAL_ENV/bin/python" --force-reinstall \
            torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        uv pip install --python "$VIRTUAL_ENV/bin/python" -e ".[cuda]"
        uv pip install --python "$VIRTUAL_ENV/bin/python" rouge-score
    else
        pip install --upgrade pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install -e ".[cuda]"
        pip install rouge-score
    fi

    mkdir -p logs models datasets

    # Verify installation
    python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
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

# ---------- Cache paths ----------
export HF_HOME="/research/d7/fyp25/yyyu2/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ============================================================
# Step 1: HuggingFace Authentication
# ============================================================
echo ""
echo "========================================"
echo "[1/5] Checking HuggingFace authentication..."
echo "========================================"

if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is set in environment."
elif [ -f "$HF_HOME/token" ]; then
    echo "HF token found at $HF_HOME/token."
else
    echo "ERROR: Llama is a gated model. No HF token found."
    echo ""
    echo "Before submitting this job, run interactively:"
    echo "  source .venv/bin/activate"
    echo "  huggingface-cli login"
    echo ""
    echo "Or set HF_TOKEN in your environment:"
    echo "  export HF_TOKEN=hf_xxxxx"
    echo ""
    exit 1
fi

# Verify model access
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.model_info('$MODEL_NAME')
    print('Llama model access: OK')
except Exception as e:
    print(f'ERROR: Cannot access $MODEL_NAME — {e}')
    print('Accept the license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct')
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
        echo "Running full dataset pipeline..."

        # Use the existing all-in-one script with Suicide Watch CSV
        SUICIDE_CSV="datasets/kaggle_suicide_watch/Suicide_Detection.csv"
        if [ -f "$SUICIDE_CSV" ]; then
            python3 scripts/download_and_process_datasets.py \
                --suicide_csv "$SUICIDE_CSV" \
                --tokenize \
                --config "$CONFIG"
        else
            python3 scripts/download_and_process_datasets.py \
                --skip_kaggle \
                --tokenize \
                --config "$CONFIG"
        fi
    fi
    echo ""
else
    echo "[2/5] Skipping data download (SKIP_DATA=1)"
fi

# ============================================================
# Step 3: Download Llama-3.1-8B Model Weights
# ============================================================
if [ "$SKIP_MODEL" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[3/5] Downloading Llama-3.1-8B model weights..."
    echo "========================================"

    python3 -c "
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_name = '$MODEL_NAME'
print(f'Downloading {model_name}...')

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f'Tokenizer vocab size: {len(tokenizer):,}')

print('Downloading model weights (this may take 10-20 min)...')
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
# Step 4: Pre-tokenize Dataset for Llama
# ============================================================
if [ "$SKIP_TOKENIZE" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[4/5] Pre-tokenizing dataset for Llama..."
    echo "========================================"

    if [ -d "$TOKENIZED_PATH" ]; then
        echo "Pre-tokenized dataset already exists at $TOKENIZED_PATH, skipping."
    elif [ -d "$DATASET_PATH" ]; then
        echo "y" | python3 scripts/pretokenize_dataset.py \
            --config "$CONFIG" \
            --output_suffix "$TOKENIZED_SUFFIX"
    else
        echo "ERROR: Dataset not found at $DATASET_PATH"
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
echo "[5/5] Starting Llama-3.1-8B training ($TRAIN_MODE)..."
echo "========================================"

nvidia-smi
echo ""

if [ "$TRAIN_MODE" = "fullft" ]; then
    echo "Running full fine-tune with torchrun..."
    torchrun \
        --nproc_per_node=1 \
        --master_port=29500 \
        scripts/train_qwen_fullft.py \
        --config "$CONFIG"
else
    echo "Running QLoRA training..."
    python3 scripts/train_qwen_counsel.py \
        --config "$CONFIG"
fi

echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "  End time: $(date)"
echo "============================================================"
echo ""
echo "Model saved to: $(python3 -c "import json; print(json.load(open('$CONFIG'))['output_dir'])")"
