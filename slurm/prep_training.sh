#!/bin/bash
#SBATCH --job-name=prep-qwen7b
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:rtx2080:8
#SBATCH -c 30
#SBATCH --time=02:00:00

# ============================================
# Pre-training Preparation Job
# - Downloads Qwen2.5-7B model weights
# - Pre-tokenizes the dataset
# Run this BEFORE the main training job
# ============================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Set project directory
PROJECT_DIR="/research/d7/fyp25/yyyu2/LLM"
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
source .venv/bin/activate

# Cache on research partition — NOT home dir (quota limit)
export HF_HOME="/research/d7/fyp25/yyyu2/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Create logs directory
mkdir -p logs

echo ""
echo "========================================"
echo "Step 0: Fixing torch/torchvision versions"
echo "========================================"

# Reinstall torch packages with matching versions to fix compatibility
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "========================================"
echo "Step 1: Downloading Qwen2.5-7B-Instruct"
echo "========================================"

# Use huggingface_hub to download without loading (avoids memory issues)
python3 -c "
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_name = 'Qwen/Qwen2.5-7B-Instruct'
print(f'Downloading {model_name}...')

# Download tokenizer first (small)
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f'Tokenizer vocab size: {len(tokenizer):,}')

# Download model files without loading into memory
print('Downloading model weights (this may take a while)...')
snapshot_download(
    repo_id=model_name,
    local_dir=None,  # Use default cache
    ignore_patterns=['*.md', '*.txt'],  # Skip docs
)
print('Model weights cached successfully!')
"

echo ""
echo "========================================"
echo "Step 2: Pre-tokenizing Dataset"
echo "========================================"

# Pre-tokenize the dataset (saves ~30-60 min during training)
python3 scripts/pretokenize_dataset.py --config configs/qwen_7b_8x2080ti.json <<< "y"

echo ""
echo "========================================"
echo "Preparation Complete!"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "You can now submit the main training job:"
echo "  sbatch slurm/train_7b_8x2080ti.sh"
