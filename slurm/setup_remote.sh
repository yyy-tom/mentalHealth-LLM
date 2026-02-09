#!/bin/bash
# ============================================
# Remote Environment Setup Script
# Run this once on the remote cluster to set up the environment
# ============================================

set -e  # Exit on error

echo "========================================"
echo "Setting up LLM training environment"
echo "========================================"

# Set project directory (adjust if needed)
PROJECT_DIR="/research/d7/fyp25/yyyu2/LLM"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found at $PROJECT_DIR"
    echo "Please sync the project first using rsync or git clone"
    exit 1
fi

cd "$PROJECT_DIR"

# ============================================
# 1. Create virtual environment
# ============================================
echo ""
echo "Step 1: Creating virtual environment..."

# Try uv first (faster), fall back to python venv
if command -v uv &> /dev/null; then
    echo "Using uv to create venv..."
    uv venv
else
    echo "uv not found, using python venv..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# ============================================
# 2. Install dependencies
# ============================================
echo ""
echo "Step 2: Installing dependencies..."

if command -v uv &> /dev/null; then
    uv pip install -e .
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    uv pip install bitsandbytes accelerate trl
else
    pip install --upgrade pip
    pip install -e .
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install bitsandbytes accelerate trl
fi

# ============================================
# 3. Verify installation
# ============================================
echo ""
echo "Step 3: Verifying installation..."

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

import transformers
print(f'Transformers version: {transformers.__version__}')

import peft
print(f'PEFT version: {peft.__version__}')

try:
    import bitsandbytes as bnb
    print(f'BitsAndBytes version: {bnb.__version__}')
except ImportError:
    print('WARNING: BitsAndBytes not installed (required for QLoRA)')
"

# ============================================
# 4. Create necessary directories
# ============================================
echo ""
echo "Step 4: Creating directories..."
mkdir -p logs models .cache/huggingface

# ============================================
# 5. Verify dataset exists
# ============================================
echo ""
echo "Step 5: Checking datasets..."

if [ -d "datasets/all_mental_health_combined" ]; then
    echo "✓ Combined dataset found"
    python3 -c "
from datasets import load_from_disk
ds = load_from_disk('datasets/all_mental_health_combined')
print(f'  Train samples: {len(ds[\"train\"]):,}')
print(f'  Validation samples: {len(ds[\"validation\"]):,}')
"
else
    echo "✗ Combined dataset not found!"
    echo "  Run: python3 scripts/download_and_process_datasets.py --tokenize --config configs/qwen_7b_8x2080ti.json"
    echo "  Or:  sbatch slurm/download_data.sh"
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To prepare data (download + process + combine + tokenize):"
echo "  python3 scripts/download_and_process_datasets.py --tokenize --config configs/qwen_7b_8x2080ti.json"
echo "  Or: sbatch slurm/download_data.sh"
echo ""
echo "To submit a training job:"
echo "  sbatch slurm/train_7b_8x2080ti.sh   # For 8x RTX 2080 Ti"
echo "  sbatch slurm/train_7b_4090.sh       # For 1x RTX 4090"
echo ""
