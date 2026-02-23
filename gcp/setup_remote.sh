#!/bin/bash
# ============================================================
# GCP: Environment Setup (run once after SSH)
# ============================================================
#
# Sets up Python venv, installs PyTorch + dependencies,
# verifies GPU access.
#
# Usage:
#    bash gcp/setup_remote.sh
#
# ============================================================

set -euo pipefail

echo "========================================"
echo "GCP Environment Setup"
echo "========================================"

PROJECT_DIR="/workspace/LLM"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project not found at $PROJECT_DIR"
    echo "Clone it first:  git clone <REPO_URL> ~/LLM"
    exit 1
fi

cd "$PROJECT_DIR"

# ---------- Step 1: Python venv ----------
echo ""
echo "[1/4] Creating virtual environment..."

if [ -d ".venv" ]; then
    echo "Venv already exists, skipping creation."
else
    python3 -m venv .venv
fi
source .venv/bin/activate

# ---------- Step 2: Install dependencies ----------
echo ""
echo "[2/4] Installing dependencies..."

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[cuda]"
pip install rouge-score

echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ---------- Step 3: Create directories ----------
echo ""
echo "[3/4] Creating directories..."
mkdir -p logs models datasets .cache/huggingface

# ---------- Step 4: Verify GPUs ----------
echo ""
echo "[4/4] Checking GPUs..."

nvidia-smi

python3 -c "
import torch
gpu_count = torch.cuda.device_count()
print(f'GPU count: {gpu_count}')
for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_mem / 1024**3:.0f} GB')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next: bash gcp/download_data.sh"
