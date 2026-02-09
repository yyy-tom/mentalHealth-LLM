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
echo "Step 1: Downloading Qwen2.5-7B-Instruct"
echo "========================================"

python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Qwen/Qwen2.5-7B-Instruct'
print(f'Downloading {model_name}...')

# Download tokenizer
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f'Tokenizer vocab size: {len(tokenizer):,}')

# Download model (this caches the weights)
print('Downloading model weights (this may take a while)...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f'Model downloaded successfully!')
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

# Clean up to free memory
del model
del tokenizer
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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
