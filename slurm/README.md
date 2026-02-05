# Remote Training Guide

Instructions for training Qwen2.5-7B on your department's GPC cluster.

## Prerequisites

- SSH access to the cluster
- SLURM job scheduler
- Either 8x RTX 2080 Ti or 1x RTX 4090 available

## Step 1: Sync Project to Remote

From your **local machine**, run:

```bash
# Replace with your cluster details
REMOTE_USER="your_username"
REMOTE_HOST="cluster.department.edu"
REMOTE_DIR="~/LLM"

# Sync entire project (excluding large cache files)
rsync -avz --progress \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'models/*' \
    --exclude 'logs/*' \
    --exclude '.cache' \
    ./ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/
```

Or use the provided sync script:
```bash
./slurm/sync_to_remote.sh username cluster.department.edu
```

## Step 2: Set Up Environment on Remote

SSH into the cluster and run:

```bash
ssh username@cluster.department.edu
cd ~/LLM
bash slurm/setup_remote.sh
```

This will:
- Create a Python virtual environment
- Install PyTorch with CUDA support
- Install all dependencies (transformers, peft, bitsandbytes, etc.)
- Verify GPU availability
- Check dataset exists

## Step 3: Submit Training Job

Choose based on your available hardware:

### Option A: 8x RTX 2080 Ti (Recommended - faster)
```bash
sbatch slurm/train_7b_8x2080ti.sh
```

### Option B: 1x RTX 4090
```bash
sbatch slurm/train_7b_4090.sh
```

## Step 4: Monitor Training

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f logs/slurm_<job_id>.out

# Check GPU usage (on the compute node)
srun --jobid=<job_id> nvidia-smi
```

## Step 5: Retrieve Results

After training completes, sync the model back:

```bash
# From local machine
rsync -avz --progress \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/models/ \
    ./models/
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` to 1
- Reduce `max_length` (768 → 512)
- Reduce `lora_r` (64 → 32)

### NCCL Errors (Multi-GPU)
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # If InfiniBand issues
```

### Module Not Found
Make sure you activate the venv:
```bash
source ~/LLM/.venv/bin/activate
```

## File Structure

```
slurm/
├── train_7b_8x2080ti.sh  # SLURM script for 8x RTX 2080 Ti
├── train_7b_4090.sh      # SLURM script for 1x RTX 4090
├── setup_remote.sh       # Environment setup script
└── README.md             # This file
```

## Expected Training Time

| Hardware | Epochs | Estimated Time |
|----------|--------|----------------|
| 8x RTX 2080 Ti | 2 | ~8-12 hours |
| 1x RTX 4090 | 2 | ~24-36 hours |
