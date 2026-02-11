# GCP Training Guide

Qwen2.5-7B full fine-tune on 8x H100 80GB (DDP, no LoRA, no quantization).

## Prerequisites

- GCP account with GPU quota for H100
- `gcloud` CLI installed locally

## Step 0: Create the VM (from local machine)

```bash
gcloud compute instances create qwen-7b-train \
    --zone=us-central1-a \
    --machine-type=a3-highgpu-8g \
    --accelerator=type=nvidia-h100-80gb,count=8 \
    --boot-disk-size=500GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

## Step 1: SSH & Clone

```bash
gcloud compute ssh qwen-7b-train --zone=us-central1-a

# On the VM:
git clone https://github.com/YOUR_USERNAME/LLM.git ~/LLM
cd ~/LLM
```

## Step 2: tmux + Setup

```bash
tmux new -s train

# Inside tmux:
bash gcp/setup_remote.sh        # ~5 min  — venv + dependencies + GPU check
```

## Step 3: Download Data

```bash
bash gcp/download_data.sh       # ~10 min — download & process datasets
```

## Step 4: Prep (model weights + tokenize)

```bash
bash gcp/prep_training.sh       # ~15 min — download Qwen2.5-7B + pre-tokenize
```

## Step 5: Train

```bash
bash gcp/train_7b_fullft.sh     # ~5-6 hrs — 8x H100 DDP full fine-tune
```

Detach tmux with `Ctrl-b d` and reconnect later with `tmux attach -t train`.

## Step 6: Download Model (from local machine)

```bash
gcloud compute scp --recurse \
    qwen-7b-train:~/LLM/models/qwen2.5-7b-mental-health-fullft-8gpu \
    ./models/ --zone=us-central1-a
```

## Step 7: Stop / Delete VM

```bash
# Stop (keeps disk, no compute charges, ~$8/month for 500GB disk)
gcloud compute instances stop qwen-7b-train --zone=us-central1-a

# Delete (removes everything, all charges stop)
gcloud compute instances delete qwen-7b-train --zone=us-central1-a
```

## Cost Estimate

| Item | Rate | Duration | Cost |
|------|------|----------|------|
| a3-highgpu-8g (8x H100) on-demand | ~$31/hr | ~6 hrs | ~$186 |
| a3-highgpu-8g spot (preemptible) | ~$10/hr | ~6 hrs | ~$60 |
| Boot disk (500GB, while stopped) | ~$0.04/GB/month | — | ~$20/month |

## Expected Training Time

| Hardware | Epochs | Dataset | Estimated Time |
|----------|--------|---------|----------------|
| 8x H100 80GB | 3 | 148K samples | ~5-6 hours |

## File Structure

```
gcp/
├── README.md             # This file
├── setup_remote.sh       # Step 2: environment setup
├── download_data.sh      # Step 3: download & process datasets
├── prep_training.sh      # Step 4: model weights + pre-tokenize
└── train_7b_fullft.sh    # Step 5: 8x H100 DDP training
```

## Troubleshooting

### NCCL Errors
```bash
export NCCL_DEBUG=INFO       # verbose logging
export NCCL_IB_DISABLE=1     # if InfiniBand issues
```

### OOM on H100
- Reduce `batch_size` from 2 to 1 in the config
- Reduce `max_length` from 1024 to 768

### Resume from Checkpoint
The training script auto-detects checkpoints in the output directory and resumes.
Just re-run `bash gcp/train_7b_fullft.sh`.
