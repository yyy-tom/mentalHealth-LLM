# Mental Health Counseling LLM

Fine-tuning Qwen2.5 for mental health counseling conversations with crisis detection capabilities.

## Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| `all_mental_health_combined` | 164,698 | General mental health counseling |
| `crisis_detection_processed` | 232,074 | Crisis/suicide detection (Kaggle SuicideWatch) |
| `mental_health_with_crisis` | 396,772 | Combined dataset for full training |

See [docs/](docs/) for detailed dataset documentation.

## Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# For GPU training with CUDA
uv pip install -e ".[cuda]"
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Training

```bash
# Basic mental health training
python scripts/train_qwen_counsel.py --config configs/qwen_1.5b_lora.json

# With crisis detection data (recommended)
python scripts/train_qwen_counsel.py --config configs/qwen_1.5b_with_crisis.json
```

## Inference

```bash
python scripts/inference.py --model models/qwen2.5-1.5b-mental-health
```

## Project Structure

```
├── configs/          # Training configurations
├── datasets/         # Training datasets
├── models/           # Saved model checkpoints
├── scripts/          # Training and inference scripts
├── evaluation/       # Evaluation scripts and results
├── logs/            # Training logs
└── docs/            # Documentation
```
