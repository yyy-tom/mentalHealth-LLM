# All Mental Health Combined Dataset

A comprehensive mental health counseling dataset combining multiple sources for training conversational AI models.

## Overview

| Metric | Value |
|--------|-------|
| **Train samples** | 148,226 |
| **Validation samples** | 16,472 |
| **Total samples** | 164,698 |
| **Dataset size** | ~192 MB |
| **Format** | HuggingFace Datasets (Arrow) |

## Data Schema

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | string | System prompt with context (counselor role + topic) |
| `input` | string | Additional input context (often empty) |
| `output` | string | Counselor's response |
| `topic` | string | Mental health topic/category |
| `upvotes` | int64 | Quality indicator from source data |
| `question_id` | string | Unique identifier for tracking |

## Source Datasets

This combined dataset was created by merging the following processed datasets:

1. **Counsel Chat** - Professional counselor Q&A from online platforms
2. **MentalChat16k** - Mental health conversation dataset
3. **Kaggle Mental Health (Nguyen)** - Processed mental health discussions
4. **ESConv** - Emotional support conversations
5. **AMOD** - Mental health advice dataset
6. **PsyDial** - Chinese psychological counseling dialogs
7. **CACTUS** - Counseling conversation dataset

## Topic Distribution

| Topic | Samples | Percentage |
|-------|---------|------------|
| (General/Unlabeled) | 139,748 | 94.3% |
| Ongoing depression | 2,297 | 1.5% |
| Job crisis | 1,886 | 1.3% |
| Breakup with partner | 1,596 | 1.1% |
| Problems with friends | 1,005 | 0.7% |
| Academic pressure | 944 | 0.6% |
| Sleep Problems | 197 | 0.1% |
| Issues with Children | 116 | <0.1% |
| Conflict with parents | 87 | <0.1% |
| Alcohol Abuse | 78 | <0.1% |
| Other topics | ~272 | <0.2% |

## Sample Entry

```json
{
  "instruction": "You are a compassionate and professional mental health counselor. Please provide helpful, empathetic, and evidence-based advice for the following situation.\n\nContext: addiction\n\nQuestion: For some reason I always need to be doing something...",
  "input": "",
  "output": "Start by spending small amounts of time with your own thoughts and feelings. Always being focused on what a phone offers keeps people at a distance from knowing themselves...",
  "topic": "addiction",
  "upvotes": 0,
  "question_id": "839"
}
```

## Usage

### Loading the Dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/all_mental_health_combined")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
```

### Formatting for Training

The dataset is structured for instruction fine-tuning. A typical training prompt format:

```python
def format_prompt(example):
    prompt = f"{example['instruction']}"
    if example['input']:
        prompt += f"\n\n{example['input']}"
    prompt += f"\n\nResponse: {example['output']}"
    return prompt
```

## Data Quality Notes

- **Multilingual**: Contains primarily English with some Chinese content (PsyDial)
- **Varied response length**: Responses range from brief advice to detailed explanations
- **Professional tone**: Most responses follow clinical/counseling best practices
- **Topic coverage**: Depression, relationships, anxiety, addiction, and more

## Regenerating the Dataset

To regenerate or customize the combined dataset:

```bash
# Combine all datasets (including Chinese)
python scripts/combine_all_datasets.py

# Exclude Chinese datasets
python scripts/combine_all_datasets.py --exclude_chinese

# Combine specific datasets only
python scripts/combine_all_datasets.py --datasets counsel_chat_processed esconv_processed
```
