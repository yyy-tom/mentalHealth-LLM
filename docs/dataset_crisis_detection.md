# Crisis Detection Dataset

A processed crisis detection dataset derived from the Kaggle SuicideWatch dataset for training mental health AI models.

## Overview

| Metric | Value |
|--------|-------|
| **Train samples** | 208,866 |
| **Validation samples** | 23,208 |
| **Total samples** | 232,074 |
| **Crisis samples** | 116,037 (50%) |
| **Non-crisis samples** | 116,037 (50%) |
| **Source** | [Kaggle SuicideWatch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) |

## Purpose

This dataset trains the model to:
1. **Recognize crisis language** - Identify messages indicating suicidal ideation or self-harm
2. **Provide appropriate responses** - Generate supportive responses with crisis resources
3. **Differentiate severity** - Distinguish between general mental health concerns and crisis situations

## Data Schema

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | string | Counselor prompt with user's message |
| `input` | string | Empty (for compatibility) |
| `output` | string | Appropriate supportive response |
| `topic` | string | "crisis_support" or "general_support" |
| `upvotes` | int64 | Quality indicator (set to 0) |
| `question_id` | string | Unique identifier |

## Response Types

### Crisis Responses (for suicidal content)
- Acknowledge the person's pain
- Validate their feelings
- Provide crisis resources (988 Suicide & Crisis Lifeline)
- Express care and concern
- Ask about immediate safety

### Supportive Responses (for non-crisis content)
- Thank the person for sharing
- Validate their experiences
- Offer to explore feelings further
- Ask about coping strategies

## Sample Entry

```json
{
  "instruction": "You are a compassionate mental health counselor. Read the following message and provide an appropriate, supportive response. If the message indicates a crisis or suicidal thoughts, provide crisis resources.\n\nMessage: I feel like there's no point anymore...",
  "input": "",
  "output": "I hear that you're going through an incredibly difficult time right now. Your feelings are valid, and I'm glad you're reaching out. Please know that you don't have to face this alone. If you're having thoughts of suicide, please reach out to a crisis helpline - in the US, you can call or text 988 for the Suicide & Crisis Lifeline, available 24/7. Would you like to talk more about what you're experiencing?",
  "topic": "crisis_support",
  "upvotes": 0,
  "question_id": "crisis_12345"
}
```

## Source Data

The original Kaggle SuicideWatch dataset contains:
- **232,074 Reddit posts** from r/SuicideWatch (2008-2021)
- **Balanced classes**: 116k suicide-related, 116k non-suicidal
- **Raw text** with minimal preprocessing

## Processing Steps

1. Load raw CSV from Kaggle
2. Convert to instruction-tuning format
3. Generate appropriate responses based on classification
4. Split into train/validation (90/10)
5. Save as HuggingFace Dataset

## Usage

### Regenerate the dataset

```bash
python scripts/prepare_crisis_dataset.py
```

### Load the dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/crisis_detection_processed")
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
```

## Combined Dataset

The crisis data is combined with the mental health counseling data to create:

**`datasets/mental_health_with_crisis`**

| Split | Samples |
|-------|---------|
| Train | 357,092 |
| Validation | 39,680 |
| **Total** | **396,772** |

This combined dataset provides comprehensive coverage of both general mental health support and crisis intervention scenarios.
