#!/usr/bin/env python3
"""
Process the Kaggle Suicide Watch dataset for crisis detection training.

This script converts the classification dataset into an instruction-tuning format
that teaches the model to:
1. Recognize crisis language
2. Provide appropriate supportive responses
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from pathlib import Path
import random

# Crisis response templates for suicide-related content
CRISIS_RESPONSES = [
    "I hear that you're going through an incredibly difficult time right now. Your feelings are valid, and I'm glad you're reaching out. Please know that you don't have to face this alone. If you're having thoughts of suicide, please reach out for help now. In Hong Kong, call The Samaritan Befrienders (2389 2222) or Suicide Prevention Services (2382 0000) — both available 24/7. If you're outside Hong Kong, contact your local crisis line such as 988 (US) or your country's emergency number. Would you like to talk more about what you're experiencing?",

    "Thank you for sharing something so personal with me. What you're feeling sounds really overwhelming, and I want you to know that help is available. In Hong Kong, crisis support is just a call away — The Samaritan Befrienders Hong Kong (2389 2222) and Suicide Prevention Services (2382 0000) are available 24/7. Outside Hong Kong, please contact your local crisis line such as 988 in the US. In the meantime, can you tell me more about what's been happening?",

    "I can sense how much pain you're in right now. These feelings, while incredibly difficult, are something that professional support can help with. In Hong Kong, please call The Samaritan Befrienders (2389 2222), Suicide Prevention Services (2382 0000), or the Hospital Authority Mental Health Line (2466 7350). If you feel unsafe, call 999 for emergency services. Outside Hong Kong, contact your local emergency services or crisis line. You matter, and there are people who want to help.",

    "It sounds like you're carrying a tremendous weight right now. I want you to know that reaching out like this shows real strength. If you're having thoughts of ending your life, please contact a crisis line immediately — in Hong Kong, call The Samaritan Befrienders (2389 2222) or Suicide Prevention Services (2382 0000), both 24/7. For emergencies, call 999 in Hong Kong or your local emergency number. While I'm here to listen, trained crisis counselors can provide the immediate support you need.",

    "I'm really concerned about what you're sharing, and I want to make sure you're safe. Your life has value, even when it doesn't feel that way. In Hong Kong, please reach out to The Samaritan Befrienders (2389 2222) or Suicide Prevention Services (2382 0000) — both available around the clock. Outside Hong Kong, call your local crisis line such as 988 in the US. Can you tell me if you're currently safe?",
]

# Supportive responses for non-crisis mental health content
SUPPORTIVE_RESPONSES = [
    "Thank you for sharing that with me. It sounds like you're dealing with some challenging emotions. Would you like to explore what's behind these feelings?",

    "I appreciate you opening up about this. What you're experiencing is understandable given the circumstances. How have you been coping with these feelings?",

    "It takes courage to talk about these things. Your feelings are valid. What kind of support do you think would be most helpful for you right now?",

    "I hear you, and what you're going through sounds difficult. Remember that it's okay to not be okay sometimes. What has helped you get through tough times before?",

    "Thank you for trusting me with this. Processing these emotions is an important step. Is there anything specific you'd like to work through together?",
]


def process_crisis_dataset(
    input_path: str = "datasets/kaggle_suicide_watch/Suicide_Detection.csv",
    output_path: str = "datasets/crisis_detection_processed",
    train_ratio: float = 0.9,
    seed: int = 42
):
    """Process the suicide detection dataset for instruction tuning."""

    print("=" * 60)
    print("Processing Crisis Detection Dataset")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} samples")

    # Process into instruction format
    print("\nConverting to instruction format...")
    random.seed(seed)

    processed_data = []

    for _, row in df.iterrows():
        text = row['text']
        is_crisis = row['class'] == 'suicide'

        # Create instruction
        instruction = (
            "You are a compassionate mental health counselor. "
            "Read the following message and provide an appropriate, supportive response. "
            "If the message indicates a crisis or suicidal thoughts, provide crisis resources.\n\n"
            f"Message: {text}"
        )

        # Select appropriate response
        if is_crisis:
            response = random.choice(CRISIS_RESPONSES)
            topic = "crisis_support"
        else:
            response = random.choice(SUPPORTIVE_RESPONSES)
            topic = "general_support"

        processed_data.append({
            "instruction": instruction,
            "input": "",
            "output": response,
            "topic": topic,
            "upvotes": 0,
            "question_id": f"crisis_{row['Unnamed: 0']}",
            "is_crisis": is_crisis
        })

    # Create dataset
    print(f"\nCreated {len(processed_data):,} processed samples")

    # Shuffle and split
    random.shuffle(processed_data)
    split_idx = int(len(processed_data) * train_ratio)

    train_data = processed_data[:split_idx]
    val_data = processed_data[split_idx:]

    print(f"Train: {len(train_data):,}, Validation: {len(val_data):,}")

    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    # Save
    print(f"\nSaving to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"Crisis samples: {sum(1 for d in processed_data if d['is_crisis']):,}")
    print(f"Non-crisis samples: {sum(1 for d in processed_data if not d['is_crisis']):,}")

    return dataset


if __name__ == "__main__":
    process_crisis_dataset()
