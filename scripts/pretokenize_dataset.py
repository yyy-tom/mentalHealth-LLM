#!/usr/bin/env python3
"""
Pre-tokenize the dataset to save time during training.
Run this locally or on a CPU node before GPU training.

Usage:
    python scripts/pretokenize_dataset.py --config configs/qwen_7b_8x2080ti.json
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_from_disk
from transformers import AutoTokenizer


def format_prompt(example, tokenizer):
    """Format the training example into a prompt using chat template."""
    instruction = example["instruction"]
    output = example["output"]

    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize a batch of examples."""
    batch_size = len(examples["instruction"])

    # Format prompts
    prompts = []
    for i in range(batch_size):
        example = {
            "instruction": examples["instruction"][i],
            "output": examples["output"][i],
        }
        prompts.append(format_prompt(example, tokenizer))

    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )

    # Labels are same as input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset for training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_suffix", type=str, default="_tokenized",
                        help="Suffix for tokenized dataset directory")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    model_name = config["model_name"]
    dataset_path = config["dataset_path"]
    max_length = config.get("max_length", 1024)

    # Resolve dataset path
    project_root = Path(__file__).parent.parent
    if not os.path.isabs(dataset_path):
        dataset_path = str(project_root / dataset_path)

    output_path = dataset_path + args.output_suffix

    print("=" * 60)
    print("Pre-tokenizing Dataset")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Max length: {max_length}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Check if already tokenized
    if os.path.exists(output_path):
        print(f"\nTokenized dataset already exists at {output_path}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Skipping tokenization.")
            return

    # Load tokenizer
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer loaded. Vocab size: {len(tokenizer):,}")

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"Train: {len(dataset['train']):,}, Validation: {len(dataset['validation']):,}")

    # Tokenize
    print(f"\nTokenizing with max_length={max_length}...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
        num_proc=4,  # Use multiple CPU cores
    )

    # Print stats
    train_lengths = [len(x) for x in tokenized_dataset["train"]["input_ids"]]
    avg_len = sum(train_lengths) / len(train_lengths)
    max_len = max(train_lengths)
    truncated = sum(1 for l in train_lengths if l == max_length)

    print(f"\nTokenization stats:")
    print(f"  Average length: {avg_len:.1f} tokens")
    print(f"  Max length: {max_len} tokens")
    print(f"  Truncated samples: {truncated:,} ({100*truncated/len(train_lengths):.1f}%)")

    # Save
    print(f"\nSaving tokenized dataset to {output_path}...")
    tokenized_dataset.save_to_disk(output_path)

    print("\n" + "=" * 60)
    print("Pre-tokenization complete!")
    print("=" * 60)
    print(f"\nTokenized dataset saved to: {output_path}")
    print("This will be automatically used during training.")


if __name__ == "__main__":
    main()
