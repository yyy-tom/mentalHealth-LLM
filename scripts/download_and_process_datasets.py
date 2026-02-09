#!/usr/bin/env python3
"""
Download and process datasets directly on the remote cluster.

Since datasets cannot be pushed via git, this script:
1. Downloads the Empathetic Counseling Dataset from HuggingFace
2. Downloads the Suicide Watch dataset from Kaggle (requires kaggle API key)
3. Processes both into instruction-tuning format
4. Combines them with any existing processed datasets
5. Optionally pre-tokenizes the combined dataset

Usage:
    # Download and process all datasets
    python scripts/download_and_process_datasets.py

    # Skip Kaggle (if no API key available, use manual CSV path instead)
    python scripts/download_and_process_datasets.py --skip_kaggle

    # Use a local CSV for Suicide Watch instead of downloading
    python scripts/download_and_process_datasets.py --suicide_csv /path/to/Suicide_Detection.csv

    # Only download, don't combine or tokenize
    python scripts/download_and_process_datasets.py --no_combine --no_tokenize

    # Full pipeline including pre-tokenization
    python scripts/download_and_process_datasets.py --tokenize --config configs/qwen_7b_8x2080ti.json
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Counselor system prompt (shared with other scripts)
# ---------------------------------------------------------------------------
COUNSELOR_INSTRUCTION = """You are a compassionate and professional mental health counselor. Please provide helpful, empathetic, and evidence-based advice.

APPROACH TO FOLLOW (YES):
• Validate, normalize, empathize, and reflect back to the client
• Use reassuring, sympathizing, affirming language
• Keep responses shorter and more conversational (closer to human messaging behavior)
• Ask elaborative, open-ended questions to understand the client better
• Ask questions about potential options to let the client think through
• Allow the user to talk through their problems instead of giving clear directions or solutions

IMPORTANT TO AVOID (NO):
• Overly strong framing, directive tone, or generic advice
• Using diagnostic labels (e.g., "you experience symptoms of social anxiety")"""

# ---------------------------------------------------------------------------
# Crisis response templates (same as prepare_crisis_dataset.py)
# ---------------------------------------------------------------------------
CRISIS_RESPONSES = [
    "I hear that you're going through an incredibly difficult time right now. Your feelings are valid, and I'm glad you're reaching out. Please know that you don't have to face this alone. If you're having thoughts of suicide, please reach out to a crisis helpline - in the US, you can call or text 988 for the Suicide & Crisis Lifeline, available 24/7. Would you like to talk more about what you're experiencing?",
    "Thank you for sharing something so personal with me. What you're feeling sounds really overwhelming, and I want you to know that help is available. Crisis support is just a call away - the 988 Suicide & Crisis Lifeline is available 24/7. In the meantime, can you tell me more about what's been happening?",
    "I can sense how much pain you're in right now. These feelings, while incredibly difficult, are something that professional support can help with. Please consider reaching out to the 988 Suicide & Crisis Lifeline or going to your nearest emergency room if you feel unsafe. You matter, and there are people who want to help.",
    "It sounds like you're carrying a tremendous weight right now. I want you to know that reaching out like this shows real strength. If you're having thoughts of ending your life, please contact emergency services or a crisis line like 988 immediately. While I'm here to listen, trained crisis counselors can provide the immediate support you need.",
    "I'm really concerned about what you're sharing, and I want to make sure you're safe. Your life has value, even when it doesn't feel that way. Please reach out to a crisis service - call or text 988, or chat at 988lifeline.org. Can you tell me if you're currently safe?",
]

SUPPORTIVE_RESPONSES = [
    "Thank you for sharing that with me. It sounds like you're dealing with some challenging emotions. Would you like to explore what's behind these feelings?",
    "I appreciate you opening up about this. What you're experiencing is understandable given the circumstances. How have you been coping with these feelings?",
    "It takes courage to talk about these things. Your feelings are valid. What kind of support do you think would be most helpful for you right now?",
    "I hear you, and what you're going through sounds difficult. Remember that it's okay to not be okay sometimes. What has helped you get through tough times before?",
    "Thank you for trusting me with this. Processing these emotions is an important step. Is there anything specific you'd like to work through together?",
]


# ===================================================================
# 1. Empathetic Counseling Dataset (HuggingFace)
# ===================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if pd.isna(text) or text is None:
        return ""
    text = re.sub(r"<[^>]+>", "", str(text))
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("&#34;", '"').replace("&#39;", "'")
    return text.strip()


def download_and_process_empathetic(output_dir: str, seed: int = 42) -> str:
    """
    Download the Empathetic Counseling Dataset from HuggingFace and process it
    into instruction-tuning format.

    Dataset: https://huggingface.co/datasets/LuangMV97/Empathetic_counseling_Dataset
    Columns: input (user utterance), label (counselor response)
    Splits:  train (30,937), test (7,736)
    """
    output_path = os.path.join(output_dir, "empathetic_counseling_processed")

    if os.path.exists(output_path):
        print(f"\n[Empathetic] Already processed at {output_path}, skipping download.")
        return output_path

    print("\n" + "=" * 60)
    print("Downloading Empathetic Counseling Dataset from HuggingFace")
    print("=" * 60)

    hf_dataset = load_dataset("LuangMV97/Empathetic_counseling_Dataset")
    print(f"Downloaded. Splits: {list(hf_dataset.keys())}")

    processed = {"train": [], "validation": []}

    # Map HF splits → our splits  (HF has train/test, we use train/validation)
    split_mapping = {"train": "train", "test": "validation"}

    for hf_split, our_split in split_mapping.items():
        if hf_split not in hf_dataset:
            continue
        split_data = hf_dataset[hf_split]
        print(f"  Processing {hf_split} ({len(split_data):,} rows) → {our_split}")

        for row in split_data:
            user_input = clean_text(row.get("input", ""))
            counselor_label = clean_text(row.get("label", ""))

            if len(user_input) < 10 or len(counselor_label) < 20:
                continue  # skip very short / empty rows

            instruction = (
                f"{COUNSELOR_INSTRUCTION}\n\n"
                f"Question: {user_input}\n\n"
                f"Response:"
            )

            processed[our_split].append({
                "instruction": instruction,
                "input": "",
                "output": counselor_label,
                "topic": "empathetic_counseling",
                "upvotes": 0,
                "question_id": f"empathetic_{hf_split}_{len(processed[our_split])}",
            })

    train_ds = Dataset.from_list(processed["train"])
    val_ds = Dataset.from_list(processed["validation"])

    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds})
    Path(output_path).mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_path)

    print(f"  ✓ Saved: train={len(train_ds):,}, validation={len(val_ds):,}")
    print(f"  ✓ Path:  {output_path}")
    return output_path


# ===================================================================
# 2. Suicide Watch Dataset (Kaggle)
# ===================================================================

def download_suicide_watch_kaggle(output_csv: str) -> str:
    """
    Download Suicide Watch dataset using the Kaggle API.

    Dataset: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data
    Requires: kaggle CLI + ~/.kaggle/kaggle.json

    Returns the path to the downloaded CSV.
    """
    if os.path.exists(output_csv):
        print(f"\n[SuicideWatch] CSV already exists at {output_csv}, skipping download.")
        return output_csv

    print("\n" + "=" * 60)
    print("Downloading Suicide Watch Dataset from Kaggle")
    print("=" * 60)

    import subprocess

    csv_dir = os.path.dirname(output_csv)
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "nikhileswarkomati/suicide-watch",
                "-p", csv_dir,
                "--unzip",
            ],
            check=True,
        )
        # Kaggle may name the file differently; find the CSV
        for f in os.listdir(csv_dir):
            if f.lower().endswith(".csv"):
                actual_path = os.path.join(csv_dir, f)
                if actual_path != output_csv:
                    os.rename(actual_path, output_csv)
                print(f"  ✓ Downloaded to {output_csv}")
                return output_csv

        raise FileNotFoundError("CSV not found after Kaggle download")

    except FileNotFoundError:
        print("  ✗ 'kaggle' CLI not found. Install with: pip install kaggle")
        print("    Then place your API key at ~/.kaggle/kaggle.json")
        print("    Or use --suicide_csv /path/to/Suicide_Detection.csv")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Kaggle download failed: {e}")
        print("    Make sure ~/.kaggle/kaggle.json exists and is valid.")
        print("    Or download manually and use --suicide_csv /path/to/Suicide_Detection.csv")
        sys.exit(1)


def process_suicide_watch(
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> str:
    """
    Process the Kaggle Suicide Watch CSV into instruction-tuning format.

    CSV columns: Unnamed: 0, text, class
    class values: 'suicide' or 'non-suicide'
    """
    output_path = os.path.join(output_dir, "crisis_detection_processed")

    if os.path.exists(output_path):
        print(f"\n[SuicideWatch] Already processed at {output_path}, skipping.")
        return output_path

    print("\n" + "=" * 60)
    print("Processing Suicide Watch Dataset")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows from {csv_path}")

    random.seed(seed)
    processed_data = []

    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if len(text) < 10:
            continue

        is_crisis = row["class"] == "suicide"

        instruction = (
            "You are a compassionate mental health counselor. "
            "Read the following message and provide an appropriate, supportive response. "
            "If the message indicates a crisis or suicidal thoughts, provide crisis resources.\n\n"
            f"Message: {text}"
        )

        response = random.choice(CRISIS_RESPONSES if is_crisis else SUPPORTIVE_RESPONSES)

        processed_data.append({
            "instruction": instruction,
            "input": "",
            "output": response,
            "topic": "crisis_support" if is_crisis else "general_support",
            "upvotes": 0,
            "question_id": f"crisis_{row.get('Unnamed: 0', len(processed_data))}",
            "is_crisis": is_crisis,
        })

    random.shuffle(processed_data)
    split_idx = int(len(processed_data) * train_ratio)
    train_data = processed_data[:split_idx]
    val_data = processed_data[split_idx:]

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds})
    Path(output_path).mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_path)

    crisis_count = sum(1 for d in processed_data if d["is_crisis"])
    print(f"  ✓ Crisis: {crisis_count:,}, Non-crisis: {len(processed_data) - crisis_count:,}")
    print(f"  ✓ Saved: train={len(train_data):,}, validation={len(val_data):,}")
    print(f"  ✓ Path:  {output_path}")
    return output_path


# ===================================================================
# 3. Combine all processed datasets
# ===================================================================

def combine_datasets(dataset_dirs: list, output_dir: str) -> str:
    """Combine multiple processed datasets into one."""
    output_path = os.path.join(output_dir, "all_mental_health_combined")

    print("\n" + "=" * 60)
    print("Combining All Processed Datasets")
    print("=" * 60)

    loaded = []
    names = []

    for d in dataset_dirs:
        abs_path = d if os.path.isabs(d) else os.path.join(PROJECT_ROOT, d)
        if not os.path.exists(abs_path):
            print(f"  ⚠ Skipping (not found): {abs_path}")
            continue
        try:
            ds = load_from_disk(abs_path)
            if "train" not in ds:
                print(f"  ⚠ Skipping (no train split): {abs_path}")
                continue

            # Normalize question_id to string
            if "question_id" in ds["train"].features:
                from datasets import Value

                def _to_str(example):
                    example["question_id"] = str(example.get("question_id", ""))
                    return example

                ds = ds.map(_to_str, desc=f"Normalizing {os.path.basename(abs_path)}")
                new_features = ds["train"].features.copy()
                new_features["question_id"] = Value("string")
                ds = ds.cast(new_features)

            # Drop is_crisis column if present (not all datasets have it)
            for split in ds:
                if "is_crisis" in ds[split].column_names:
                    ds[split] = ds[split].remove_columns(["is_crisis"])

            loaded.append(ds)
            t = len(ds["train"])
            v = len(ds.get("validation", []))
            names.append(os.path.basename(abs_path))
            print(f"  ✓ {os.path.basename(abs_path)}: train={t:,}, val={v:,}")
        except Exception as e:
            print(f"  ✗ Error loading {abs_path}: {e}")

    if not loaded:
        print("  No datasets loaded — nothing to combine.")
        return ""

    # Combine train splits
    combined_train = concatenate_datasets([ds["train"] for ds in loaded])

    # Combine validation splits (only from datasets that have them)
    val_splits = [
        ds["validation"] for ds in loaded
        if "validation" in ds and len(ds["validation"]) > 0
    ]
    if val_splits:
        combined_val = concatenate_datasets(val_splits)
    else:
        combined_val = Dataset.from_dict({c: [] for c in combined_train.column_names})

    combined = DatasetDict({"train": combined_train, "validation": combined_val})
    Path(output_path).mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(output_path)

    print(f"\n  ✓ Combined train:      {len(combined_train):,}")
    print(f"  ✓ Combined validation: {len(combined_val):,}")
    print(f"  ✓ Datasets included:   {', '.join(names)}")
    print(f"  ✓ Path: {output_path}")
    return output_path


# ===================================================================
# 4. Pre-tokenize
# ===================================================================

def pretokenize(dataset_path: str, config_path: str):
    """Pre-tokenize the combined dataset using the model's tokenizer."""
    from transformers import AutoTokenizer

    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config["model_name"]
    max_length = config.get("max_length", 1024)
    output_path = dataset_path + "_tokenized"

    print("\n" + "=" * 60)
    print("Pre-tokenizing Dataset")
    print("=" * 60)
    print(f"  Model:      {model_name}")
    print(f"  Dataset:    {dataset_path}")
    print(f"  Max length: {max_length}")
    print(f"  Output:     {output_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(dataset_path)
    print(f"  Train: {len(dataset['train']):,}, Val: {len(dataset['validation']):,}")

    def _format_and_tokenize(examples):
        batch_size = len(examples["instruction"])
        prompts = []
        for i in range(batch_size):
            messages = [
                {"role": "user", "content": examples["instruction"][i]},
                {"role": "assistant", "content": examples["output"][i]},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        _format_and_tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
        num_proc=4,
    )

    lengths = [len(x) for x in tokenized["train"]["input_ids"]]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    truncated = sum(1 for l in lengths if l == max_length)

    print(f"  Avg length:  {avg_len:.1f} tokens")
    print(f"  Truncated:   {truncated:,} ({100 * truncated / max(len(lengths), 1):.1f}%)")

    tokenized.save_to_disk(output_path)
    print(f"  ✓ Saved tokenized dataset to {output_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download, process, combine, and tokenize datasets on remote.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets"),
        help="Base directory for datasets (default: <project>/datasets)",
    )
    parser.add_argument(
        "--skip_kaggle",
        action="store_true",
        help="Skip downloading/processing the Kaggle Suicide Watch dataset",
    )
    parser.add_argument(
        "--suicide_csv",
        type=str,
        default=None,
        help="Path to an already-downloaded Suicide_Detection.csv (skips Kaggle download)",
    )
    parser.add_argument(
        "--no_combine",
        action="store_true",
        help="Don't combine datasets after processing",
    )
    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Pre-tokenize the combined dataset after combining",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "qwen_7b_8x2080ti.json"),
        help="Config file for pre-tokenization (model name, max_length, etc.)",
    )
    parser.add_argument(
        "--extra_datasets",
        type=str,
        nargs="*",
        default=None,
        help="Additional already-processed dataset dirs to include in combining",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    datasets_dir = args.datasets_dir

    print("=" * 60)
    print("  Remote Dataset Download & Processing Pipeline")
    print("=" * 60)
    print(f"  Project root:  {PROJECT_ROOT}")
    print(f"  Datasets dir:  {datasets_dir}")
    print()

    processed_paths = []

    # ---- 1. Empathetic Counseling (HuggingFace) ----
    emp_path = download_and_process_empathetic(datasets_dir, seed=args.seed)
    processed_paths.append(emp_path)

    # ---- 2. Suicide Watch (Kaggle) ----
    if not args.skip_kaggle:
        csv_path = args.suicide_csv or os.path.join(
            datasets_dir, "kaggle_suicide_watch", "Suicide_Detection.csv"
        )
        if not args.suicide_csv:
            download_suicide_watch_kaggle(csv_path)
        crisis_path = process_suicide_watch(csv_path, datasets_dir, seed=args.seed)
        processed_paths.append(crisis_path)
    else:
        print("\n[SuicideWatch] Skipped (--skip_kaggle).")

    # ---- 3. Include any extra already-processed datasets ----
    if args.extra_datasets:
        for ed in args.extra_datasets:
            abs_ed = ed if os.path.isabs(ed) else os.path.join(PROJECT_ROOT, ed)
            processed_paths.append(abs_ed)

    # Also pick up existing processed datasets if they exist
    existing = [
        os.path.join(datasets_dir, "counsel_chat_processed"),
        os.path.join(datasets_dir, "mental_health_with_crisis"),
    ]
    for ep in existing:
        if os.path.exists(ep) and ep not in processed_paths:
            processed_paths.append(ep)

    # ---- 4. Combine ----
    combined_path = ""
    if not args.no_combine and len(processed_paths) > 0:
        combined_path = combine_datasets(processed_paths, datasets_dir)
    else:
        print("\n[Combine] Skipped.")

    # ---- 5. Pre-tokenize ----
    if args.tokenize and combined_path:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)
        pretokenize(combined_path, config_path)
    elif args.tokenize and not combined_path:
        print("\n[Tokenize] Skipped — no combined dataset path.")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Processed datasets:")
    for p in processed_paths:
        exists = "✓" if os.path.exists(p) else "✗"
        print(f"    {exists} {p}")
    if combined_path:
        print(f"\n  Combined dataset: {combined_path}")
    if args.tokenize and combined_path:
        print(f"  Tokenized dataset: {combined_path}_tokenized")
    print()


if __name__ == "__main__":
    main()
