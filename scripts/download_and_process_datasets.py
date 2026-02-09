#!/usr/bin/env python3
"""
Download, process, combine, and pre-tokenize ALL mental health datasets on the
remote cluster.

This script rebuilds datasets/all_mental_health_combined from scratch by:
  1. Using existing *_processed/ folders where the processing is acceptable
  2. Re-downloading & processing datasets that need fixing or are new
  3. Combining everything into datasets/all_mental_health_combined/
  4. Optionally pre-tokenizing for training

Dataset inventory:
  ──────────────────────────────────────────────────────────────────────
  REUSE existing processed (no re-download needed):
    datasets/counsel_chat_processed/          Counsel Chat (single-turn Q&A)
    datasets/amod_processed/                  Amod mental health (single-turn Q&A)
    datasets/esconv_processed/                ESConv (multi-turn, strategy-annotated)
    datasets/cactus_processed/                CACTUS CBT (multi-turn, CBT techniques)
    datasets/kaggle_mental_health_nguyen_processed_combined/   Kaggle MH CSV

  RE-PROCESS (fix issues or are new):
    datasets/mentalchat16k_processed/         MentalChat16K — relax length filters
    datasets/empathetic_counseling_processed/  Empathetic Counseling (download from HF)
    datasets/crisis_detection_processed/       Suicide Watch (download from Kaggle)

  EXCLUDED (by default):
    datasets/psydial_processed/               PsyDial-D4 — Chinese language
  ──────────────────────────────────────────────────────────────────────

Usage:
    # Full rebuild + tokenize
    python scripts/download_and_process_datasets.py --tokenize --config configs/qwen_7b_8x2080ti.json

    # Rebuild without re-downloading (reuse all existing processed folders)
    python scripts/download_and_process_datasets.py --skip_downloads --tokenize

    # Include PsyDial (Chinese) for bilingual training
    python scripts/download_and_process_datasets.py --include_chinese --tokenize

    # Skip Kaggle (no API key)
    python scripts/download_and_process_datasets.py --skip_kaggle --tokenize
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Set HuggingFace cache BEFORE importing datasets/transformers to avoid
# writing to ~/.cache (home directory has limited quota on the cluster).
# The env vars can be overridden externally (e.g. in the SLURM script).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent.absolute()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_CACHE = _PROJECT_ROOT / ".cache" / "huggingface"

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_DEFAULT_CACHE)
if "HF_DATASETS_CACHE" not in os.environ:
    os.environ["HF_DATASETS_CACHE"] = str(Path(os.environ["HF_HOME"]) / "datasets")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Unified counselor system prompt
# ---------------------------------------------------------------------------
COUNSELOR_INSTRUCTION = """\
You are a compassionate and professional mental health counselor. \
Please provide helpful, empathetic, and evidence-based advice.

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
# Crisis response templates
# ---------------------------------------------------------------------------
CRISIS_RESPONSES = [
    "I hear that you're going through an incredibly difficult time right now. "
    "Your feelings are valid, and I'm glad you're reaching out. Please know "
    "that you don't have to face this alone. If you're having thoughts of "
    "suicide, please reach out to a crisis helpline - in the US, you can call "
    "or text 988 for the Suicide & Crisis Lifeline, available 24/7. Would you "
    "like to talk more about what you're experiencing?",

    "Thank you for sharing something so personal with me. What you're feeling "
    "sounds really overwhelming, and I want you to know that help is available. "
    "Crisis support is just a call away - the 988 Suicide & Crisis Lifeline is "
    "available 24/7. In the meantime, can you tell me more about what's been "
    "happening?",

    "I can sense how much pain you're in right now. These feelings, while "
    "incredibly difficult, are something that professional support can help "
    "with. Please consider reaching out to the 988 Suicide & Crisis Lifeline "
    "or going to your nearest emergency room if you feel unsafe. You matter, "
    "and there are people who want to help.",

    "It sounds like you're carrying a tremendous weight right now. I want you "
    "to know that reaching out like this shows real strength. If you're having "
    "thoughts of ending your life, please contact emergency services or a "
    "crisis line like 988 immediately. While I'm here to listen, trained "
    "crisis counselors can provide the immediate support you need.",

    "I'm really concerned about what you're sharing, and I want to make sure "
    "you're safe. Your life has value, even when it doesn't feel that way. "
    "Please reach out to a crisis service - call or text 988, or chat at "
    "988lifeline.org. Can you tell me if you're currently safe?",
]

SUPPORTIVE_RESPONSES = [
    "Thank you for sharing that with me. It sounds like you're dealing with "
    "some challenging emotions. Would you like to explore what's behind these "
    "feelings?",

    "I appreciate you opening up about this. What you're experiencing is "
    "understandable given the circumstances. How have you been coping with "
    "these feelings?",

    "It takes courage to talk about these things. Your feelings are valid. "
    "What kind of support do you think would be most helpful for you right now?",

    "I hear you, and what you're going through sounds difficult. Remember "
    "that it's okay to not be okay sometimes. What has helped you get through "
    "tough times before?",

    "Thank you for trusting me with this. Processing these emotions is an "
    "important step. Is there anything specific you'd like to work through "
    "together?",
]


# ===================================================================
# Utility
# ===================================================================

def clean_text(text) -> str:
    """Clean and normalize text content."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", "", text)          # strip HTML
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("&#34;", '"').replace("&#39;", "'")
    return text.strip()


# ===================================================================
# 1. Empathetic Counseling Dataset (HuggingFace)
#    https://huggingface.co/datasets/LuangMV97/Empathetic_counseling_Dataset
#    Columns: input, label  |  Splits: train (30,937), test (7,736)
#    Single-turn Q&A — concatenation of empathetic_dialogues + Amod +
#    Psych8k + counsel-chat
# ===================================================================

def process_empathetic(output_dir: str, force: bool = False) -> str:
    """Download and process the Empathetic Counseling dataset from HF."""
    output_path = os.path.join(output_dir, "empathetic_counseling_processed")
    if os.path.exists(output_path) and not force:
        print(f"  [Empathetic] Already exists, reusing {output_path}")
        return output_path

    print("\n" + "=" * 60)
    print("  Downloading & Processing: Empathetic Counseling Dataset")
    print("=" * 60)

    hf_ds = load_dataset("LuangMV97/Empathetic_counseling_Dataset")
    processed = {"train": [], "validation": []}
    split_map = {"train": "train", "test": "validation"}

    for hf_split, our_split in split_map.items():
        if hf_split not in hf_ds:
            continue
        data = hf_ds[hf_split]
        print(f"    {hf_split} -> {our_split}: {len(data):,} rows")
        for i, row in enumerate(data):
            user = clean_text(row.get("input", ""))
            resp = clean_text(row.get("label", ""))
            if len(user) < 10 or len(resp) < 20:
                continue
            processed[our_split].append({
                "instruction": f"{COUNSELOR_INSTRUCTION}\n\nQuestion: {user}\n\nResponse:",
                "input": "",
                "output": resp,
                "topic": "empathetic_counseling",
                "upvotes": 0,
                "question_id": f"empathetic_{hf_split}_{i}",
            })

    _save_dataset(processed, output_path)
    return output_path


# ===================================================================
# 2. Suicide Watch / Crisis Detection (Kaggle)
#    https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
#    Columns: text, class ('suicide' | 'non-suicide')
#    Classification -> instruction-tuning with template responses
# ===================================================================

def download_suicide_watch_csv(output_csv: str) -> str:
    """Download the Suicide Watch CSV from Kaggle via CLI."""
    if os.path.exists(output_csv):
        print(f"  [SuicideWatch] CSV exists at {output_csv}")
        return output_csv
    print("\n  Downloading Suicide Watch CSV from Kaggle...")
    import subprocess
    csv_dir = os.path.dirname(output_csv)
    Path(csv_dir).mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "nikhileswarkomati/suicide-watch",
                "-p", csv_dir, "--unzip",
            ],
            check=True,
        )
        for f in os.listdir(csv_dir):
            if f.lower().endswith(".csv"):
                actual = os.path.join(csv_dir, f)
                if actual != output_csv:
                    os.rename(actual, output_csv)
                return output_csv
        raise FileNotFoundError("CSV not found after download")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  ! Kaggle download failed: {e}")
        print("    Install kaggle CLI and set up ~/.kaggle/kaggle.json")
        print("    Or use --suicide_csv /path/to/Suicide_Detection.csv")
        print("    Skipping Suicide Watch dataset (pipeline continues).")
        return ""


def process_suicide_watch(
    csv_path: str, output_dir: str, force: bool = False, seed: int = 42
) -> str:
    """Process Suicide Watch CSV into instruction-tuning format."""
    output_path = os.path.join(output_dir, "crisis_detection_processed")
    if os.path.exists(output_path) and not force:
        print(f"  [SuicideWatch] Already exists, reusing {output_path}")
        return output_path

    print("\n" + "=" * 60)
    print("  Processing: Suicide Watch Dataset")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"    Loaded {len(df):,} rows")

    random.seed(seed)
    data = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if len(text) < 10:
            continue
        is_crisis = row["class"] == "suicide"
        instruction = (
            "You are a compassionate mental health counselor. "
            "Read the following message and provide an appropriate, "
            "supportive response. If the message indicates a crisis or "
            "suicidal thoughts, provide crisis resources.\n\n"
            f"Message: {text}"
        )
        data.append({
            "instruction": instruction,
            "input": "",
            "output": random.choice(
                CRISIS_RESPONSES if is_crisis else SUPPORTIVE_RESPONSES
            ),
            "topic": "crisis_support" if is_crisis else "general_support",
            "upvotes": 0,
            "question_id": f"crisis_{row.get('Unnamed: 0', len(data))}",
        })

    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    _save_dataset(
        {"train": data[:split_idx], "validation": data[split_idx:]}, output_path
    )
    crisis_n = sum(1 for d in data if d["topic"] == "crisis_support")
    print(f"    Crisis: {crisis_n:,}, Non-crisis: {len(data) - crisis_n:,}")
    return output_path


# ===================================================================
# 3. MentalChat16K (re-process with relaxed filters)
#    https://huggingface.co/datasets/ShenLab/MentalChat16K
#    Columns: instruction, input, output  |  16,084 rows (train only)
#    Mix of: Synthetic 10K (single-turn) + Interview 6K (single-turn, shorter)
#    Issue: old script had aggressive length filters dropping valid Interview data
# ===================================================================

def process_mentalchat16k(output_dir: str, force: bool = False) -> str:
    """Download and re-process MentalChat16K with relaxed filters."""
    output_path = os.path.join(output_dir, "mentalchat16k_processed")
    if os.path.exists(output_path) and not force:
        print(f"  [MentalChat16K] Already exists, reusing {output_path}")
        return output_path

    print("\n" + "=" * 60)
    print("  Downloading & Processing: MentalChat16K")
    print("=" * 60)

    hf_ds = load_dataset("ShenLab/MentalChat16K", split="train")
    print(f"    Loaded {len(hf_ds):,} rows")
    print(f"    Columns: {hf_ds.column_names}")

    data = []
    for i, row in enumerate(hf_ds):
        # MentalChat16K has columns: instruction, input, output
        question = clean_text(row.get("input", ""))
        answer = clean_text(row.get("output", ""))

        # Relaxed: min 10 char question, min 20 char answer
        if len(question) < 10 or len(answer) < 20:
            continue
        # Upper bound: skip extremely long (likely malformed)
        if len(answer) > 4000:
            continue

        instruction = (
            f"{COUNSELOR_INSTRUCTION}\n\nQuestion: {question}\n\nResponse:"
        )
        data.append({
            "instruction": instruction,
            "input": "",
            "output": answer,
            "topic": "mentalchat16k",
            "upvotes": 0,
            "question_id": f"mentalchat_{i}",
        })

    print(f"    Valid examples: {len(data):,} / {len(hf_ds):,}")
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    _save_dataset(
        {"train": data[:split_idx], "validation": data[split_idx:]}, output_path
    )
    return output_path


# ===================================================================
# Helper: save processed dataset
# ===================================================================

def _save_dataset(splits: dict, output_path: str):
    """Save train/validation lists to disk as a DatasetDict."""
    train_ds = Dataset.from_list(splits["train"])
    val_ds = Dataset.from_list(splits["validation"])
    dd = DatasetDict({"train": train_ds, "validation": val_ds})
    Path(output_path).mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(output_path)
    print(
        f"    -> Saved: train={len(train_ds):,}, val={len(val_ds):,}"
        f" -> {output_path}"
    )


# ===================================================================
# 4. Combine all processed datasets
# ===================================================================

def combine_all(dataset_dirs: list, output_path: str) -> str:
    """Combine multiple processed datasets into one DatasetDict."""
    print("\n" + "=" * 60)
    print("  Combining All Datasets")
    print("=" * 60)

    loaded = []
    names = []

    # Canonical column set (every processed dataset must have these)
    required_cols = {
        "instruction", "input", "output", "topic", "upvotes", "question_id"
    }

    for d in dataset_dirs:
        abs_path = d if os.path.isabs(d) else os.path.join(PROJECT_ROOT, d)
        if not os.path.exists(abs_path):
            print(f"    ! Not found, skipping: {abs_path}")
            continue
        try:
            ds = load_from_disk(abs_path)
            if "train" not in ds:
                print(f"    ! No train split, skipping: {abs_path}")
                continue

            # Normalize question_id to string (some datasets use int)
            if "question_id" in ds["train"].features:
                def _to_str(ex):
                    ex["question_id"] = str(ex.get("question_id", ""))
                    return ex
                ds = ds.map(
                    _to_str,
                    desc=f"Normalize {os.path.basename(abs_path)}",
                )
                for split in ds:
                    feats = ds[split].features.copy()
                    feats["question_id"] = Value("string")
                    ds[split] = ds[split].cast(feats)

            # Drop extra columns not in the required set
            for split in ds:
                extra = set(ds[split].column_names) - required_cols
                if extra:
                    ds[split] = ds[split].remove_columns(list(extra))

            t = len(ds["train"])
            v = len(ds.get("validation", []))
            loaded.append(ds)
            name = os.path.basename(abs_path)
            names.append(name)
            print(f"    + {name}: train={t:,}, val={v:,}")
        except Exception as e:
            print(f"    x Error loading {abs_path}: {e}")

    if not loaded:
        print("    No datasets loaded - nothing to combine.")
        return ""

    combined_train = concatenate_datasets([ds["train"] for ds in loaded])
    val_splits = [
        ds["validation"]
        for ds in loaded
        if "validation" in ds and len(ds["validation"]) > 0
    ]
    combined_val = (
        concatenate_datasets(val_splits)
        if val_splits
        else Dataset.from_dict({c: [] for c in required_cols})
    )

    combined = DatasetDict({"train": combined_train, "validation": combined_val})

    # Back up old combined dataset if it exists
    if os.path.exists(output_path):
        import shutil
        backup = output_path + "_backup"
        if os.path.exists(backup):
            shutil.rmtree(backup)
        os.rename(output_path, backup)
        print(f"    i Backed up old dataset to {backup}")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(output_path)

    print(f"\n    Combined train:      {len(combined_train):,}")
    print(f"    Combined validation: {len(combined_val):,}")
    print(f"    Datasets ({len(names)}):     {', '.join(names)}")
    print(f"    Saved to: {output_path}")
    return output_path


# ===================================================================
# 5. Pre-tokenize
# ===================================================================

def pretokenize(dataset_path: str, config_path: str):
    """Pre-tokenize the combined dataset using the Qwen chat template."""
    from transformers import AutoTokenizer

    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config["model_name"]
    max_length = config.get("max_length", 1024)
    output_path = dataset_path + "_tokenized"

    print("\n" + "=" * 60)
    print("  Pre-tokenizing Dataset")
    print("=" * 60)
    print(f"    Model:      {model_name}")
    print(f"    Dataset:    {dataset_path}")
    print(f"    Max length: {max_length}")
    print(f"    Output:     {output_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(dataset_path)
    print(f"    Train: {len(dataset['train']):,}, Val: {len(dataset['validation']):,}")

    def _tok(examples):
        bs = len(examples["instruction"])
        prompts = []
        for i in range(bs):
            msgs = [
                {"role": "user", "content": examples["instruction"][i]},
                {"role": "assistant", "content": examples["output"][i]},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            )
        out = tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    # Remove old tokenized if exists
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)

    tokenized = dataset.map(
        _tok,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
        num_proc=4,
    )

    lengths = [len(x) for x in tokenized["train"]["input_ids"]]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    truncated = sum(1 for l in lengths if l == max_length)
    print(f"    Avg length:  {avg_len:.1f} tokens")
    print(f"    Truncated:   {truncated:,} ({100 * truncated / max(len(lengths), 1):.1f}%)")

    tokenized.save_to_disk(output_path)
    print(f"    -> Saved to {output_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rebuild all_mental_health_combined from scratch on remote.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets_dir", type=str,
        default=str(PROJECT_ROOT / "datasets"),
    )
    parser.add_argument(
        "--skip_kaggle", action="store_true",
        help="Skip Suicide Watch (Kaggle)",
    )
    parser.add_argument(
        "--suicide_csv", type=str, default=None,
        help="Path to existing Suicide_Detection.csv",
    )
    parser.add_argument(
        "--skip_downloads", action="store_true",
        help="Only reuse existing *_processed folders, no downloads",
    )
    parser.add_argument(
        "--force_reprocess", action="store_true",
        help="Force re-download & re-process even if output exists",
    )
    parser.add_argument(
        "--include_chinese", action="store_true",
        help="Include PsyDial (Chinese) in the combined dataset",
    )
    parser.add_argument(
        "--tokenize", action="store_true",
        help="Pre-tokenize after combining",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "qwen_7b_8x2080ti.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dd = args.datasets_dir
    force = args.force_reprocess
    random.seed(args.seed)

    print("=" * 60)
    print("  Rebuild all_mental_health_combined")
    print("=" * 60)
    print(f"  Project root:  {PROJECT_ROOT}")
    print(f"  Datasets dir:  {dd}")
    print(f"  Force re-proc: {force}")
    print(f"  Inc. Chinese:  {args.include_chinese}")
    print()

    all_paths = []

    # ---- A. Datasets that REUSE existing processed folders --------
    reuse = [
        "counsel_chat_processed",
        "amod_processed",
        "esconv_processed",
        "cactus_processed",
        "kaggle_mental_health_nguyen_processed_combined",
    ]
    if args.include_chinese:
        reuse.append("psydial_processed")

    for name in reuse:
        p = os.path.join(dd, name)
        if os.path.exists(p):
            all_paths.append(p)
            print(f"  + Reuse: {name}")
        else:
            print(f"  ! Not found (skipped): {name}")

    # ---- B. MentalChat16K — always re-process (relaxed filters) ---
    if not args.skip_downloads:
        mc_path = process_mentalchat16k(dd, force=force)
        all_paths.append(mc_path)
    else:
        mc_path = os.path.join(dd, "mentalchat16k_processed")
        if os.path.exists(mc_path):
            all_paths.append(mc_path)
            print(f"  + Reuse: mentalchat16k_processed")

    # ---- C. Empathetic Counseling (HuggingFace) -------------------
    if not args.skip_downloads:
        emp_path = process_empathetic(dd, force=force)
        all_paths.append(emp_path)
    else:
        emp_path = os.path.join(dd, "empathetic_counseling_processed")
        if os.path.exists(emp_path):
            all_paths.append(emp_path)
            print(f"  + Reuse: empathetic_counseling_processed")

    # ---- D. Suicide Watch (Kaggle) --------------------------------
    if not args.skip_kaggle and not args.skip_downloads:
        csv = args.suicide_csv or os.path.join(
            dd, "kaggle_suicide_watch", "Suicide_Detection.csv"
        )
        if not args.suicide_csv and not os.path.exists(csv):
            download_suicide_watch_csv(csv)
        if os.path.exists(csv):
            crisis_path = process_suicide_watch(
                csv, dd, force=force, seed=args.seed
            )
            all_paths.append(crisis_path)
        else:
            print(f"  ! Suicide Watch CSV not found at {csv}, skipping")
    else:
        crisis_path = os.path.join(dd, "crisis_detection_processed")
        if os.path.exists(crisis_path):
            all_paths.append(crisis_path)
            print(f"  + Reuse: crisis_detection_processed")

    # ---- E. Combine -----------------------------------------------
    output_combined = os.path.join(dd, "all_mental_health_combined")
    combined_path = combine_all(all_paths, output_combined)

    # ---- F. Pre-tokenize ------------------------------------------
    if args.tokenize and combined_path:
        cfg = (
            args.config
            if os.path.isabs(args.config)
            else os.path.join(PROJECT_ROOT, args.config)
        )
        pretokenize(combined_path, cfg)

    # ---- Summary --------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Datasets included ({len(all_paths)}):")
    for p in all_paths:
        print(f"    + {os.path.basename(p)}")
    if combined_path:
        print(f"\n  Combined:  {combined_path}")
    if args.tokenize and combined_path:
        print(f"  Tokenized: {combined_path}_tokenized")

    if args.include_chinese:
        print(
            "\n  Note: PsyDial (Chinese) is included. "
            "This is a bilingual dataset."
        )
    else:
        print(
            "\n  PsyDial (Chinese) excluded. Use --include_chinese to add it."
        )

    print()
    print("  Next steps:")
    print("    sbatch slurm/prep_training.sh      # download model weights")
    print("    sbatch slurm/train_7b_8x2080ti.sh  # start training")
    print()


if __name__ == "__main__":
    main()
