#!/usr/bin/env python3
"""
Prepare skill-specific datasets for LoRA adapter training.

Splits and combines existing processed datasets into per-skill subsets
under datasets/skills/{skill_name}/ in HuggingFace DatasetDict format.

Skills:
  crisis-intervention   -> crisis_detection_processed (topic=crisis_support)
  general-support       -> crisis_detection_processed (topic=general_support)
  cbt-therapy           -> cactus_processed
  empathetic-listening  -> esconv_processed
  psychoeducation       -> mentalchat16k_processed
  professional-counseling -> counsel_chat + amod + kaggle_nguyen combined

Usage:
    python scripts/prepare_skill_datasets.py
    python scripts/prepare_skill_datasets.py --output_dir datasets/skills
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# HuggingFace cache setup (before importing datasets/transformers)
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

from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_from_disk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = _PROJECT_ROOT

REQUIRED_COLS = {"instruction", "input", "output", "topic", "upvotes", "question_id"}

SKILL_DEFINITIONS = {
    "crisis-intervention": {
        "source": "datasets/crisis_detection_processed",
        "filter_field": "topic",
        "filter_value": "crisis_support",
    },
    "general-support": {
        "source": "datasets/crisis_detection_processed",
        "filter_field": "topic",
        "filter_value": "general_support",
    },
    "cbt-therapy": {
        "source": "datasets/cactus_processed",
    },
    "empathetic-listening": {
        "source": "datasets/esconv_processed",
    },
    "psychoeducation": {
        "source": "datasets/mentalchat16k_processed",
    },
    "professional-counseling": {
        "sources": [
            "datasets/counsel_chat_processed",
            "datasets/amod_processed",
            "datasets/kaggle_mental_health_nguyen_processed_combined",
        ],
    },
}


def _resolve_path(rel_path: str) -> str:
    """Resolve a relative path against PROJECT_ROOT."""
    if os.path.isabs(rel_path):
        return rel_path
    return str(PROJECT_ROOT / rel_path)


def _normalize_dataset(ds: DatasetDict) -> DatasetDict:
    """Normalize question_id to string and drop extra columns."""
    # Normalize question_id to string
    if "train" in ds and "question_id" in ds["train"].features:
        def _to_str(ex):
            ex["question_id"] = str(ex.get("question_id", ""))
            return ex

        ds = ds.map(_to_str, desc="Normalize question_id")
        for split in ds:
            feats = ds[split].features.copy()
            feats["question_id"] = Value("string")
            ds[split] = ds[split].cast(feats)

    # Drop extra columns
    for split in ds:
        extra = set(ds[split].column_names) - REQUIRED_COLS
        if extra:
            ds[split] = ds[split].remove_columns(list(extra))

    return ds


def _load_and_filter(source_path: str, filter_field: str = None, filter_value: str = None) -> DatasetDict:
    """Load a dataset from disk and optionally filter by a field value."""
    abs_path = _resolve_path(source_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Dataset not found: {abs_path}")

    ds = load_from_disk(abs_path)
    if "train" not in ds:
        raise ValueError(f"No train split in {abs_path}")

    if filter_field and filter_value:
        filtered = {}
        for split in ds:
            filtered[split] = ds[split].filter(
                lambda ex: ex[filter_field] == filter_value,
                desc=f"Filter {split} where {filter_field}={filter_value}",
            )
        ds = DatasetDict(filtered)

    return _normalize_dataset(ds)


def _combine_datasets(source_paths: list) -> DatasetDict:
    """Combine multiple datasets following the combine_all() pattern."""
    loaded = []
    names = []

    for source_path in source_paths:
        abs_path = _resolve_path(source_path)
        if not os.path.exists(abs_path):
            print(f"    ! Not found, skipping: {abs_path}")
            continue
        try:
            ds = load_from_disk(abs_path)
            if "train" not in ds:
                print(f"    ! No train split, skipping: {abs_path}")
                continue

            ds = _normalize_dataset(ds)

            t = len(ds["train"])
            v = len(ds.get("validation", []))
            loaded.append(ds)
            name = os.path.basename(abs_path)
            names.append(name)
            print(f"      + {name}: train={t:,}, val={v:,}")
        except Exception as e:
            print(f"      x Error loading {abs_path}: {e}")

    if not loaded:
        raise ValueError(f"No datasets loaded from: {source_paths}")

    combined_train = concatenate_datasets([ds["train"] for ds in loaded])
    val_splits = [
        ds["validation"]
        for ds in loaded
        if "validation" in ds and len(ds["validation"]) > 0
    ]
    combined_val = (
        concatenate_datasets(val_splits)
        if val_splits
        else Dataset.from_dict({c: [] for c in REQUIRED_COLS})
    )

    return DatasetDict({"train": combined_train, "validation": combined_val})


def prepare_skill(skill_name: str, skill_def: dict, output_dir: str) -> bool:
    """Prepare a single skill dataset."""
    output_path = os.path.join(output_dir, skill_name)

    print(f"\n  [{skill_name}]")

    try:
        if "sources" in skill_def:
            # Combine multiple datasets
            print(f"    Combining {len(skill_def['sources'])} datasets...")
            ds = _combine_datasets(skill_def["sources"])
        else:
            # Single dataset (possibly filtered)
            source = skill_def["source"]
            filter_field = skill_def.get("filter_field")
            filter_value = skill_def.get("filter_value")
            if filter_field:
                print(f"    Filtering {source} where {filter_field}={filter_value}...")
            else:
                print(f"    Copying {source}...")
            ds = _load_and_filter(source, filter_field, filter_value)
    except (FileNotFoundError, ValueError) as e:
        print(f"    WARNING: {e} — skipping {skill_name}")
        return False

    # Save
    Path(output_path).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(output_path)

    train_count = len(ds["train"])
    val_count = len(ds.get("validation", []))
    print(f"    Saved: train={train_count:,}, val={val_count:,} -> {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare skill-specific datasets for LoRA training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/skills",
        help="Output directory for skill datasets (default: datasets/skills)",
    )
    parser.add_argument(
        "--skills",
        type=str,
        nargs="+",
        default=None,
        help="Specific skills to prepare (default: all)",
    )
    args = parser.parse_args()

    # Resolve output dir
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = str(PROJECT_ROOT / output_dir)

    print("=" * 60)
    print("  Preparing Skill-Specific Datasets")
    print("=" * 60)
    print(f"  Output: {output_dir}")

    skills_to_prepare = args.skills or list(SKILL_DEFINITIONS.keys())
    success_count = 0
    fail_count = 0

    for skill_name in skills_to_prepare:
        if skill_name not in SKILL_DEFINITIONS:
            print(f"\n  WARNING: Unknown skill '{skill_name}', skipping.")
            fail_count += 1
            continue

        ok = prepare_skill(skill_name, SKILL_DEFINITIONS[skill_name], output_dir)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"  Done: {success_count} prepared, {fail_count} skipped/failed")
    print("=" * 60)

    if fail_count > 0:
        print("\n  Some datasets were missing. Ensure all source datasets exist under datasets/.")
        print("  Run: python scripts/download_and_process_datasets.py")


if __name__ == "__main__":
    main()
