#!/usr/bin/env python3
"""
Build a 200-sample stratified test set for LLM-as-a-Judge evaluation.

Samples 40 items from each of 5 source datasets, extracts the user message
from the composite instruction field, and saves to evaluation/test_set.json.

Usage:
    python scripts/evaluation/build_test_set.py --datasets-dir datasets --output evaluation/test_set.json
"""

import argparse
import json
import re
import random
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_from_disk


# Dataset configs: (directory name, category label, risk hint, samples to draw)
DATASET_CONFIGS = [
    ("crisis_detection_processed", "crisis", "high", 40),
    ("cactus_processed", "cbt", "low", 40),
    ("esconv_processed", "empathy", "medium", 40),
    ("counsel_chat_processed", "general_counseling", "low", 40),
    ("mentalchat16k_processed", "psychoeducation", "low", 40),
]

SEED = 42

# Minimum lengths for quality filtering
MIN_INSTRUCTION_LEN = 50
MIN_OUTPUT_LEN = 100


def extract_user_message(instruction: str) -> tuple[str, str, str]:
    """Extract user message, conversation context, and topic from the instruction field.

    Different datasets embed the user question in different patterns:
    - Most: "...Question: {user_msg}\\n\\nResponse:" or "...Question: {user_msg}\\n\\nPlease provide..."
    - Crisis: "...Message: {user_msg}" (no trailing marker)
    - Cactus: May include "CBT Plan: {plan}" block before question

    Returns:
        (user_message, conversation_context, topic)
    """
    context = ""
    topic = ""

    # Try to extract Context: field
    ctx_match = re.search(r"Context:\s*(.+?)(?=\n\n|\nQuestion:|\nCBT Plan:)", instruction, re.DOTALL)
    if ctx_match:
        context = ctx_match.group(1).strip()
        topic = context  # use context as topic hint

    # Try "Question: ..." pattern (most datasets)
    # Match up to Response:, or "Please provide", or end of string
    q_match = re.search(
        r"Question:\s*(.+?)(?=\n\nResponse:|\n\nPlease provide|\Z)",
        instruction,
        re.DOTALL,
    )
    if q_match:
        return q_match.group(1).strip(), context, topic

    # Try "Message: ..." pattern (crisis dataset)
    m_match = re.search(r"Message:\s*(.+)", instruction, re.DOTALL)
    if m_match:
        return m_match.group(1).strip(), context, topic

    # Fallback: return the full instruction
    return instruction.strip(), context, topic


def load_and_sample(dataset_dir: Path, category: str, risk_hint: str, n: int, rng: random.Random) -> list[dict]:
    """Load a dataset, filter, and sample n items."""
    if not dataset_dir.exists():
        print(f"  WARNING: {dataset_dir} not found, skipping")
        return []

    ds = load_from_disk(str(dataset_dir))

    # Use train split (all processed datasets have it)
    if "train" in ds:
        split = ds["train"]
    else:
        split = ds[list(ds.keys())[0]]

    print(f"  {dataset_dir.name}: {len(split)} total samples")

    # Filter by minimum length
    candidates = []
    for row in split:
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        if len(instruction) >= MIN_INSTRUCTION_LEN and len(output) >= MIN_OUTPUT_LEN:
            candidates.append(row)

    print(f"    After filtering: {len(candidates)} candidates (need {n})")

    if len(candidates) < n:
        print(f"    WARNING: Only {len(candidates)} candidates available, using all")
        sampled = candidates
    else:
        sampled = rng.sample(candidates, n)

    samples = []
    for row in sampled:
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        user_message, context, topic_hint = extract_user_message(instruction)

        # Prefer dataset-provided topic if available
        ds_topic = row.get("topic", "")
        if ds_topic:
            topic_hint = ds_topic

        samples.append({
            "source_dataset": dataset_dir.name,
            "category": category,
            "risk_hint": risk_hint,
            "instruction": instruction,
            "user_message": user_message,
            "conversation_context": context,
            "ground_truth": output,
            "topic": topic_hint,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Build stratified test set for LLM judge evaluation")
    parser.add_argument("--datasets-dir", type=str, default="datasets", help="Root directory containing processed datasets")
    parser.add_argument("--output", type=str, default="evaluation/test_set.json", help="Output JSON path")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    output_path = Path(args.output)
    rng = random.Random(args.seed)

    print("=" * 60)
    print("Building LLM Judge Test Set")
    print("=" * 60)
    print(f"  Datasets dir: {datasets_dir}")
    print(f"  Output:       {output_path}")
    print(f"  Seed:         {args.seed}")
    print()

    all_samples = []
    for ds_name, category, risk_hint, n in DATASET_CONFIGS:
        ds_path = datasets_dir / ds_name
        samples = load_and_sample(ds_path, category, risk_hint, n, rng)
        all_samples.extend(samples)

    # Assign sequential IDs
    for i, sample in enumerate(all_samples):
        sample["sample_id"] = f"test_{i:03d}"

    # Build output
    result = {
        "metadata": {
            "seed": args.seed,
            "total_samples": len(all_samples),
            "samples_per_dataset": {cfg[0]: cfg[3] for cfg in DATASET_CONFIGS},
            "min_instruction_len": MIN_INSTRUCTION_LEN,
            "min_output_len": MIN_OUTPUT_LEN,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "samples": all_samples,
    }

    # Verify category counts
    from collections import Counter
    cat_counts = Counter(s["category"] for s in all_samples)
    print(f"\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")
    print(f"  Total: {len(all_samples)}")

    # Check for duplicate user messages
    messages = [s["user_message"] for s in all_samples]
    n_unique = len(set(messages))
    if n_unique < len(messages):
        print(f"\n  WARNING: {len(messages) - n_unique} duplicate user messages found")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
