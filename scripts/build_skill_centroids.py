#!/usr/bin/env python3
"""
Build skill centroid embeddings for the embedding router.

Computes per-skill centroid vectors by averaging sentence-transformer embeddings
over labeled training data. For skills with few labeled samples, augments with
template sentences generated from skills_config.json keywords.

Data sources per skill:
  crisis-intervention    <- kaggle_suicide_watch (suicide class) + crisis_training_examples.json
  general-support        <- kaggle_suicide_watch (non-suicide class)
  cbt-therapy            <- cactus_processed + evaluate_skills test prompts + keyword templates
  empathetic-listening   <- esconv_processed + empathy_examples.json + keyword templates
  psychoeducation        <- mentalchat16k_processed + keyword templates
  professional-counseling <- counsel_chat_processed + keyword templates

Output:
  mental_health_llm/centroids/skill_centroids.npz  (~10KB)
  mental_health_llm/centroids/centroid_metadata.json

Usage:
    python scripts/build_skill_centroids.py
    python scripts/build_skill_centroids.py --model sentence-transformers/all-MiniLM-L6-v2
    python scripts/build_skill_centroids.py --max-samples 5000
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Skill names in canonical order
SKILL_NAMES = [
    "crisis-intervention",
    "general-support",
    "cbt-therapy",
    "empathetic-listening",
    "psychoeducation",
    "professional-counseling",
]

# Keyword template patterns for augmentation
KEYWORD_TEMPLATES = [
    "I need help with {keyword}",
    "Tell me about {keyword}",
    "I'm struggling with {keyword}",
    "Can you help me with {keyword}?",
    "I've been dealing with {keyword}",
    "How do I handle {keyword}?",
]

# Test prompts from evaluate_skills.py (hardcoded here to avoid import complexity)
EVALUATE_PROMPTS = {
    "crisis-intervention": [
        "I don't see the point anymore. Everything feels hopeless and I think everyone would be better off without me.",
        "I've been thinking about ending it all. I have a plan but I'm scared.",
        "I want to kill myself. I can't take this pain anymore.",
    ],
    "general-support": [
        "I'm stressed about my exams next week. Any tips for managing study anxiety?",
        "I've been having trouble sleeping lately. What can I do?",
    ],
    "cbt-therapy": [
        "I always think in black and white terms - everything is either perfect or a total failure. How can I change this?",
        "I keep catastrophizing about small problems. How does CBT help with this?",
        "What are automatic negative thoughts and how do I identify them?",
    ],
    "empathetic-listening": [
        "I feel so alone. No one in my life understands what I'm going through.",
        "I just need someone to listen. I'm falling apart and don't know what to do.",
    ],
    "psychoeducation": [
        "What is depression and what are its main symptoms?",
        "What is the difference between anxiety and depression?",
        "How does therapy work for treating anxiety disorders?",
    ],
    "professional-counseling": [
        "My partner and I keep arguing about small things. I feel like we're drifting apart.",
        "I'm experiencing severe burnout at work and don't know how to set boundaries with my boss.",
    ],
}


def load_skills_config() -> dict:
    """Load skills_config.json and return skill keyword map."""
    config_path = PROJECT_ROOT / "mental_health_llm" / "skills_config.json"
    with open(config_path) as f:
        config = json.load(f)
    skill_keywords = {}
    for skill_def in config["skills"]:
        skill_keywords[skill_def["name"]] = skill_def.get("keywords", [])
    return skill_keywords


def generate_keyword_sentences(keywords: list) -> list:
    """Generate template sentences from keywords."""
    sentences = []
    for kw in keywords:
        for template in KEYWORD_TEMPLATES:
            sentences.append(template.format(keyword=kw))
    return sentences


def load_kaggle_suicide_watch(max_samples: int = 50000) -> tuple:
    """Load kaggle_suicide_watch CSV, return (crisis_texts, non_crisis_texts)."""
    csv_path = DATASETS_DIR / "kaggle_suicide_watch" / "Suicide_Detection.csv"
    if not csv_path.exists():
        print(f"  kaggle_suicide_watch not found at {csv_path}")
        return [], []

    crisis_texts = []
    non_crisis_texts = []

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("class", "").strip().lower()
            if not text or len(text) < 10:
                continue

            # Truncate very long posts to first 512 chars for embedding
            if len(text) > 512:
                text = text[:512]

            if label == "suicide":
                if len(crisis_texts) < max_samples:
                    crisis_texts.append(text)
            else:
                if len(non_crisis_texts) < max_samples:
                    non_crisis_texts.append(text)

            if len(crisis_texts) >= max_samples and len(non_crisis_texts) >= max_samples:
                break

    return crisis_texts, non_crisis_texts


def load_json_examples(path: Path, field: str = "question") -> list:
    """Load examples from a JSON file, extracting the specified field."""
    if not path.exists():
        print(f"    NOT FOUND: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    items = [item[field] for item in data if field in item]
    print(f"    {path.name}: {len(items)} examples")
    return items


def _extract_question_from_instruction(instruction: str) -> str:
    """Extract the user question from a formatted instruction field.

    Handles the pattern:
        ... Question: <user question>\n\nPlease provide ...
    """
    marker = "Question: "
    idx = instruction.find(marker)
    if idx == -1:
        return ""
    text = instruction[idx + len(marker):]
    # Trim at the "Please provide" footer
    end = text.find("\n\nPlease provide")
    if end != -1:
        text = text[:end]
    return text.strip()


def load_hf_dataset_texts(dataset_name: str, field: str = "input", max_samples: int = 50000) -> list:
    """Load text samples from a HuggingFace dataset on disk.

    Args:
        dataset_name: Directory name under datasets/ (e.g. "cactus_processed").
        field: Column name to extract text from. If the field is empty,
               falls back to extracting from "instruction" (Question: ... pattern).
        max_samples: Maximum number of samples to load.
    """
    abs_path = DATASETS_DIR / dataset_name
    if not abs_path.exists():
        print(f"    NOT FOUND: {abs_path}")
        return []

    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(abs_path))

        texts = []
        fallback_count = 0
        split = ds.get("train", ds.get("validation"))
        if split is None:
            print(f"    No train/validation split in {abs_path}")
            return []

        for i, row in enumerate(split):
            if i >= max_samples:
                break
            text = row.get(field, "").strip()

            # Fallback: extract question from instruction field
            if not text and "instruction" in row:
                text = _extract_question_from_instruction(row["instruction"])
                if text:
                    fallback_count += 1

            if text and len(text) >= 10:
                if len(text) > 512:
                    text = text[:512]
                texts.append(text)

        if fallback_count > 0:
            print(f"    ({fallback_count} extracted from instruction field)")

        return texts
    except Exception as e:
        print(f"    Error loading {abs_path}: {e}")
        return []


def collect_skill_texts(skill_keywords: dict, max_samples: int) -> dict:
    """Collect text samples for each skill from all available data sources."""
    skill_texts = {name: [] for name in SKILL_NAMES}

    print(f"\nCollecting training texts per skill...")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  DATASETS_DIR: {DATASETS_DIR}")

    # --- crisis-intervention ---
    print("\n  [crisis-intervention]")
    crisis_kaggle, non_crisis_kaggle = load_kaggle_suicide_watch(max_samples)
    print(f"    kaggle_suicide_watch: {len(crisis_kaggle)} crisis, {len(non_crisis_kaggle)} non-crisis")

    skill_texts["crisis-intervention"].extend(crisis_kaggle)

    crisis_examples = load_json_examples(
        DATASETS_DIR / "crisis_training_examples.json"
    )
    skill_texts["crisis-intervention"].extend(crisis_examples)

    # --- general-support ---
    print("\n  [general-support]")
    skill_texts["general-support"].extend(non_crisis_kaggle)

    # --- cbt-therapy ---
    print("\n  [cbt-therapy]")
    cbt_texts = load_hf_dataset_texts("cactus_processed", "input", max_samples)
    print(f"    cactus_processed: {len(cbt_texts)} texts")
    skill_texts["cbt-therapy"].extend(cbt_texts)

    # --- empathetic-listening ---
    print("\n  [empathetic-listening]")
    esconv_texts = load_hf_dataset_texts("esconv_processed", "input", max_samples)
    print(f"    esconv_processed: {len(esconv_texts)} texts")
    skill_texts["empathetic-listening"].extend(esconv_texts)

    empathy_examples = load_json_examples(
        DATASETS_DIR / "empathy_examples.json"
    )
    skill_texts["empathetic-listening"].extend(empathy_examples)

    # --- psychoeducation ---
    print("\n  [psychoeducation]")
    mentalchat_texts = load_hf_dataset_texts("mentalchat16k_processed", "input", max_samples)
    print(f"    mentalchat16k_processed: {len(mentalchat_texts)} texts")
    skill_texts["psychoeducation"].extend(mentalchat_texts)

    # --- professional-counseling ---
    print("\n  [professional-counseling]")
    counsel_texts = load_hf_dataset_texts("counsel_chat_processed", "input", max_samples)
    print(f"    counsel_chat_processed: {len(counsel_texts)} texts")
    skill_texts["professional-counseling"].extend(counsel_texts)

    # --- Augment all skills with evaluate_skills test prompts + keyword templates ---
    print("\n  [augmentation]")
    for skill_name in SKILL_NAMES:
        # Test prompts from evaluate_skills.py
        eval_prompts = EVALUATE_PROMPTS.get(skill_name, [])
        skill_texts[skill_name].extend(eval_prompts)

        # Keyword-derived template sentences
        keywords = skill_keywords.get(skill_name, [])
        if keywords:
            kw_sentences = generate_keyword_sentences(keywords)
            skill_texts[skill_name].extend(kw_sentences)

        count = len(skill_texts[skill_name])
        print(f"    {skill_name}: {count} total texts")

    return skill_texts


def build_centroids(
    skill_texts: dict,
    model_name: str,
    batch_size: int = 256,
) -> tuple:
    """Encode all texts and compute per-skill centroid vectors.

    Returns:
        (centroids, metadata) where centroids is (n_skills, embed_dim) ndarray
        and metadata is a dict with per-skill statistics.
    """
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading sentence-transformer: {model_name}")
    model = SentenceTransformer(model_name)
    embed_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {embed_dim}")

    centroids = np.zeros((len(SKILL_NAMES), embed_dim), dtype=np.float32)
    metadata = {"model_name": model_name, "embed_dim": embed_dim, "skills": {}}

    for i, skill_name in enumerate(SKILL_NAMES):
        texts = skill_texts[skill_name]
        if not texts:
            print(f"\n  WARNING: No texts for {skill_name}, using zero vector")
            metadata["skills"][skill_name] = {"n_samples": 0}
            continue

        print(f"\n  Encoding {skill_name}: {len(texts)} texts...")
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 1000,
        )

        # Compute centroid (mean of unit-norm vectors, then re-normalize)
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[i] = centroid

        metadata["skills"][skill_name] = {
            "n_samples": len(texts),
            "centroid_norm_before_renorm": float(norm),
        }
        print(f"    Centroid computed (pre-norm magnitude: {norm:.4f})")

    return centroids, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Build skill centroid embeddings for the embedding router"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model name or path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Max samples per data source (default: 50000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size (default: 256)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: mental_health_llm/centroids/)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "mental_health_llm" / "centroids")
    os.makedirs(output_dir, exist_ok=True)

    # Collect texts
    skill_keywords = load_skills_config()
    skill_texts = collect_skill_texts(skill_keywords, args.max_samples)

    # Build centroids
    centroids, metadata = build_centroids(
        skill_texts, args.model, args.batch_size
    )

    # Save
    npz_path = os.path.join(output_dir, "skill_centroids.npz")
    np.savez(
        npz_path,
        centroids=centroids,
        skill_names=np.array(SKILL_NAMES),
    )
    print(f"\nSaved centroids: {npz_path} ({os.path.getsize(npz_path):,} bytes)")

    meta_path = os.path.join(output_dir, "centroid_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  Centroid Build Summary")
    print("=" * 60)
    for skill_name in SKILL_NAMES:
        info = metadata["skills"].get(skill_name, {})
        n = info.get("n_samples", 0)
        print(f"  {skill_name:30s}  {n:>8,} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
