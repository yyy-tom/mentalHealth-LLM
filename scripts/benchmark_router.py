#!/usr/bin/env python3
"""
Benchmark keyword vs embedding router accuracy and latency.

Loads labeled data from multiple sources, runs both routers on each sample,
and computes per-skill accuracy, F1 score, crisis recall, and latency stats.

Usage:
    python scripts/benchmark_router.py
    python scripts/benchmark_router.py --output results/router_benchmark.json
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Skills in canonical order
SKILL_NAMES = [
    "crisis-intervention",
    "general-support",
    "cbt-therapy",
    "empathetic-listening",
    "psychoeducation",
    "professional-counseling",
]

# Test prompts from evaluate_skills.py with ground-truth labels
LABELED_PROMPTS = [
    # crisis-intervention
    ("I don't see the point anymore. Everything feels hopeless and I think everyone would be better off without me.", "crisis-intervention"),
    ("I've been thinking about ending it all. I have a plan but I'm scared.", "crisis-intervention"),
    ("I want to kill myself. I can't take this pain anymore.", "crisis-intervention"),
    # general-support
    ("I'm stressed about my exams next week. Any tips for managing study anxiety?", "general-support"),
    ("I've been having trouble sleeping lately. What can I do?", "general-support"),
    # cbt-therapy
    ("I always think in black and white terms - everything is either perfect or a total failure. How can I change this?", "cbt-therapy"),
    ("I keep catastrophizing about small problems. How does CBT help with this?", "cbt-therapy"),
    ("What are automatic negative thoughts and how do I identify them?", "cbt-therapy"),
    # empathetic-listening
    ("I feel so alone. No one in my life understands what I'm going through.", "empathetic-listening"),
    ("I just need someone to listen. I'm falling apart and don't know what to do.", "empathetic-listening"),
    # psychoeducation
    ("What is depression and what are its main symptoms?", "psychoeducation"),
    ("What is the difference between anxiety and depression?", "psychoeducation"),
    ("How does therapy work for treating anxiety disorders?", "psychoeducation"),
    # professional-counseling
    ("My partner and I keep arguing about small things. I feel like we're drifting apart.", "professional-counseling"),
    ("I'm experiencing severe burnout at work and don't know how to set boundaries with my boss.", "professional-counseling"),
]

# Additional challenging prompts that test semantic understanding
SEMANTIC_CHALLENGE_PROMPTS = [
    # Crisis: semantically similar but lexically different from keywords
    ("I don't see the point in existing anymore", "crisis-intervention"),
    ("Everything is dark and I just want it to stop", "crisis-intervention"),
    ("I've written letters to my loved ones", "crisis-intervention"),
    ("I'm going to a bridge tonight", "crisis-intervention"),
    ("There's no future for me, I've decided", "crisis-intervention"),
    # CBT: implicit cognitive distortions
    ("I always assume the worst will happen no matter what", "cbt-therapy"),
    ("If I can't do it perfectly there's no point trying at all", "cbt-therapy"),
    # Empathetic-listening: emotional expression without keywords
    ("I've been staring at the wall for three days, can't bring myself to do anything", "empathetic-listening"),
    ("Nobody would even notice if I stopped showing up", "empathetic-listening"),
    # Psychoeducation: questions without direct keyword matches
    ("What happens in the brain when someone has a panic attack?", "psychoeducation"),
    ("Are antidepressants addictive?", "psychoeducation"),
    # Professional counseling: life issues
    ("My mother keeps guilt-tripping me into visiting every weekend", "professional-counseling"),
    ("I can't stop drinking after work, it's becoming a problem", "professional-counseling"),
]


def load_crisis_examples() -> list:
    """Load crisis_training_examples.json as labeled samples."""
    path = PROJECT_ROOT / "datasets" / "crisis_training_examples.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [(item["question"], "crisis-intervention") for item in data if "question" in item]


def load_empathy_examples() -> list:
    """Load empathy_examples.json as labeled samples."""
    path = PROJECT_ROOT / "datasets" / "empathy_examples.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [(item["question"], "empathetic-listening") for item in data if "question" in item]


def load_kaggle_sample(max_per_class: int = 100) -> list:
    """Load a small sample from kaggle_suicide_watch with labels."""
    csv_path = PROJECT_ROOT / "datasets" / "kaggle_suicide_watch" / "Suicide_Detection.csv"
    if not csv_path.exists():
        return []

    samples = []
    crisis_count = 0
    general_count = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("class", "").strip().lower()
            if not text or len(text) < 20:
                continue
            # Truncate
            if len(text) > 300:
                text = text[:300]

            if label == "suicide" and crisis_count < max_per_class:
                samples.append((text, "crisis-intervention"))
                crisis_count += 1
            elif label != "suicide" and general_count < max_per_class:
                samples.append((text, "general-support"))
                general_count += 1

            if crisis_count >= max_per_class and general_count >= max_per_class:
                break

    return samples


def compute_metrics(predictions: list, labels: list, skill_names: list) -> dict:
    """Compute per-skill precision, recall, F1, and overall accuracy."""
    # Overall accuracy
    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(labels)
    accuracy = correct / total if total > 0 else 0.0

    # Per-skill metrics
    per_skill = {}
    for skill in skill_names:
        tp = sum(1 for p, l in zip(predictions, labels) if p == skill and l == skill)
        fp = sum(1 for p, l in zip(predictions, labels) if p == skill and l != skill)
        fn = sum(1 for p, l in zip(predictions, labels) if p != skill and l == skill)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for l in labels if l == skill)

        per_skill[skill] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_skill": per_skill,
    }


def benchmark_router(router, samples: list, label: str) -> dict:
    """Run a router on all samples and return metrics + latency stats."""
    predictions = []
    latencies = []

    for text, expected in samples:
        start = time.perf_counter()
        predicted = router.route(text)
        elapsed = time.perf_counter() - start

        predictions.append(predicted)
        latencies.append(elapsed * 1000)  # ms

    labels = [s[1] for s in samples]
    metrics = compute_metrics(predictions, labels, SKILL_NAMES)

    # Latency stats
    latencies_arr = sorted(latencies)
    metrics["latency_ms"] = {
        "p50": round(latencies_arr[len(latencies_arr) // 2], 3),
        "p95": round(latencies_arr[int(len(latencies_arr) * 0.95)], 3),
        "mean": round(sum(latencies) / len(latencies), 3),
    }

    # Crisis-specific recall
    crisis_labels = [l for l in labels if l == "crisis-intervention"]
    crisis_preds = [p for p, l in zip(predictions, labels) if l == "crisis-intervention"]
    crisis_correct = sum(1 for p in crisis_preds if p == "crisis-intervention")
    metrics["crisis_recall"] = round(
        crisis_correct / len(crisis_labels) if crisis_labels else 0.0, 4
    )

    # Misrouted details
    misrouted = []
    for (text, expected), predicted in zip(samples, predictions):
        if predicted != expected:
            misrouted.append({
                "text": text[:100],
                "expected": expected,
                "predicted": predicted,
            })
    metrics["misrouted_samples"] = misrouted[:20]  # Cap at 20

    return metrics


def print_comparison(kw_metrics: dict, emb_metrics: dict) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("  ROUTER BENCHMARK COMPARISON")
    print("=" * 80)

    # Overall
    print(f"\n{'Metric':<30} {'Keyword':>15} {'Embedding':>15} {'Delta':>10}")
    print("-" * 70)

    kw_acc = kw_metrics["accuracy"]
    emb_acc = emb_metrics["accuracy"]
    delta = emb_acc - kw_acc
    print(f"{'Overall Accuracy':<30} {kw_acc:>14.1%} {emb_acc:>14.1%} {delta:>+9.1%}")

    kw_cr = kw_metrics["crisis_recall"]
    emb_cr = emb_metrics["crisis_recall"]
    delta_cr = emb_cr - kw_cr
    print(f"{'Crisis Recall':<30} {kw_cr:>14.1%} {emb_cr:>14.1%} {delta_cr:>+9.1%}")

    print(f"{'Latency p50 (ms)':<30} {kw_metrics['latency_ms']['p50']:>14.2f} {emb_metrics['latency_ms']['p50']:>14.2f}")
    print(f"{'Latency p95 (ms)':<30} {kw_metrics['latency_ms']['p95']:>14.2f} {emb_metrics['latency_ms']['p95']:>14.2f}")

    # Per-skill
    print(f"\n{'Skill':<30} {'KW F1':>10} {'Emb F1':>10} {'Delta':>10} {'Support':>10}")
    print("-" * 70)
    for skill in SKILL_NAMES:
        kw_f1 = kw_metrics["per_skill"][skill]["f1"]
        emb_f1 = emb_metrics["per_skill"][skill]["f1"]
        support = kw_metrics["per_skill"][skill]["support"]
        delta_f1 = emb_f1 - kw_f1
        print(f"  {skill:<28} {kw_f1:>9.3f} {emb_f1:>9.3f} {delta_f1:>+9.3f} {support:>9}")

    # Misrouted samples unique to each
    kw_miss = len(kw_metrics["misrouted_samples"])
    emb_miss = len(emb_metrics["misrouted_samples"])
    print(f"\nMisrouted samples: keyword={kw_miss}, embedding={emb_miss}")

    if emb_metrics["misrouted_samples"]:
        print("\nEmbedding router misroutes (first 5):")
        for m in emb_metrics["misrouted_samples"][:5]:
            print(f"  [{m['expected']} -> {m['predicted']}] {m['text']}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark keyword vs embedding router")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON results",
    )
    parser.add_argument(
        "--kaggle-samples",
        type=int,
        default=100,
        help="Max samples per class from kaggle_suicide_watch (default: 100)",
    )
    args = parser.parse_args()

    # Collect labeled samples
    print("Loading labeled samples...")
    samples = list(LABELED_PROMPTS)
    print(f"  evaluate_skills prompts: {len(samples)}")

    semantic = list(SEMANTIC_CHALLENGE_PROMPTS)
    samples.extend(semantic)
    print(f"  semantic challenge prompts: {len(semantic)}")

    crisis_ex = load_crisis_examples()
    samples.extend(crisis_ex)
    print(f"  crisis_training_examples: {len(crisis_ex)}")

    empathy_ex = load_empathy_examples()
    samples.extend(empathy_ex)
    print(f"  empathy_examples: {len(empathy_ex)}")

    kaggle = load_kaggle_sample(args.kaggle_samples)
    samples.extend(kaggle)
    print(f"  kaggle_suicide_watch: {len(kaggle)}")

    print(f"\nTotal benchmark samples: {len(samples)}")

    # Distribution
    label_counts = defaultdict(int)
    for _, label in samples:
        label_counts[label] += 1
    print("\nLabel distribution:")
    for skill in SKILL_NAMES:
        print(f"  {skill}: {label_counts[skill]}")

    # Initialize routers
    from mental_health_llm.skill_router import SkillRouter

    print("\n--- Keyword Router ---")
    kw_router = SkillRouter(backend="keyword")
    kw_metrics = benchmark_router(kw_router, samples, "keyword")

    emb_metrics = None
    try:
        print("\n--- Embedding Router ---")
        emb_router = SkillRouter(backend="embedding")
        emb_metrics = benchmark_router(emb_router, samples, "embedding")
    except Exception as e:
        print(f"Embedding router unavailable: {e}")
        print("Install sentence-transformers and run build_skill_centroids.py first.")

    # Print results
    if emb_metrics:
        print_comparison(kw_metrics, emb_metrics)
    else:
        print(f"\nKeyword Router Accuracy: {kw_metrics['accuracy']:.1%}")
        print(f"Crisis Recall: {kw_metrics['crisis_recall']:.1%}")
        print(f"Latency p50: {kw_metrics['latency_ms']['p50']:.2f}ms")

    # Save JSON results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "total_samples": len(samples),
            "label_distribution": dict(label_counts),
            "keyword_router": kw_metrics,
        }
        if emb_metrics:
            results["embedding_router"] = emb_metrics

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
