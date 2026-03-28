#!/usr/bin/env python3
"""
Aggregate LLM judge scores and generate markdown tables for the paper.

Computes per-dimension averages, overall scores, Krippendorff's alpha,
risk-level breakdowns, score distributions, and optional human-LLM correlation.

Usage:
    python scripts/evaluation/aggregate_results.py \
        --scores-dir evaluation/scores \
        --output-dir evaluation/results

    # With human scores for correlation
    python scripts/evaluation/aggregate_results.py \
        --scores-dir evaluation/scores \
        --output-dir evaluation/results \
        --human-scores evaluation/human_scores.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DIMENSIONS = ["empathy", "cbt", "guided_discovery", "safety", "clinical_appropriateness"]
DIMENSION_LABELS = {
    "empathy": "Empathy",
    "cbt": "CBT",
    "guided_discovery": "Guided Disc.",
    "safety": "Safety",
    "clinical_appropriateness": "Clinical",
}

MODEL_DISPLAY_NAMES = {
    "qwen2.5-7b": "Qwen 2.5 7B",
    "gemma2-9b": "Gemma 2 9B",
    "mistral-7b": "Mistral 7B",
    "llama-3.1-8b": "Llama 3.1 8B",
}


def round_to_half(value: float) -> float:
    """Round to nearest 0.5."""
    return round(value * 2) / 2


def collect_scores(scores_dir: Path) -> dict:
    """Load all score files and organize by model -> run -> sample.

    Returns:
        {model_key: {run_id: {sample_id: {dim: score, ...}}}}
    """
    data = defaultdict(lambda: defaultdict(dict))

    for path in sorted(scores_dir.glob("*_run*.json")):
        match = re.match(r"(.+)_run(\d+)\.json", path.name)
        if not match:
            continue
        model_key = match.group(1)
        run_id = int(match.group(2))

        with open(path) as f:
            content = json.load(f)

        for entry in content["scores"]:
            sid = entry["sample_id"]
            scores = {"risk_level": entry.get("risk_level", "unknown")}
            for dim in DIMENSIONS:
                dim_data = entry.get(dim, {})
                score = dim_data.get("score", "N/A") if isinstance(dim_data, dict) else "N/A"
                scores[dim] = score
            data[model_key][run_id][sid] = scores

    return dict(data)


def compute_dimension_averages(model_data: dict) -> dict:
    """Compute per-dimension averages across all runs and samples.

    Returns {dim: rounded_average} with CBT N/A excluded.
    """
    dim_values = defaultdict(list)

    for run_id, samples in model_data.items():
        for sid, scores in samples.items():
            for dim in DIMENSIONS:
                val = scores.get(dim, "N/A")
                if isinstance(val, (int, float)):
                    dim_values[dim].append(val)

    averages = {}
    for dim in DIMENSIONS:
        values = dim_values[dim]
        if values:
            averages[dim] = round_to_half(np.mean(values))
        else:
            averages[dim] = "N/A"

    return averages


def compute_overall(averages: dict) -> float | str:
    """Mean of applicable dimension scores."""
    numeric = [v for v in averages.values() if isinstance(v, (int, float))]
    if numeric:
        return round_to_half(np.mean(numeric))
    return "N/A"


def compute_krippendorff_alpha(model_data: dict) -> dict:
    """Compute Krippendorff's alpha per dimension across runs.

    Each run is a 'rater'. For each dimension, builds a raters x samples matrix.
    """
    import krippendorff

    run_ids = sorted(model_data.keys())
    if len(run_ids) < 2:
        return {dim: "N/A (single run)" for dim in DIMENSIONS}

    # Get all sample IDs present in all runs
    all_sids = set()
    for rid in run_ids:
        all_sids.update(model_data[rid].keys())
    all_sids = sorted(all_sids)

    alphas = {}
    for dim in DIMENSIONS:
        # Build reliability matrix: raters x units
        matrix = []
        for rid in run_ids:
            row = []
            for sid in all_sids:
                val = model_data[rid].get(sid, {}).get(dim, "N/A")
                if isinstance(val, (int, float)):
                    row.append(val)
                else:
                    row.append(np.nan)
            matrix.append(row)

        reliability_data = np.array(matrix)

        # Check if enough non-NaN values exist
        valid_counts = np.sum(~np.isnan(reliability_data), axis=0)
        if np.sum(valid_counts >= 2) < 2:
            alphas[dim] = "N/A"
            continue

        try:
            alpha = krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement="ordinal",
            )
            alphas[dim] = round(alpha, 3)
        except Exception:
            alphas[dim] = "N/A"

    return alphas


def compute_safety_by_risk(model_data: dict) -> dict:
    """Compute average Safety score grouped by risk level."""
    risk_scores = defaultdict(list)

    for run_id, samples in model_data.items():
        for sid, scores in samples.items():
            risk = scores.get("risk_level", "unknown")
            safety = scores.get("safety", "N/A")
            if isinstance(safety, (int, float)):
                risk_scores[risk].append(safety)

    result = {}
    for risk in ["low", "medium", "high"]:
        values = risk_scores.get(risk, [])
        if values:
            result[risk] = {"mean": round_to_half(np.mean(values)), "count": len(values)}
        else:
            result[risk] = {"mean": "N/A", "count": 0}

    return result


def compute_score_distributions(model_data: dict) -> dict:
    """Histogram of 1-5 scores per dimension."""
    distributions = {}

    for dim in DIMENSIONS:
        counts = {i: 0 for i in range(1, 6)}
        na_count = 0
        for run_id, samples in model_data.items():
            for sid, scores in samples.items():
                val = scores.get(dim, "N/A")
                if isinstance(val, (int, float)) and 1 <= val <= 5:
                    counts[int(val)] += 1
                else:
                    na_count += 1
        distributions[dim] = {"counts": counts, "na": na_count}

    return distributions


def compute_spearman(scores_data: dict, human_scores_path: str) -> dict | None:
    """Compute Spearman correlation between LLM judge and human scores."""
    from scipy.stats import spearmanr

    with open(human_scores_path) as f:
        human_data = json.load(f)

    # Expect human_data to be a dict of {sample_id: {dim: score, ...}}
    if isinstance(human_data, list):
        human_map = {item["sample_id"]: item for item in human_data}
    else:
        human_map = human_data

    correlations = {}
    for dim in DIMENSIONS:
        llm_scores = []
        human_scores = []

        for model_key, model_data in scores_data.items():
            for run_id, samples in model_data.items():
                for sid, scores in samples.items():
                    if sid not in human_map:
                        continue
                    llm_val = scores.get(dim, "N/A")
                    human_val = human_map[sid].get(dim, {})
                    if isinstance(human_val, dict):
                        human_val = human_val.get("score", "N/A")

                    if isinstance(llm_val, (int, float)) and isinstance(human_val, (int, float)):
                        llm_scores.append(llm_val)
                        human_scores.append(human_val)

        if len(llm_scores) >= 5:
            rho, pval = spearmanr(llm_scores, human_scores)
            correlations[dim] = {"rho": round(rho, 3), "p_value": round(pval, 4), "n": len(llm_scores)}
        else:
            correlations[dim] = {"rho": "N/A", "n": len(llm_scores)}

    return correlations


def generate_markdown(
    all_averages: dict,
    all_overalls: dict,
    all_alphas: dict,
    all_safety_by_risk: dict,
    all_distributions: dict,
    correlations: dict | None,
) -> str:
    """Generate markdown tables for the paper."""
    lines = []

    # Main comparison table
    lines.append("## Model Comparison\n")
    header = "| Model | " + " | ".join(DIMENSION_LABELS[d] for d in DIMENSIONS) + " | Overall |"
    sep = "|" + "|".join(["---"] * (len(DIMENSIONS) + 2)) + "|"
    lines.append(header)
    lines.append(sep)

    for model_key in sorted(all_averages.keys()):
        avgs = all_averages[model_key]
        overall = all_overalls[model_key]
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        cells = [name]
        for dim in DIMENSIONS:
            val = avgs[dim]
            cells.append(f"{val}" if isinstance(val, str) else f"{val:.1f}")
        cells.append(f"{overall}" if isinstance(overall, str) else f"{overall:.1f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # Inter-run consistency (Krippendorff's alpha)
    lines.append("## Inter-Run Consistency (Krippendorff's Alpha)\n")
    header = "| Model | " + " | ".join(DIMENSION_LABELS[d] for d in DIMENSIONS) + " |"
    sep = "|" + "|".join(["---"] * (len(DIMENSIONS) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for model_key in sorted(all_alphas.keys()):
        alphas = all_alphas[model_key]
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        cells = [name]
        for dim in DIMENSIONS:
            val = alphas[dim]
            cells.append(f"{val}" if isinstance(val, str) else f"{val:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # Safety by risk level
    lines.append("## Safety Score by Risk Level\n")
    lines.append("| Model | Low | Medium | High |")
    lines.append("|---|---|---|---|")

    for model_key in sorted(all_safety_by_risk.keys()):
        sbr = all_safety_by_risk[model_key]
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        cells = [name]
        for risk in ["low", "medium", "high"]:
            val = sbr[risk]["mean"]
            n = sbr[risk]["count"]
            if isinstance(val, str):
                cells.append(f"{val}")
            else:
                cells.append(f"{val:.1f} (n={n})")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # Score distributions
    lines.append("## Score Distributions\n")
    for model_key in sorted(all_distributions.keys()):
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        lines.append(f"### {name}\n")
        lines.append("| Dimension | 1 | 2 | 3 | 4 | 5 | N/A |")
        lines.append("|---|---|---|---|---|---|---|")

        dists = all_distributions[model_key]
        for dim in DIMENSIONS:
            dist = dists[dim]
            label = DIMENSION_LABELS[dim]
            counts = dist["counts"]
            row = [label] + [str(counts[i]) for i in range(1, 6)] + [str(dist["na"])]
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    # Spearman correlation (if provided)
    if correlations:
        lines.append("## Human-LLM Correlation (Spearman's Rho)\n")
        lines.append("| Dimension | Rho | p-value | n |")
        lines.append("|---|---|---|---|")

        for dim in DIMENSIONS:
            c = correlations[dim]
            label = DIMENSION_LABELS[dim]
            rho = c.get("rho", "N/A")
            pval = c.get("p_value", "N/A")
            n = c.get("n", 0)
            if isinstance(rho, str):
                lines.append(f"| {label} | {rho} | - | {n} |")
            else:
                lines.append(f"| {label} | {rho:.3f} | {pval:.4f} | {n} |")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate LLM judge scores and generate tables")
    parser.add_argument("--scores-dir", type=str, default="evaluation/scores", help="Directory with score files")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Output directory")
    parser.add_argument("--human-scores", type=str, default=None, help="Optional human scores JSON for correlation")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Aggregating LLM Judge Results")
    print("=" * 60)

    # Collect all scores
    scores_data = collect_scores(scores_dir)
    if not scores_data:
        print(f"No score files found in {scores_dir}")
        return

    print(f"Found scores for models: {list(scores_data.keys())}")

    all_averages = {}
    all_overalls = {}
    all_alphas = {}
    all_safety_by_risk = {}
    all_distributions = {}

    for model_key, model_data in sorted(scores_data.items()):
        print(f"\nProcessing: {model_key}")
        runs = sorted(model_data.keys())
        total_scores = sum(len(samples) for samples in model_data.values())
        print(f"  Runs: {runs}, Total scored entries: {total_scores}")

        all_averages[model_key] = compute_dimension_averages(model_data)
        all_overalls[model_key] = compute_overall(all_averages[model_key])
        all_alphas[model_key] = compute_krippendorff_alpha(model_data)
        all_safety_by_risk[model_key] = compute_safety_by_risk(model_data)
        all_distributions[model_key] = compute_score_distributions(model_data)

        print(f"  Averages: {all_averages[model_key]}")
        print(f"  Overall:  {all_overalls[model_key]}")

    # Spearman correlation (optional)
    correlations = None
    if args.human_scores:
        print(f"\nComputing human-LLM correlation from {args.human_scores}")
        correlations = compute_spearman(scores_data, args.human_scores)

    # Generate markdown
    markdown = generate_markdown(
        all_averages, all_overalls, all_alphas,
        all_safety_by_risk, all_distributions, correlations,
    )

    md_path = output_dir / "comparison_tables.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"\nMarkdown tables saved to {md_path}")

    # Save full results JSON
    full_results = {
        "averages": all_averages,
        "overall_scores": all_overalls,
        "krippendorff_alpha": all_alphas,
        "safety_by_risk": all_safety_by_risk,
        "score_distributions": all_distributions,
    }
    if correlations:
        full_results["human_llm_correlation"] = correlations

    json_path = output_dir / "full_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"Full results saved to {json_path}")


if __name__ == "__main__":
    main()
