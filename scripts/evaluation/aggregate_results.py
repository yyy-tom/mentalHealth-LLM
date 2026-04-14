#!/usr/bin/env python3
"""
Aggregate LLM judge scores and generate markdown tables for the paper.

Supports multi-judge scoring: parses new-format filenames ({model}_{judge}_run{N}.json)
and old-format filenames ({model}_run{N}.json) for backward compatibility.

Computes per-dimension averages, overall 0-8 scores, CBT technique subscore
averages, Krippendorff's alpha (intra-judge and inter-judge), risk-level
breakdowns, score distributions, and optional human-LLM correlation.

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

DIMENSIONS = ["empathy", "cbt", "guided_discovery", "safety"]
CBT_SUBSCORES = [
    "cognitive_reconstruction",
    "behavioral_activation",
    "positive_encouragement",
    "psychoeducation",
]
OVERALL_FIELD = "overall_score_0_to_8"
CATEGORY_SUBSCORES = [f"cbt_{s}" for s in CBT_SUBSCORES]
DIMENSION_LABELS = {
    "empathy": "Empathy",
    "cbt": "CBT",
    "guided_discovery": "Guided Disc.",
    "safety": "Safety",
}
CBT_SUBSCORE_LABELS = {
    "cbt_cognitive_reconstruction": "CBT: Cognitive Recon.",
    "cbt_behavioral_activation": "CBT: Behavioral Act.",
    "cbt_positive_encouragement": "CBT: Positive Encour.",
    "cbt_psychoeducation": "CBT: Psychoeducation",
}

MODEL_DISPLAY_NAMES = {
    "qwen2.5-7b": "Qwen 2.5 7B",
    "gemma2-9b": "Gemma 2 9B",
    "mistral-7b": "Mistral 7B",
    "llama-3.1-8b": "Llama 3.1 8B",
}

KNOWN_JUDGES = {"gpt-4o", "deepseek", "gemini", "claude"}


def normalize_score(value):
    """Normalize a score value to float/int or 'N/A'."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip().upper()
        if stripped == "N/A":
            return "N/A"
        try:
            return float(stripped)
        except ValueError:
            return "N/A"
    return "N/A"


def round_to_half(value: float) -> float:
    """Round to nearest 0.5."""
    return round(value * 2) / 2


def collect_scores(scores_dir: Path) -> dict:
    """Load all score files and organize by model -> judge -> run -> sample.

    Handles both new-format ({model}_{judge}_run{N}.json) and old-format
    ({model}_run{N}.json) filenames. For old-format files, attempts to read
    judge_model from file metadata; falls back to "unknown".

    Returns:
        {model_key: {judge: {run_id: {sample_id: {dim: score, ...}}}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for path in sorted(scores_dir.glob("*_run*.json")):
        match = re.match(r"(.+)_run(\d+)\.json", path.name)
        if not match:
            continue
        prefix = match.group(1)
        run_id = int(match.group(2))

        # Try to extract judge from filename: check if prefix ends with a known judge
        model_key = None
        judge = None
        for j in KNOWN_JUDGES:
            suffix = f"_{j}"
            if prefix.endswith(suffix):
                model_key = prefix[: -len(suffix)]
                judge = j
                break

        with open(path) as f:
            content = json.load(f)

        # Fallback for old-format files without judge in filename
        if judge is None:
            model_key = prefix
            # Try to infer judge from file metadata
            metadata = content.get("metadata", {})
            judge_model_name = metadata.get("judge_model", "")
            if "gpt-4o" in judge_model_name:
                judge = "gpt-4o"
            elif "claude" in judge_model_name:
                judge = "claude"
            elif "deepseek" in judge_model_name:
                judge = "deepseek"
            elif "gemini" in judge_model_name:
                judge = "gemini"
            else:
                judge = "unknown"

        for entry in content["scores"]:
            sid = entry["sample_id"]
            scores = {"risk_level": entry.get("risk_level", "unknown")}
            for dim in DIMENSIONS:
                dim_data = entry.get(dim, {})
                score = dim_data.get("score", "N/A") if isinstance(dim_data, dict) else "N/A"
                scores[dim] = normalize_score(score)

            overall_data = entry.get(OVERALL_FIELD, {})
            overall_score = (
                overall_data.get("score", "N/A")
                if isinstance(overall_data, dict)
                else "N/A"
            )
            scores[OVERALL_FIELD] = normalize_score(overall_score)

            subscores = entry.get("cbt_subscores", {})
            if not isinstance(subscores, dict):
                # Backward-compatible fallback if nested under cbt.subscores
                cbt_data = entry.get("cbt", {})
                subscores = cbt_data.get("subscores", {}) if isinstance(cbt_data, dict) else {}
            for sub in CBT_SUBSCORES:
                key = f"cbt_{sub}"
                sub_data = subscores.get(sub, {})
                sub_score = (
                    sub_data.get("score", "N/A")
                    if isinstance(sub_data, dict)
                    else "N/A"
                )
                scores[key] = normalize_score(sub_score)
            data[model_key][judge][run_id][sid] = scores

    return dict(data)


def flatten_judge_data(model_judges: dict) -> dict:
    """Merge all judges' runs into a flat {run_key: samples} dict.

    Creates synthetic run keys like "gpt-4o_1", "deepseek_2" so that existing
    compute functions (which expect {run_id: {sample_id: scores}}) work unchanged.

    Args:
        model_judges: {judge: {run_id: {sample_id: scores}}}

    Returns:
        {synthetic_run_key: {sample_id: scores}}
    """
    flat = {}
    for judge, runs in model_judges.items():
        for run_id, samples in runs.items():
            flat[f"{judge}_{run_id}"] = samples
    return flat


def compute_dimension_averages(model_data: dict) -> dict:
    """Compute per-dimension averages across all runs and samples.

    Returns {dim: rounded_average} for the four rubric dimensions.
    """
    dim_values = defaultdict(list)
    overall_values = []

    for run_id, samples in model_data.items():
        for sid, scores in samples.items():
            for dim in DIMENSIONS:
                val = scores.get(dim, "N/A")
                if isinstance(val, (int, float)):
                    dim_values[dim].append(val)
            overall_val = scores.get(OVERALL_FIELD, "N/A")
            if isinstance(overall_val, (int, float)):
                overall_values.append(overall_val)
            else:
                row_scores = [scores.get(dim, "N/A") for dim in DIMENSIONS]
                if all(isinstance(v, (int, float)) for v in row_scores):
                    overall_values.append(sum(row_scores))

    averages = {}
    for dim in DIMENSIONS:
        values = dim_values[dim]
        if values:
            averages[dim] = round_to_half(np.mean(values))
        else:
            averages[dim] = "N/A"

    if overall_values:
        averages[OVERALL_FIELD] = round_to_half(np.mean(overall_values))
    else:
        averages[OVERALL_FIELD] = "N/A"

    return averages


def compute_overall(averages: dict) -> float | str:
    """Mean overall score on the rubric 0-8 scale."""
    overall = averages.get(OVERALL_FIELD)
    if isinstance(overall, (int, float)):
        return round_to_half(float(overall))
    numeric = [v for v in averages.values() if isinstance(v, (int, float))]
    if numeric:
        # Backward-compatible fallback for older files without explicit overall field
        return round_to_half(float(np.sum(numeric)))
    return "N/A"


def compute_cbt_subscore_averages(model_data: dict) -> dict:
    """Compute averages for the four CBT technique subscores."""
    vals = defaultdict(list)
    for _, samples in model_data.items():
        for _, scores in samples.items():
            for key in CATEGORY_SUBSCORES:
                score = scores.get(key, "N/A")
                if isinstance(score, (int, float)):
                    vals[key].append(score)

    out = {}
    for key in CATEGORY_SUBSCORES:
        if vals[key]:
            out[key] = round_to_half(np.mean(vals[key]))
        else:
            out[key] = "N/A"
    return out


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


def compute_inter_judge_alpha(model_judges: dict) -> dict:
    """Compute Krippendorff's alpha across judges for a single model.

    Each judge-run combination is treated as a separate rater. For each dimension,
    builds a raters x samples matrix across all judges and runs.

    Args:
        model_judges: {judge: {run_id: {sample_id: scores}}}

    Returns:
        {dim: alpha_value}
    """
    import krippendorff

    # Build list of (judge, run_id) tuples as raters
    raters = []
    for judge, runs in sorted(model_judges.items()):
        for run_id in sorted(runs.keys()):
            raters.append((judge, run_id))

    if len(raters) < 2:
        return {dim: "N/A (insufficient raters)" for dim in DIMENSIONS}

    # Check we have at least 2 distinct judges
    distinct_judges = set(j for j, _ in raters)
    if len(distinct_judges) < 2:
        return {dim: "N/A (single judge)" for dim in DIMENSIONS}

    # Get all sample IDs
    all_sids = set()
    for judge, runs in model_judges.items():
        for run_id, samples in runs.items():
            all_sids.update(samples.keys())
    all_sids = sorted(all_sids)

    alphas = {}
    for dim in DIMENSIONS:
        matrix = []
        for judge, run_id in raters:
            row = []
            for sid in all_sids:
                val = model_judges[judge].get(run_id, {}).get(sid, {}).get(dim, "N/A")
                if isinstance(val, (int, float)):
                    row.append(val)
                else:
                    row.append(np.nan)
            matrix.append(row)

        reliability_data = np.array(matrix)

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
    """Histogram of 0-2 scores per dimension."""
    distributions = {}

    for dim in DIMENSIONS:
        counts = {i: 0 for i in range(0, 3)}
        na_count = 0
        for run_id, samples in model_data.items():
            for sid, scores in samples.items():
                val = scores.get(dim, "N/A")
                if isinstance(val, (int, float)) and 0 <= val <= 2:
                    counts[int(round(val))] += 1
                else:
                    na_count += 1
        distributions[dim] = {"counts": counts, "na": na_count}

    return distributions


def compute_spearman(flat_scores_data: dict, human_scores_path: str) -> dict | None:
    """Compute Spearman correlation between LLM judge and human scores.

    Args:
        flat_scores_data: {model_key: {run_key: {sample_id: scores}}}
    """
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

        for model_key, model_data in flat_scores_data.items():
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


def _format_comparison_table(all_averages: dict, all_overalls: dict, title: str) -> list[str]:
    """Generate a model comparison table section."""
    lines = []
    lines.append(f"## {title}\n")
    header = (
        "| Model | "
        + " | ".join(DIMENSION_LABELS[d] for d in DIMENSIONS)
        + " | Overall (0-8) |"
    )
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
    return lines


def _format_cbt_subscore_table(all_subscores: dict, title: str) -> list[str]:
    """Generate a CBT subscore comparison table section."""
    lines = []
    lines.append(f"## {title}\n")
    ordered = list(CBT_SUBSCORE_LABELS.keys())
    header = "| Model | " + " | ".join(CBT_SUBSCORE_LABELS[k] for k in ordered) + " |"
    sep = "|" + "|".join(["---"] * (len(ordered) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for model_key in sorted(all_subscores.keys()):
        vals = all_subscores[model_key]
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        cells = [name]
        for key in ordered:
            v = vals.get(key, "N/A")
            cells.append(f"{v}" if isinstance(v, str) else f"{v:.1f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def _format_alpha_table(all_alphas: dict, title: str) -> list[str]:
    """Generate a Krippendorff's alpha table section."""
    lines = []
    lines.append(f"## {title}\n")
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
    return lines


def generate_markdown(
    combined_averages: dict,
    combined_overalls: dict,
    combined_cbt_subscores: dict,
    combined_alphas: dict,
    combined_safety_by_risk: dict,
    combined_distributions: dict,
    per_judge_results: dict,
    inter_judge_alphas: dict,
    correlations: dict | None,
) -> str:
    """Generate markdown tables for the paper.

    Sections:
    1. Combined Model Comparison (all judges averaged)
    2. Per-Judge Model Comparison (one table per judge)
    3. Intra-Judge Consistency (Krippendorff alpha within each judge)
    4. Inter-Judge Agreement (Krippendorff alpha across judges)
    5. Safety by Risk Level (combined)
    6. Score Distributions (combined)
    7. Human-LLM Correlation (if provided)
    """
    lines = []

    # 1. Combined Model Comparison
    lines.extend(_format_comparison_table(
        combined_averages, combined_overalls, "Model Comparison (All Judges Combined)",
    ))
    lines.extend(_format_cbt_subscore_table(
        combined_cbt_subscores, "CBT Technique Subscores (All Judges Combined)",
    ))

    # 2. Per-Judge Model Comparison
    judges = sorted(per_judge_results.keys())
    if len(judges) > 1:
        for judge in judges:
            jr = per_judge_results[judge]
            lines.extend(_format_comparison_table(
                jr["averages"], jr["overall_scores"],
                f"Model Comparison ({judge})",
            ))
            lines.extend(_format_cbt_subscore_table(
                jr["cbt_subscores"], f"CBT Technique Subscores ({judge})",
            ))

    # 3. Intra-Judge Consistency
    if len(judges) > 1:
        for judge in judges:
            jr = per_judge_results[judge]
            lines.extend(_format_alpha_table(
                jr["krippendorff_alpha"],
                f"Intra-Judge Consistency ({judge}, Krippendorff's Alpha)",
            ))
    else:
        lines.extend(_format_alpha_table(
            combined_alphas, "Inter-Run Consistency (Krippendorff's Alpha)",
        ))

    # 4. Inter-Judge Agreement
    if inter_judge_alphas:
        lines.extend(_format_alpha_table(
            inter_judge_alphas, "Inter-Judge Agreement (Krippendorff's Alpha)",
        ))

    # 5. Safety by Risk Level
    lines.append("## Safety Score by Risk Level\n")
    lines.append("| Model | Low | Medium | High |")
    lines.append("|---|---|---|---|")

    for model_key in sorted(combined_safety_by_risk.keys()):
        sbr = combined_safety_by_risk[model_key]
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

    # 6. Score Distributions
    lines.append("## Score Distributions\n")
    for model_key in sorted(combined_distributions.keys()):
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        lines.append(f"### {name}\n")
        lines.append("| Dimension | 0 | 1 | 2 | N/A |")
        lines.append("|---|---|---|---|---|")

        dists = combined_distributions[model_key]
        for dim in DIMENSIONS:
            dist = dists[dim]
            label = DIMENSION_LABELS[dim]
            counts = dist["counts"]
            row = [label] + [str(counts[i]) for i in range(0, 3)] + [str(dist["na"])]
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    # 7. Spearman correlation (if provided)
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

    # Collect all scores: {model_key: {judge: {run_id: {sample_id: scores}}}}
    scores_data = collect_scores(scores_dir)
    if not scores_data:
        print(f"No score files found in {scores_dir}")
        return

    print(f"Found scores for models: {list(scores_data.keys())}")
    for model_key, judges in scores_data.items():
        for judge, runs in judges.items():
            total = sum(len(s) for s in runs.values())
            print(f"  {model_key} / {judge}: {len(runs)} run(s), {total} scored entries")

    # Identify all judges across all models
    all_judges = set()
    for model_key, judges in scores_data.items():
        all_judges.update(judges.keys())
    print(f"Judges found: {sorted(all_judges)}")

    # --- Per-judge analysis ---
    per_judge_results = {}
    for judge in sorted(all_judges):
        print(f"\n--- Per-judge analysis: {judge} ---")
        judge_averages = {}
        judge_overalls = {}
        judge_cbt_subscores = {}
        judge_alphas = {}

        for model_key, judges in sorted(scores_data.items()):
            if judge not in judges:
                continue
            model_data = judges[judge]  # {run_id: {sample_id: scores}}
            judge_averages[model_key] = compute_dimension_averages(model_data)
            judge_overalls[model_key] = compute_overall(judge_averages[model_key])
            judge_cbt_subscores[model_key] = compute_cbt_subscore_averages(model_data)
            judge_alphas[model_key] = compute_krippendorff_alpha(model_data)

            print(f"  {model_key}: avg={judge_averages[model_key]}, overall={judge_overalls[model_key]}")

        per_judge_results[judge] = {
            "averages": judge_averages,
            "overall_scores": judge_overalls,
            "cbt_subscores": judge_cbt_subscores,
            "krippendorff_alpha": judge_alphas,
        }

    # --- Combined analysis (all judges merged) ---
    print("\n--- Combined analysis (all judges) ---")
    combined_averages = {}
    combined_overalls = {}
    combined_cbt_subscores = {}
    combined_alphas = {}
    combined_safety_by_risk = {}
    combined_distributions = {}

    for model_key, judges in sorted(scores_data.items()):
        flat = flatten_judge_data(judges)
        combined_averages[model_key] = compute_dimension_averages(flat)
        combined_overalls[model_key] = compute_overall(combined_averages[model_key])
        combined_cbt_subscores[model_key] = compute_cbt_subscore_averages(flat)
        combined_alphas[model_key] = compute_krippendorff_alpha(flat)
        combined_safety_by_risk[model_key] = compute_safety_by_risk(flat)
        combined_distributions[model_key] = compute_score_distributions(flat)

        print(f"  {model_key}: avg={combined_averages[model_key]}, overall={combined_overalls[model_key]}")

    # --- Inter-judge agreement ---
    inter_judge_alphas = {}
    if len(all_judges) >= 2:
        print("\n--- Inter-judge agreement ---")
        for model_key, judges in sorted(scores_data.items()):
            inter_judge_alphas[model_key] = compute_inter_judge_alpha(judges)
            print(f"  {model_key}: {inter_judge_alphas[model_key]}")

    # Spearman correlation (optional) -- uses combined flat data
    correlations = None
    if args.human_scores:
        print(f"\nComputing human-LLM correlation from {args.human_scores}")
        flat_all = {}
        for model_key, judges in scores_data.items():
            flat_all[model_key] = flatten_judge_data(judges)
        correlations = compute_spearman(flat_all, args.human_scores)

    # Generate markdown
    markdown = generate_markdown(
        combined_averages, combined_overalls, combined_cbt_subscores, combined_alphas,
        combined_safety_by_risk, combined_distributions,
        per_judge_results, inter_judge_alphas, correlations,
    )

    md_path = output_dir / "comparison_tables.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"\nMarkdown tables saved to {md_path}")

    # Save full results JSON
    full_results = {
        "combined": {
            "averages": combined_averages,
            "overall_scores": combined_overalls,
            "cbt_subscores": combined_cbt_subscores,
            "krippendorff_alpha": combined_alphas,
            "safety_by_risk": combined_safety_by_risk,
            "score_distributions": combined_distributions,
        },
        "per_judge": per_judge_results,
    }
    if inter_judge_alphas:
        full_results["combined"]["inter_judge_alpha"] = inter_judge_alphas
    if correlations:
        full_results["combined"]["human_llm_correlation"] = correlations

    json_path = output_dir / "full_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"Full results saved to {json_path}")


if __name__ == "__main__":
    main()
