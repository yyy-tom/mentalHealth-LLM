#!/usr/bin/env python3
"""
Merge evaluation results from multiple GPU shards into unified output.

Usage:
    python scripts/evaluation/merge_gpu_results.py \
        --input-dirs evaluation/results/multi_gpu/gpu_0 evaluation/results/multi_gpu/gpu_1 ... \
        --output-dir evaluation/results/full_eval/qwen-ft \
        --model qwen-ft
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.harness.config import HarnessConfig
from evaluation.harness.metrics import MetricsAggregator, StatisticalAnalyzer
from evaluation.harness.runner import EvalResults

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def find_raw_files(input_dir: Path, variant: str) -> list[Path]:
    files = sorted(input_dir.glob(f"{variant}_*.raw.json"))
    if not files:
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                files.extend(sorted(subdir.glob(f"{variant}_*.raw.json")))
    return files


def merge_variant(input_dirs: list[Path], variant: str) -> list[dict]:
    all_results = []
    for d in input_dirs:
        for raw_file in find_raw_files(d, variant):
            with open(raw_file) as f:
                data = json.load(f)
            all_results.extend(data)
            log.info("  %s: %d cases from %s", variant, len(data), raw_file.name)
    seen = set()
    deduped = []
    for r in all_results:
        cid = r.get("case_id", "")
        if cid not in seen:
            seen.add(cid)
            deduped.append(r)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Merge multi-GPU evaluation shards")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="GPU shard directories")
    parser.add_argument("--output-dir", required=True, help="Merged output directory")
    parser.add_argument("--model", required=True, help="Model ID for metadata")
    parser.add_argument("--config", default="evaluation/harness/config.yaml")
    args = parser.parse_args()

    input_dirs = [Path(d) for d in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    config = HarnessConfig.from_yaml(config_path) if config_path.exists() else HarnessConfig()
    aggregator = MetricsAggregator(
        bootstrap_samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
    )

    variants = ["baseline", "no_compaction", "no_tiered_context", "no_memory_persistence",
                "no_response_guard", "no_dynamic_prompts", "no_session_store", "all_features_off"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_data: dict[str, list[dict]] = {}

    for variant in variants:
        log.info("Merging variant: %s", variant)
        merged = merge_variant(input_dirs, variant)
        if not merged:
            log.warning("  No data found for %s, skipping", variant)
            continue
        merged_data[variant] = merged
        log.info("  Total: %d cases", len(merged))

        metrics = aggregator.aggregate(merged)
        result = EvalResults(
            model=args.model,
            test_suite="all",
            timestamp=datetime.now(timezone.utc).isoformat(),
            commit="merged",
            features={},
            metrics=metrics,
            raw_results=merged,
            evaluation_mode="real",
        )
        result.save(output_dir / f"{variant}_{ts}.json")

    if "baseline" in merged_data and len(merged_data) > 1:
        log.info("Generating comparison reports...")
        analyzer = StatisticalAnalyzer(
            alpha=0.05,
            min_sample_size=config.min_sample_size,
            multiple_comparison=config.multiple_comparison_correction,
        )
        from evaluation.harness.runner import EvaluationHarness
        harness = EvaluationHarness(config)
        baseline_scores = harness._extract_scores(merged_data["baseline"])

        rows = []
        for variant in variants:
            if variant == "baseline" or variant not in merged_data:
                continue
            comparisons = analyzer.compare(
                harness._extract_scores(merged_data[variant]),
                baseline_scores,
            )
            for comp in comparisons:
                rows.append({
                    "variant": variant,
                    "dimension": comp.dimension,
                    "baseline_mean": round(comp.baseline_mean, 4),
                    "variant_mean": round(comp.current_mean, 4),
                    "delta": round(comp.difference, 4),
                    "p_value": round(comp.p_value, 4),
                    "significant": comp.is_significant,
                })

        with open(output_dir / f"ablation_comparison_{ts}.json", "w") as f:
            json.dump(rows, f, indent=2)

        md_lines = [
            f"# Merged Ablation Comparison — {args.model}\n",
            f"Timestamp: {ts}  \nShards: {len(input_dirs)} GPUs\n",
            "| Variant | Dimension | Baseline | Variant | Delta | p-value | Sig? |",
            "|---------|-----------|----------|---------|-------|---------|------|",
        ]
        for row in rows:
            sig = "yes" if row["significant"] else ""
            md_lines.append(
                f"| {row['variant']} | {row['dimension']} "
                f"| {row['baseline_mean']:.3f} | {row['variant_mean']:.3f} "
                f"| {row['delta']:+.3f} | {row['p_value']:.4f} | {sig} |"
            )
        (output_dir / f"ablation_comparison_{ts}.md").write_text("\n".join(md_lines))
        log.info("Comparison report saved.")

    log.info("Done. Merged results in %s", output_dir)


if __name__ == "__main__":
    main()
