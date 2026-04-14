#!/usr/bin/env python3
"""
Full evaluation pipeline with a single model load.

Loads the model ONCE, then runs:
  1. Baseline evaluation (all features ON)
  2. Feature-off ablation runs (one feature disabled per run)
  3. Comparison reports against baseline

This avoids the N x load_model overhead of calling the harness CLI separately
for each ablation variant.

Usage:
    python scripts/evaluation/run_full_eval_single_load.py \
        --model qwen-ft \
        --config evaluation/harness/config.yaml \
        --output-dir evaluation/results/full_eval

    # Skip ablation, just capture baseline
    python scripts/evaluation/run_full_eval_single_load.py \
        --model qwen-ft --baseline-only

    # Run multiple models sequentially
    python scripts/evaluation/run_full_eval_single_load.py \
        --model qwen-ft gemma-ft mistral-ft
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.harness.config import FeatureFlags, HarnessConfig
from evaluation.harness.metrics import MetricsAggregator, StatisticalAnalyzer
from evaluation.harness.runner import EvalResults, EvaluationHarness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("eval_run.log")],
)
log = logging.getLogger(__name__)

# Ablation variants: (output_name, flag_to_disable)
# flag_to_disable=None means disable ALL features
ABLATION_VARIANTS: list[tuple[str, str | None]] = [
    ("no_compaction",         "multi_layer_compaction"),
    ("no_tiered_context",     "tiered_context"),
    ("no_memory_persistence", "memory_persistence"),
    ("no_response_guard",     "response_guard"),
    ("no_dynamic_prompts",    "dynamic_prompts"),
    ("no_session_store",      "session_store"),
    ("all_features_off",      None),
]


def build_ablation_flags(flag_to_disable: str | None) -> FeatureFlags:
    if flag_to_disable is None:
        return FeatureFlags.all_off()
    flags = FeatureFlags.all_on()
    setattr(flags, flag_to_disable, False)
    return flags


def run_cases_with_loaded_model(
    *,
    harness: EvaluationHarness,
    model,
    tokenizer,
    cases: list[dict],
    features: FeatureFlags,
    generate_response_multiturn,
    base_system_prompt: str,
    judge_system_prompt: str,
    judge_user_template: str,
    call_judge_with_retry,
    judge_call_fn,
    clamp_scores,
    use_system_role: bool,
) -> list[dict]:
    """Run all cases with a pre-loaded model — no load/unload inside."""
    results = []
    for i, case in enumerate(cases):
        log.info("  Case %d/%d: %s", i + 1, len(cases), case.get("case_id", "?"))
        result = harness._evaluate_single_case(
            model=model,
            tokenizer=tokenizer,
            case=case,
            features=features,
            generate_response_multiturn=generate_response_multiturn,
            base_system_prompt=base_system_prompt,
            judge_system_prompt=judge_system_prompt,
            judge_user_template=judge_user_template,
            call_judge_with_retry=call_judge_with_retry,
            judge_call_fn=judge_call_fn,
            clamp_scores=clamp_scores,
            use_system_role=use_system_role,
        )
        results.append(result)
    return results


def run_model(
    model_id: str,
    config: HarnessConfig,
    output_dir: Path,
    baseline_only: bool,
    test_suite: str,
) -> None:
    log.info("=" * 70)
    log.info("Model: %s", model_id)
    log.info("=" * 70)

    model_path = config.model_registry.get(model_id)
    if not model_path:
        raise ValueError(f"Model '{model_id}' not in registry. Check config.yaml.")

    from scripts.evaluation.generate_responses import load_model, unload_model
    from scripts.evaluation.run_case_eval import (
        SYSTEM_PROMPT,
        _supports_system_role,
        clamp_scores,
        generate_response_multiturn,
    )
    from scripts.evaluation.run_llm_judge import (
        call_anthropic,
        call_judge_with_retry,
        call_openai,
        load_judge_prompt,
    )

    harness = EvaluationHarness(config)
    judge_call_fn = harness._build_judge_call_fn(call_openai, call_anthropic)

    prompt_path = config.project_root / "evaluation" / "llm_judge_prompt.md"
    judge_system_prompt, judge_user_template = load_judge_prompt(
        str(prompt_path) if prompt_path.exists() else None
    )

    cases = harness._load_test_cases(test_suite)
    log.info("Loaded %d test cases (suite=%s)", len(cases), test_suite)

    # ------------------------------------------------------------------ #
    # Load model ONCE                                                       #
    # ------------------------------------------------------------------ #
    log.info("Loading model from %s ...", model_path)
    t_load = time.time()
    model, tokenizer = load_model(model_path)
    model.eval()
    use_system_role = _supports_system_role(tokenizer)
    log.info("Model loaded in %.1fs", time.time() - t_load)

    model_output_dir = output_dir / model_id
    model_output_dir.mkdir(parents=True, exist_ok=True)

    aggregator = MetricsAggregator(
        bootstrap_samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    shared_kwargs = dict(
        harness=harness,
        model=model,
        tokenizer=tokenizer,
        cases=cases,
        generate_response_multiturn=generate_response_multiturn,
        base_system_prompt=SYSTEM_PROMPT,
        judge_system_prompt=judge_system_prompt,
        judge_user_template=judge_user_template,
        call_judge_with_retry=call_judge_with_retry,
        judge_call_fn=judge_call_fn,
        clamp_scores=clamp_scores,
        use_system_role=use_system_role,
    )

    # ------------------------------------------------------------------ #
    # Pass 1: baseline (all features ON)                                   #
    # ------------------------------------------------------------------ #
    log.info("--- Pass: baseline (all features ON) ---")
    baseline_features = FeatureFlags.all_on()
    t0 = time.time()
    baseline_raw = run_cases_with_loaded_model(features=baseline_features, **shared_kwargs)
    log.info("Baseline done in %.1fs", time.time() - t0)

    baseline_metrics = aggregator.aggregate(baseline_raw)
    EvalResults(
        model=model_id,
        test_suite=test_suite,
        timestamp=datetime.now(timezone.utc).isoformat(),
        commit=harness._get_current_commit(),
        features=baseline_features.to_dict(),
        metrics=baseline_metrics,
        raw_results=baseline_raw,
        evaluation_mode="real",
    ).save(model_output_dir / f"baseline_{ts}.json")

    baseline_id = f"{model_id}_term2_baseline"
    harness.baseline_manager.capture(
        baseline_id=baseline_id,
        model=model_id,
        test_suite=test_suite,
        metrics=baseline_metrics,
        features=baseline_features,
        raw_results=baseline_raw,
        description=f"Term 2 baseline — all features ON — {ts}",
    )
    log.info("Baseline captured as '%s'", baseline_id)

    if baseline_only:
        log.info("--baseline-only set, skipping ablation.")
        unload_model(model, tokenizer)
        return

    # ------------------------------------------------------------------ #
    # Passes 2-N: ablation variants (model stays loaded)                   #
    # ------------------------------------------------------------------ #
    ablation_raw: dict[str, list[dict]] = {}

    for variant_name, flag_to_disable in ABLATION_VARIANTS:
        log.info("--- Pass: %s ---", variant_name)
        flags = build_ablation_flags(flag_to_disable)
        t0 = time.time()
        raw = run_cases_with_loaded_model(features=flags, **shared_kwargs)
        log.info("%s done in %.1fs", variant_name, time.time() - t0)

        metrics = aggregator.aggregate(raw)
        EvalResults(
            model=model_id,
            test_suite=test_suite,
            timestamp=datetime.now(timezone.utc).isoformat(),
            commit=harness._get_current_commit(),
            features=flags.to_dict(),
            metrics=metrics,
            raw_results=raw,
            evaluation_mode="real",
        ).save(model_output_dir / f"{variant_name}_{ts}.json")
        ablation_raw[variant_name] = raw

    # ------------------------------------------------------------------ #
    # Unload model — GPU free from here                                    #
    # ------------------------------------------------------------------ #
    log.info("Unloading model...")
    unload_model(model, tokenizer)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Comparison reports (CPU only)                                        #
    # ------------------------------------------------------------------ #
    log.info("--- Generating comparison reports ---")
    analyzer = StatisticalAnalyzer(
        alpha=0.05,
        min_sample_size=config.min_sample_size,
        multiple_comparison=config.multiple_comparison_correction,
    )
    baseline_scores = harness._extract_scores(baseline_raw)
    rows = []

    for variant_name, _ in ABLATION_VARIANTS:
        if variant_name not in ablation_raw:
            continue
        comparisons = analyzer.compare(
            harness._extract_scores(ablation_raw[variant_name]),
            baseline_scores,
        )
        for comp in comparisons:
            rows.append({
                "variant": variant_name,
                "dimension": comp.dimension,
                "baseline_mean": round(comp.baseline_mean, 4),
                "variant_mean": round(comp.current_mean, 4),
                "delta": round(comp.difference, 4),
                "p_value": round(comp.p_value, 4),
                "effect_size": round(comp.effect_size, 4),
                "significant": comp.is_significant,
                "interpretation": comp.interpretation,
            })

    json_path = model_output_dir / f"ablation_comparison_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    md_lines = [
        f"# Ablation Comparison — {model_id}\n",
        f"Baseline: `{baseline_id}`  \nTimestamp: {ts}\n",
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
    (model_output_dir / f"ablation_comparison_{ts}.md").write_text("\n".join(md_lines))
    log.info("Reports saved to %s", model_output_dir)
    log.info("Done: %s", model_id)


def main():
    parser = argparse.ArgumentParser(
        description="Full evaluation with single model load per model"
    )
    parser.add_argument(
        "--model", nargs="+", default=["qwen-ft"],
        help="Model key(s) from config registry (default: qwen-ft)",
    )
    parser.add_argument(
        "--config", default="evaluation/harness/config.yaml",
        help="Path to harness config YAML",
    )
    parser.add_argument(
        "--output-dir", default="evaluation/results/full_eval",
        help="Directory to write all results",
    )
    parser.add_argument(
        "--test-suite", default="all",
        help="Test suite: all | crisis | cbt | psychoeducation | empathetic | professional | general",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only capture baseline, skip ablation passes",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = HarnessConfig.from_yaml(config_path) if config_path.exists() else HarnessConfig()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_id in args.model:
        run_model(
            model_id=model_id,
            config=config,
            output_dir=output_dir,
            baseline_only=args.baseline_only,
            test_suite=args.test_suite,
        )

    log.info("All models complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
