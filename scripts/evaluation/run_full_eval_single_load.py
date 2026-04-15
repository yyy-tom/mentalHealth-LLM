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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    judge_workers: int = 4,
) -> list[dict]:
    """Generate responses on GPU, then judge all turns in parallel via thread pool.

    Within each case, turns are generated sequentially (each depends on prior
    history). But all judge API calls are deferred and executed concurrently,
    since they are network-bound and don't touch the GPU.
    """
    from evaluation.harness.runner import HARNESS_DIMENSIONS

    # ------------------------------------------------------------------
    # Phase 1: Generate all responses (GPU-bound, sequential)
    # ------------------------------------------------------------------
    generated_cases: list[dict] = []

    for ci, case in enumerate(cases):
        log.info("  [gen] Case %d/%d: %s", ci + 1, len(cases), case.get("case_id", "?"))
        case_skill = case.get("skill", "general-support")
        case_risk_level = str(case.get("risk_level", "UNKNOWN"))
        crisis_level = case_risk_level.lower()
        situation = case.get("situation", "")
        user_turns = case.get("user_turns", [])
        session_summary = ""

        compactor = None
        if features.multi_layer_compaction:
            from mental_health_llm import MultiLayerCompactor
            compactor = MultiLayerCompactor()

        guard = None
        if features.response_guard:
            from mental_health_llm import ResponseGuard
            guard = ResponseGuard()

        prompt_builder = None
        if features.dynamic_prompts:
            from mental_health_llm import TherapyPromptBuilder
            prompt_builder = TherapyPromptBuilder()

        history: list[tuple[str, str]] = []
        turn_data: list[dict] = []

        for i, user_msg in enumerate(user_turns):
            history_for_generation = history
            if compactor and len(history) > 4:
                compacted = compactor.compact(
                    history=history, target_tokens=2000, preserve_recent=4,
                )
                history_for_generation = harness._messages_to_history_pairs(compacted.compacted)
                session_summary = compacted.session_summary

            system_prompt = base_system_prompt
            if prompt_builder:
                dynamic_prompt = (
                    prompt_builder.with_skill(case_skill)
                    .with_crisis_context(crisis_level)
                    .with_user_profile(region=str(case.get("region", "US")))
                    .with_session_summary(session_summary)
                    .build()
                )
                if dynamic_prompt:
                    system_prompt = dynamic_prompt

            response = generate_response_multiturn(
                model=model, tokenizer=tokenizer, user_message=user_msg,
                system_prompt=system_prompt, history=history_for_generation,
                use_system_role=use_system_role,
                seed=harness.config.seed + i, max_new_tokens=harness.config.max_new_tokens,
            )

            if guard:
                guard_result = guard.validate(response=response, skill=case_skill, crisis_level=crisis_level)
                response = guard_result.response

            context_str = harness._format_context(history, situation)
            judge_user_msg = judge_user_template.format(
                CONVERSATION_HISTORY=context_str,
                USER_INPUT=user_msg,
                MODEL_RESPONSE=response,
            )

            turn_data.append({
                "turn_index": i + 1,
                "user_message": user_msg,
                "counselor_response": response,
                "judge_user_msg": judge_user_msg,
            })
            history.append((user_msg, response))

        generated_cases.append({
            "case": case,
            "turn_data": turn_data,
            "case_skill": case_skill,
            "case_risk_level": case_risk_level,
            "situation": situation,
        })

    # ------------------------------------------------------------------
    # Phase 2: Judge all turns in parallel (network-bound)
    # ------------------------------------------------------------------
    log.info("  [judge] Scoring %d cases with %d workers...",
             len(generated_cases), judge_workers)

    def _judge_turn(judge_user_msg: str) -> dict:
        try:
            scores = call_judge_with_retry(judge_call_fn, judge_system_prompt, judge_user_msg)
            return {"ok": True, "scores": clamp_scores(scores)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _judge_coherence(situation: str, turns: list[dict]) -> dict:
        try:
            return {"ok": True, "scores": harness._score_coherence(
                judge_call_fn=judge_call_fn,
                judge_system_prompt=judge_system_prompt,
                situation=situation,
                turns=turns,
            )}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=judge_workers) as pool:
        for gc_entry in generated_cases:
            case = gc_entry["case"]
            turn_data = gc_entry["turn_data"]

            turn_futures = []
            for td in turn_data:
                fut = pool.submit(_judge_turn, td["judge_user_msg"])
                turn_futures.append((td, fut))

            turns_for_result: list[dict] = []
            for td, fut in turn_futures:
                judge_result = fut.result()
                if judge_result["ok"]:
                    js = judge_result["scores"]
                    normalized = harness._normalize_dimension_scores(js)
                    risk = js.get("risk_level", "unknown")
                    comment = js.get("overall_comment", "")
                else:
                    normalized = {
                        dim: {"score": "N/A", "evidence": f"Error: {judge_result['error']}"}
                        for dim in HARNESS_DIMENSIONS
                    }
                    risk = "unknown"
                    comment = f"Scoring failed: {judge_result['error']}"

                turns_for_result.append({
                    "turn_index": td["turn_index"],
                    "user_message": td["user_message"],
                    "counselor_response": td["counselor_response"],
                    "scores": normalized,
                    "risk_level": risk,
                    "overall_comment": comment,
                })

            coherence_scores = {}
            if len(turns_for_result) >= 2:
                coh_fut = pool.submit(
                    _judge_coherence, gc_entry["situation"], turns_for_result,
                )
                coh_result = coh_fut.result()
                if coh_result["ok"]:
                    coherence_scores = coh_result["scores"]

            results.append({
                "case_id": case.get("case_id", "unknown"),
                "title": case.get("title", ""),
                "risk_level": gc_entry["case_risk_level"],
                "skill": gc_entry["case_skill"],
                "turns": turns_for_result,
                "coherence_scores": coherence_scores,
            })

    return results


def run_model(
    model_id: str,
    config: HarnessConfig,
    output_dir: Path,
    baseline_only: bool,
    test_suite: str,
    judge_workers: int = 4,
    case_start: int = 0,
    case_end: int | None = None,
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
    total_cases = len(cases)
    cases = cases[case_start:case_end]
    log.info("Loaded %d/%d test cases (suite=%s, slice=[%d:%s])",
             len(cases), total_cases, test_suite, case_start,
             str(case_end) if case_end is not None else "end")

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
        judge_workers=judge_workers,
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
    parser.add_argument(
        "--judge-workers", type=int, default=4,
        help="Number of concurrent judge API threads (default: 4)",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=None,
        help="Pin to a specific GPU (sets CUDA_VISIBLE_DEVICES). If not set, uses default.",
    )
    parser.add_argument(
        "--case-start", type=int, default=0,
        help="Start index for case slice (0-based, inclusive). Used for multi-GPU splits.",
    )
    parser.add_argument(
        "--case-end", type=int, default=None,
        help="End index for case slice (exclusive). If not set, runs all cases from --case-start.",
    )
    args = parser.parse_args()

    if args.gpu_id is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        log.info("Pinned to GPU %d", args.gpu_id)

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
            judge_workers=args.judge_workers,
            case_start=args.case_start,
            case_end=args.case_end,
        )

    log.info("All models complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
