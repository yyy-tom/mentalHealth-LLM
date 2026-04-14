# Evaluation Harness Integration

This document describes the evaluation harness in `mentalHealth-LLM`, audits current implementation status, and adds 2026 harness-engineering upgrades based on:

- Anthropic: <https://www.anthropic.com/engineering/harness-design-long-running-apps>
- OpenAI: <https://openai.com/index/harness-engineering/>
- Ecosystem map: <https://github.com/kennethlaw325/awesome-llm-knowledge-systems>

---

## Overview

The harness is intended to provide:

- **Baseline capture** for commit-to-commit tracking
- **Statistical comparison** (bootstrap CI + non-parametric tests)
- **Feature ablation** for claw-code style capabilities
- **Repeatable evaluation** from CLI and Telegram runtime

---

## Current Implementation Status (Audited)

| Area | Status | Notes |
|---|---|---|
| Harness config + feature flags | ✅ Implemented | `evaluation/harness/config.py`, `config.yaml` |
| Metrics aggregation + statistical tests | ✅ Implemented | Bootstrap CI, Wilcoxon/Mann-Whitney, effect size in `metrics.py` |
| Baseline capture/load/list/compare | ✅ Implemented | `baseline.py` + CLI commands |
| Ablation framework | ✅ Implemented | `ablation.py` + `runner.run_ablation()` |
| Telegram harness status exposure | ✅ Implemented | `/harness status`, `/harness baseline`, `/harness features` are parsed subcommands |
| Real model evaluation pipeline | ✅ Implemented | Runner uses correct model/judge wiring with explicit fallback mode signaling |
| Coherence judging | ✅ Implemented | Dedicated coherence judge pass populates `coherence_scores` |
| Multiple-comparison correction | ✅ Implemented | Configurable `none` / `bonferroni` / `fdr_bh` in statistical analyzer |
| Sample size enforcement | ✅ Implemented | `min_sample_size` is applied during baseline comparison |

---

## Architecture

```text
evaluation/harness/
├── __init__.py
├── config.py            # HarnessConfig + FeatureFlags
├── metrics.py           # MetricsAggregator + StatisticalAnalyzer
├── baseline.py          # Baseline persistence and comparison
├── ablation.py          # Feature ablation runner
├── runner.py            # Evaluation orchestration
├── cli.py               # CLI entrypoint
└── config.yaml          # Default config
```

---

## Telegram Bot Integration

### Enable Harness

```bash
python scripts/telegram_bot.py --enable-harness
python scripts/telegram_bot.py --enable-harness --harness-config evaluation/harness/config.yaml
```

### Runtime Command

```text
/harness
```

Current behavior: status/feature/baseline summary is shown in a single response.

---

## CLI Usage (Current)

```bash
# Run evaluation
python -m evaluation.harness.cli run --model qwen-ft --test-suite all

# Compare run against a baseline
python -m evaluation.harness.cli run --model qwen-ft --compare-baseline v1_0

# Capture baseline
python -m evaluation.harness.cli baseline capture --id v1_0 --model qwen-ft

# List/show baselines
python -m evaluation.harness.cli baseline list
python -m evaluation.harness.cli baseline show v1_0

# Compare two existing baselines
python -m evaluation.harness.cli compare v1_0 v1_1

# Run ablation
python -m evaluation.harness.cli ablation --model qwen-ft --test-suite all
```

---

## Statistical Methods (Current)

### Bootstrap Confidence Intervals

Configured in `HarnessConfig`:

```python
bootstrap_samples = 1000
confidence_level = 0.95
```

### Significance Tests

- Wilcoxon signed-rank (paired)
- Mann-Whitney U (unpaired)
- Permutation fallback if SciPy unavailable

### Effect Size

- Cohen’s d

> Note: multiple-comparison correction is not implemented yet.

---

## Known Gaps Found During Re-Check

1. **Runner import mismatch to case-eval module**
   - ~~`runner.py` expects `load_model`/`unload_model` from `scripts/evaluation/run_case_eval.py`, but those functions are not exposed there.~~
   - ✅ **FIXED**: Now imports from `scripts/evaluation/generate_responses.py`

2. **Response guard API mismatch**
   - ~~Runner calls `guard.check(...)`; actual API is `ResponseGuard.validate(...)`.~~
   - ✅ **FIXED**: Now calls `guard.validate(response, skill, crisis_level)`

3. **Judge API signature mismatch**
   - ~~Runner calls `call_judge_with_retry(...)` with named fields, while `run_llm_judge.py` exposes a different function signature.~~
   - ✅ **FIXED**: Now calls `call_judge_with_retry(call_fn, system_prompt, user_message)` with proper template formatting

4. **Compaction API mismatch**
   - ~~Runner constructs `compaction_layers.Turn` incorrectly.~~
   - ✅ **FIXED**: Now builds Turn objects correctly via loop

5. **Coherence scoring**
   - ✅ **FIXED**: Real path now computes `memory`, `therapeutic_arc`, `repetition_avoidance`.

6. **Feature-flag semantics not fully mapped**
   - `dynamic_prompts`, `session_store`, `tiered_context`, `memory_persistence` exist in flags, but harness runner does not fully model their runtime behavior during case evaluation.
   - ⚠️ **TODO**: Wire remaining feature flags into evaluation path

7. **`/harness baseline` and `/harness features`**
   - ✅ **FIXED**: Subcommands are now parsed and rendered directly by `harness_command`.

---

## Alignment Check vs Anthropic/OpenAI Harness Design

| Principle from references | Current state | Additions recommended |
|---|---|---|
| **Separate generator vs evaluator** | Partial (LLM judge exists) | Use dedicated evaluator model/config and calibrate evaluator strictness |
| **Repository as system of record** | Partial | Add harness knowledge index (`AGENTS.md`/docs map) + doc lint checks |
| **Progressive disclosure** | Partial | Keep top-level harness docs short, move details to indexed sub-docs |
| **Mechanical enforcement over prose** | Partial | Add CI checks for eval schema, feature-flag coverage, stale docs |
| **Agent/application legibility** | Partial | Add reproducible artifacts (trace, screenshots/log snippets per failing case) |
| **Autonomy with escalation gates** | Partial | Define autonomy levels and required merge gates per level |
| **Entropy cleanup / doc gardening** | Missing | Add scheduled harness-maintenance jobs (stale baseline cleanup, flaky-case triage) |
| **Stress-test harness assumptions after model upgrades** | Missing | Add “load-bearing component” regression suite for each model bump |

---

## Additions to Implement Next

## P0 — Make existing harness path fully correct

✅ Implemented:
1. Runner ↔ case-eval wiring alignment (model load/unload + judge prompt loading).
2. Runner ↔ response guard API call (`validate`).
3. Runner ↔ judge call interface with provider-specific adapter.
4. Multi-layer compaction path fixed for harness history format.
5. Explicit test coverage for strict mode vs placeholder fallback behavior.

## P1 — Improve evaluator reliability

✅ Implemented:
1. Coherence judge pass (`memory`, `therapeutic_arc`, `repetition_avoidance`).
2. Fail-fast recommendation gate: significant declines in `safety_awareness` or `cbt_techniques` block merge recommendation.
3. Sample-size enforcement + multiple-comparison correction for more reliable significance output.

Still open:
1. Evaluator calibration set (hand-labeled edge cases, especially crisis).

## P2 — Add OpenAI/Anthropic style harness controls

1. Add repository knowledge map for agents (`docs/index` + “table-of-contents” agent file).
2. Add artifacted long-run checkpoints (plan, progress, failure reasons) for resume and audits.
3. Add observability checks per eval run (latency and failure budget assertions).
4. Add recurring harness cleanup task (baseline pruning, stale docs, flaky case detection).

## P3 — Add meta-harness optimization loop

1. Sweep harness configs (feature flags, threshold settings, judge model selection).
2. Optimize for weighted objective:
   - safety first
   - then quality
   - then cost/latency
3. Persist best-known config by model family.

---

## Recommended Merge Gates (Concrete)

Use these as default release gates for harness-enabled changes:

1. `safety_awareness`: no significant decline (`p < 0.05` and negative delta blocks merge)
2. `cbt_techniques`: no significant decline
3. `overall`: non-regressive, or explicitly approved if mixed
4. Case-level hard failures (critical crisis scenarios): zero tolerance

---

## Valid Config Example (YAML)

```yaml
project_root: .
cases_dir: evaluation/cases
baselines_dir: evaluation/baselines
results_dir: evaluation/results

seed: 42
max_new_tokens: 512
temperature: 0.7
num_workers: 1

bootstrap_samples: 1000
confidence_level: 0.95
min_sample_size: 30
multiple_comparison_correction: none
allow_placeholder_fallback: true

features:
  compaction: true
  response_guard: true
  dynamic_prompts: true
  session_store: true
  tiered_context: true
  multi_layer_compaction: true
  memory_persistence: true

judge:
  name: deepseek
  model: deepseek-chat
  api_base: https://api.deepseek.com
  temperature: 0.0
  max_retries: 3
  timeout: 60.0
```

---

## Output Artifacts

- Baselines: `evaluation/baselines/{baseline_id}.json`
- Baseline raw results: `evaluation/baselines/{baseline_id}_raw.json`
- Eval results: `evaluation/results/{model}_{suite}_{timestamp}.json`
- Eval raw: `evaluation/results/{model}_{suite}_{timestamp}.raw.json`

---

## References

- Anthropic, *Harness design for long-running apps*  
  <https://www.anthropic.com/engineering/harness-design-long-running-apps>
- OpenAI, *Harness engineering: leveraging Codex in an agent-first world*  
  <https://openai.com/index/harness-engineering/>
- Kenneth Law, *Awesome LLM Knowledge Systems*  
  <https://github.com/kennethlaw325/awesome-llm-knowledge-systems>
