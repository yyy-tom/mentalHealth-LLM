# Mental Health LLM Optimization Plan

> Comprehensive optimization roadmap combining Claw-Code architecture patterns and Harness Engineering evaluation framework.

**Document Version:** 1.0  
**Created:** 2026-04-05  
**Reference Commit:** d44a8dd (baseline) → HEAD (current)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Part I: Claw-Code Architecture Optimization](#part-i-claw-code-architecture-optimization)
3. [Part II: Harness Engineering Evaluation Framework](#part-ii-harness-engineering-evaluation-framework)
4. [Part III: Academic Evaluation Credibility](#part-iii-academic-evaluation-credibility)
5. [Part IV: Implementation Roadmap](#part-iv-implementation-roadmap)
6. [Appendix: References & Citations](#appendix-references--citations)

---

## Executive Summary

This document outlines a two-pronged optimization strategy:

1. **Claw-Code Patterns**: Memory management, context compaction, and session persistence inspired by the [claw-code](https://github.com/ultraworkers/claw-code) architecture
2. **Harness Engineering**: Rigorous evaluation infrastructure based on [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [Prometheus 2](https://github.com/prometheus-eval/prometheus-eval)

### Current Implementation Status

| Component | Claw-Code Reference | Implementation | Status |
|-----------|---------------------|----------------|--------|
| Self-Healing Memory | Memory persistence + reconciliation | `memory_persistence.py` | ✅ 100% |
| Hot/Warm/Cold Tiers | Dynamic context tiering | `tiered_context.py` | ✅ 100% |
| 5-Layer Compaction | Progressive summarization | `compaction_layers.py` | ✅ 100% |
| Evaluation Harness | — | `run_case_eval.py` | 60% |
| Ablation Framework | — | Not implemented | 0% |

### Part I Implementation Complete

The following modules have been implemented:

1. **`mental_health_llm/memory_persistence.py`** - Self-healing memory with:
   - SQLite persistence for compaction summaries
   - Integrity verification via checksums
   - Semantic search for cross-session recall
   - Auto-repair for corrupted records

2. **`mental_health_llm/tiered_context.py`** - Hot/Warm/Cold tiers with:
   - Dynamic tier transitions based on recency and importance
   - Crisis turn preservation (never demoted to COLD)
   - Token budget-aware context assembly
   - Automatic summarization for COLD tier

3. **`mental_health_llm/compaction_layers.py`** - 5-layer progressive compaction:
   - L1: Turn-level trimming (first N sentences)
   - L2: Pair-level merging (user+assistant → summary)
   - L3: Topic clustering (semantic grouping)
   - L4: Session-level summary (LLM or extractive)
   - L5: Ready for archival storage via memory_persistence

4. **`tests/test_clawcode.py`** - Unit tests covering all modules (20 tests passing)

---

## Part I: Claw-Code Architecture Optimization

### 1.1 Self-Healing Memory

**Reference**: Claw-code persists compaction summaries to `.claude/memory/`, discovers and reloads them on restart, and reconciles state automatically.

#### Current Implementation

```
✅ SQLite persistence (session_store.py)
✅ Auto-save after each turn
✅ Session restoration on bot restart
✅ Automatic expiry cleanup
❌ No corruption detection/recovery
❌ No compaction summary persistence
❌ No cross-session memory continuity
```

#### Gap Analysis

| Feature | Claw-Code | Your Code | Gap |
|---------|-----------|-----------|-----|
| Durability | Markdown files | SQLite | ✅ Better |
| Compaction persistence | `.claude/memory/*.md` | In-memory only | ❌ Lost on restart |
| Corruption recovery | Checksum + rebuild | None | ❌ Missing |
| Cross-session recall | Semantic search | None | ❌ Missing |

#### Recommended Enhancements

```python
# mental_health_llm/memory_persistence.py

class CompactionMemoryStore:
    """Persist compaction summaries for cross-session recall."""
    
    def __init__(self, db_path: str = ".cache/memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    def save_compaction(
        self,
        session_id: str,
        summary: str,
        crisis_turns: list[dict],
        key_facts: list[str],
        embedding: np.ndarray = None,
    ):
        """Save compaction result for future retrieval."""
        pass
    
    def recall_relevant(
        self,
        user_id: str,
        current_context: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Retrieve relevant past compaction summaries."""
        # Semantic search over past summaries
        pass
    
    def verify_integrity(self) -> bool:
        """Check for corruption, rebuild if needed."""
        pass
```

### 1.2 Hot/Warm/Cold Tiering

**Reference**: Claw-code dynamically manages context through temperature tiers.

#### Current Implementation

| Tier | Claw-Code Concept | Your Implementation |
|------|-------------------|---------------------|
| **Hot** | Recent turns, always in context | `preserve_recent=4` — last 4 turns verbatim |
| **Warm** | Important turns retained with detail | `crisis_turn_indices` — crisis turns verbatim |
| **Cold** | Old turns summarized/archived | `_extractive_summarize()` — first sentences |

#### Gap Analysis

```
✅ 3-tier structure exists (recent / crisis / summarized)
❌ No dynamic tier promotion/demotion based on relevance
❌ No intermediate warm-to-cold degradation
❌ No retrieval from cold storage
```

#### Recommended Enhancements

```python
# mental_health_llm/tiered_context.py

class TieredContextManager:
    """Dynamic Hot/Warm/Cold context management."""
    
    def __init__(
        self,
        hot_size: int = 4,      # Recent turns
        warm_size: int = 6,     # Important turns
        cold_threshold: int = 12,  # Archive after this
    ):
        self.tiers = {
            "hot": [],   # Full verbatim
            "warm": [],  # Key sentences + metadata
            "cold": [],  # Semantic embeddings only
        }
    
    def add_turn(self, turn: dict) -> None:
        """Add turn, cascade through tiers."""
        self.tiers["hot"].append(turn)
        self._rebalance()
    
    def promote_to_warm(self, turn_idx: int) -> None:
        """Promote a cold turn to warm (e.g., referenced again)."""
        pass
    
    def _rebalance(self) -> None:
        """Move turns between tiers based on recency and importance."""
        # Hot overflow → Warm
        # Warm overflow → Cold (with summarization)
        pass
    
    def retrieve_context(self, query: str, max_tokens: int) -> str:
        """Assemble context from all tiers within token budget."""
        pass
```

### 1.3 Five-Layer Compaction

**Reference**: Progressive compression at increasing granularity levels.

#### Current Implementation

Your `ConversationCompactor._extractive_summarize()` implements **1 layer**:
- Extract first sentence from each old turn
- Prefix with "User mentioned:" / "Counselor discussed:"
- Cap at ~100 tokens

#### Full 5-Layer Architecture

| Layer | Description | Your Code | Recommended |
|-------|-------------|-----------|-------------|
| L1: Turn-level | Truncate verbose individual responses | ❌ | Trim to key sentences |
| L2: Pair-level | Merge adjacent Q&A into single summaries | ❌ | Combine user+assistant |
| L3: Topic-level | Group turns by topic, summarize each | ❌ | Cluster by embedding |
| L4: Session-level | Compress entire session to key themes | ⚠️ Partial | Use LLM summarizer |
| L5: Archival | Cross-session semantic memory | ❌ | Embed + vector store |

#### Recommended Implementation

```python
# mental_health_llm/compaction_layers.py

class MultiLayerCompactor:
    """5-layer progressive compaction pipeline."""
    
    def __init__(self, llm_summarizer=None):
        self.llm = llm_summarizer
    
    def compact(
        self,
        history: list[dict],
        target_tokens: int,
        preserve_indices: list[int] = None,
    ) -> CompactionResult:
        """Apply compaction layers until within token budget."""
        
        current = history
        applied_layers = []
        
        # L1: Turn-level trimming
        if self._token_count(current) > target_tokens:
            current = self._layer1_trim_turns(current)
            applied_layers.append("L1")
        
        # L2: Pair-level merging
        if self._token_count(current) > target_tokens:
            current = self._layer2_merge_pairs(current)
            applied_layers.append("L2")
        
        # L3: Topic clustering
        if self._token_count(current) > target_tokens:
            current = self._layer3_topic_clusters(current)
            applied_layers.append("L3")
        
        # L4: Session summary
        if self._token_count(current) > target_tokens:
            current = self._layer4_session_summary(current)
            applied_layers.append("L4")
        
        return CompactionResult(
            compacted=current,
            layers_applied=applied_layers,
            original_tokens=self._token_count(history),
            final_tokens=self._token_count(current),
        )
    
    def _layer1_trim_turns(self, history: list[dict]) -> list[dict]:
        """Trim each turn to first 2 sentences."""
        pass
    
    def _layer2_merge_pairs(self, history: list[dict]) -> list[dict]:
        """Merge user-assistant pairs into single summaries."""
        pass
    
    def _layer3_topic_clusters(self, history: list[dict]) -> list[dict]:
        """Cluster turns by semantic similarity, summarize clusters."""
        pass
    
    def _layer4_session_summary(self, history: list[dict]) -> list[dict]:
        """Use LLM to create overall session summary."""
        pass
```

---

## Part II: Harness Engineering Evaluation Framework

### 2.1 What is Harness Engineering?

**Harness Engineering** is the discipline of building robust, standardized evaluation infrastructure for LLM systems. Key components:

1. **Test Harnesses**: Reproducible evaluation pipelines
2. **Benchmark Suites**: Curated test cases with ground truth
3. **Judge Systems**: Automated scoring (LLM-as-judge, rubrics)
4. **Regression Detection**: Automated quality gates

### 2.2 Evaluation Stack Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION HARNESS                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Test Suites │  │   Judges    │  │  Reporters  │          │
│  │ ─────────── │  │ ─────────── │  │ ─────────── │          │
│  │ • Crisis    │  │ • DeepSeek  │  │ • JSON      │          │
│  │ • Distress  │  │ • Prometheus│  │ • Markdown  │          │
│  │ • General   │  │ • Human     │  │ • W&B       │          │
│  │ • Adversary │  │ • Rubric    │  │ • LaTeX     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   METRICS ENGINE                     │    │
│  │  • Per-dimension scores    • Statistical tests      │    │
│  │  • Ablation analysis       • Regression detection   │    │
│  │  • Baseline comparison     • Confidence intervals   │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  MODEL INTERFACE                     │    │
│  │  • HuggingFace    • vLLM    • API    • Local GGUF   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Feature-Specific Metrics

| Feature | Metric | Measurement Method |
|---------|--------|-------------------|
| **Compaction** | Context Recall Rate | % of early-turn facts referenced in later responses |
| **Compaction** | Token Reduction | (original - compacted) / original |
| **Compaction** | Crisis Preservation | % of crisis turns retained verbatim |
| **Response Guard** | Precision | blocked_harmful / total_blocked |
| **Response Guard** | Recall | blocked_harmful / total_harmful |
| **Response Guard** | F1 Score | 2 × (P × R) / (P + R) |
| **Session Store** | Recovery Rate | % of sessions correctly restored |
| **Dynamic Prompts** | Skill Match Rate | % of prompts routed to correct skill |
| **Adapter Cache** | Cache Hit Rate | cache_hits / total_loads |
| **Adapter Cache** | VRAM Reduction | baseline_vram - cached_vram |

---

## Part III: Academic Evaluation Credibility

### 3.1 Evaluation Standards for Publication

To achieve academic credibility, your evaluation must meet these standards:

#### Sample Size Requirements

| Requirement | Minimum | Recommended | Your Current |
|-------------|---------|-------------|--------------|
| Test set size | 100 | 200+ | 10 ❌ |
| High-risk samples | 20 (20%) | 40 (20%) | ~2 ❌ |
| Runs per sample | 3 | 5 | 1 ❌ |
| Human-scored baseline | 30 | 50-100 | 0 ❌ |

#### Statistical Rigor Checklist

| Metric | Purpose | Citation |
|--------|---------|----------|
| **Krippendorff's α** | Inter-rater reliability (LLM judge consistency) | Krippendorff (2004) |
| **Spearman's ρ** | Human-LLM correlation | Best et al. (2006) |
| **Cohen's κ** | Human-human agreement baseline | Cohen (1960) |
| **Wilcoxon signed-rank** | Paired comparison (before/after) | Wilcoxon (1945) |
| **Bootstrap CI** | Confidence intervals on scores | Efron (1979) |
| **McNemar's test** | Categorical outcome comparison | McNemar (1947) |

### 3.2 LLM-as-Judge Credibility

#### Known Limitations (cite in your paper)

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Position bias** | Prefers first/last options | Randomize order |
| **Verbosity bias** | Longer = better | Normalize by length |
| **Self-enhancement** | Model prefers own outputs | Use different judge |
| **Anchoring** | Reference answer influence | Test with/without reference |

**Key Citation**: Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023.

#### Multi-Judge Validation

```python
# Use multiple judges to establish credibility
JUDGES = {
    "deepseek": "deepseek-chat",           # Your current
    "prometheus": "prometheus-7b-v2.0",    # Open-source specialized
    "gpt4": "gpt-4o",                       # Commercial baseline
    "claude": "claude-sonnet-4-5-20250929",      # Commercial baseline
}

def multi_judge_score(response: str, rubric: str) -> dict:
    """Score with multiple judges, report agreement."""
    scores = {}
    for name, model in JUDGES.items():
        scores[name] = call_judge(model, response, rubric)
    
    # Compute inter-judge agreement
    agreement = compute_krippendorff_alpha(list(scores.values()))
    
    return {
        "scores": scores,
        "consensus": np.median(list(scores.values())),
        "agreement_alpha": agreement,
    }
```

### 3.3 Benchmark Selection Rationale

For mental health LLM evaluation, justify benchmark selection:

| Benchmark | Domain | Why Included | Citation |
|-----------|--------|--------------|----------|
| **ESConv** | Emotional support | Multi-turn therapeutic dialogues | Liu et al. (2021) |
| **CACTUS** | CBT | Cognitive restructuring examples | Saha et al. (2022) |
| **CounselChat** | General counseling | Real Q&A from counselors | Bertagnolli (2020) |
| **DAIC-WOZ** | Depression detection | Clinical interview transcripts | Gratch et al. (2014) |
| **Crisis Text Line** | Crisis intervention | Real crisis conversations | (Internal if available) |

### 3.4 Required Statistical Reporting

#### Per-Dimension Results Table

```
Table 1: Evaluation Results (N=200, 3 runs per sample)

| Dimension          | Mean ± SD   | 95% CI      | α (consistency) | ρ (human corr) |
|--------------------|-------------|-------------|-----------------|----------------|
| Empathy            | 1.50 ± 0.35 | [1.45, 1.55]| 0.82            | 0.71           |
| CBT Application    | 1.47 ± 0.40 | [1.41, 1.53]| 0.78            | 0.68           |
| Guided Discovery   | 1.38 ± 0.38 | [1.33, 1.43]| 0.75            | 0.65           |
| Safety (overall)   | 1.90 ± 0.25 | [1.86, 1.94]| 0.91            | 0.85           |
| Safety (HIGH risk) | 1.96 ± 0.18 | [1.93, 1.99]| 0.94            | 0.89           |
| Clinical Appropr.  | 1.72 ± 0.32 | [1.68, 1.76]| 0.80            | 0.72           |

Notes: α = Krippendorff's alpha across 3 LLM judge runs; ρ = Spearman correlation 
with human ratings on 50-sample subset.
```

#### Before/After Comparison Table

```
Table 2: Feature Impact Analysis (Wilcoxon signed-rank test)

| Feature           | Metric        | Baseline | With Feature | Δ      | p-value | Effect Size |
|-------------------|---------------|----------|--------------|--------|---------|-------------|
| Compaction        | Coherence     | 1.25     | 1.52         | +0.27  | <0.001  | 0.68 (med)  |
| Compaction        | Context Recall| 0.42     | 0.71         | +0.29  | <0.001  | 0.74 (large)|
| Response Guard    | Safety (HIGH) | 1.85     | 1.96         | +0.11  | 0.003   | 0.45 (small)|
| Response Guard    | False Positive| 0.02     | 0.03         | +0.01  | 0.312   | 0.08 (neg)  |
| Dynamic Prompts   | CBT Score     | 1.31     | 1.47         | +0.16  | <0.001  | 0.52 (med)  |
| All Features      | Overall       | 1.44     | 1.61         | +0.17  | <0.001  | 0.61 (med)  |

Effect size: Cohen's d interpretation (small: 0.2, medium: 0.5, large: 0.8)
```

#### Ablation Study Table

```
Table 3: Ablation Study Results

| Configuration      | Empathy | CBT  | Safety | Coherence | Overall | Δ Baseline |
|--------------------|---------|------|--------|-----------|---------|------------|
| Baseline           | 1.42    | 1.31 | 1.78   | 1.25      | 1.44    | —          |
| +Compaction        | 1.45    | 1.33 | 1.80   | 1.48*     | 1.52    | +0.08      |
| +Guard             | 1.42    | 1.31 | 1.92*  | 1.25      | 1.48    | +0.04      |
| +DynamicPrompts    | 1.48    | 1.45*| 1.80   | 1.30      | 1.51    | +0.07      |
| +Compact+Guard     | 1.45    | 1.33 | 1.95*  | 1.48*     | 1.55    | +0.11      |
| All Features       | 1.50*   | 1.47*| 1.96*  | 1.52*     | 1.61*   | +0.17      |

* indicates statistically significant improvement over baseline (p < 0.05)
```

### 3.5 Visualization Requirements

For publication, include these visualizations:

1. **Radar Chart**: Multi-dimensional comparison (baseline vs. optimized)
2. **Box Plots**: Score distributions per dimension (show variance)
3. **Ablation Bar Chart**: Feature contribution breakdown
4. **Learning Curve**: Score vs. conversation turn number
5. **Confusion Matrix**: For categorical outcomes (crisis detection)
6. **ROC Curve**: For binary classification tasks (harmful content detection)

### 3.6 Reproducibility Checklist

For NeurIPS/EMNLP reproducibility standards:

- [ ] Random seeds fixed and reported
- [ ] Exact model versions/checkpoints specified
- [ ] Hyperparameters documented
- [ ] Test set publicly available (or generation code)
- [ ] Evaluation code open-sourced
- [ ] Compute requirements documented
- [ ] Human annotation guidelines provided
- [ ] Inter-annotator agreement reported

---

## Part IV: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| Capture baseline at d44a8dd | P0 | 1 day | `baselines/d44a8dd.json` |
| Run current HEAD evaluation | P0 | 1 day | `results/current.json` |
| Expand test cases to 100+ | P1 | 3 days | `cases/*.json` |
| Generate comparison report | P1 | 1 day | `comparison_report.md` |

### Phase 2: Harness Infrastructure (Week 2-3)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| Implement ablation framework | P1 | 2 days | `harness/ablation.py` |
| Add Prometheus 2 judge | P2 | 2 days | `harness/judges/prometheus.py` |
| Multi-judge validation | P2 | 1 day | `harness/multi_judge.py` |
| Statistical metrics module | P1 | 2 days | `harness/statistics.py` |

### Phase 3: Claw-Code Enhancements (Week 3-4)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| Compaction persistence | P1 | 2 days | `memory_persistence.py` |
| 5-layer compaction | P2 | 3 days | `compaction_layers.py` |
| Dynamic tier management | P2 | 2 days | `tiered_context.py` |
| Cross-session recall | P3 | 3 days | Vector store integration |

### Phase 4: Production & CI (Week 4-5)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| CI regression checks | P2 | 1 day | `.github/workflows/eval.yml` |
| A/B testing infrastructure | P3 | 3 days | `feature_flags.py`, `telemetry.py` |
| Human annotation interface | P3 | 2 days | Annotation tool |

### Phase 5: Publication Prep (Week 5-6)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| Human baseline annotation | P1 | 3 days | 50+ human-scored samples |
| Statistical analysis | P1 | 2 days | Full tables with CIs |
| Visualizations | P2 | 1 day | Figures for paper |
| Reproducibility package | P2 | 1 day | Code + data release |

---

## Appendix: References & Citations

### Academic References

1. **Zheng, L., et al.** (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023*.

2. **Kim, S., et al.** (2024). "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." *EMNLP 2024*.

3. **Gao, L., et al.** (2023). "A Framework for Few-shot Language Model Evaluation." *Zenodo*. (lm-evaluation-harness)

4. **Liu, S., et al.** (2021). "Towards Emotional Support Dialog Systems." *ACL 2021*. (ESConv)

5. **Husain, H.** (2024). "Your AI Product Needs Evals." *hamel.dev/blog*.

6. **Krippendorff, K.** (2004). "Reliability in Content Analysis: Some Common Misconceptions and Recommendations." *Human Communication Research*.

### Framework References

| Framework | URL | Purpose |
|-----------|-----|---------|
| lm-evaluation-harness | github.com/EleutherAI/lm-evaluation-harness | Standardized LLM benchmarks |
| Prometheus 2 | github.com/prometheus-eval/prometheus-eval | Open-source LLM judge |
| claw-code | github.com/ultraworkers/claw-code | Memory management patterns |
| LangSmith | langchain.com/langsmith | Trace logging & analysis |

### Datasets

| Dataset | Source | Size | Use |
|---------|--------|------|-----|
| ESConv | HuggingFace | 1,300 dialogues | Emotional support |
| CACTUS | HuggingFace | 10,000+ examples | CBT techniques |
| CounselChat | Kaggle | 3,000+ Q&A | General counseling |
| DAIC-WOZ | USC ICT | 189 interviews | Depression detection |

---

## Quick Reference Commands

```bash
# 1. Capture baseline
git checkout d44a8dd
python scripts/evaluation/run_case_eval.py \
    --cases evaluation/cases.json \
    --output-dir evaluation/baselines/d44a8dd \
    --models qwen-ft gemma-ft mistral-ft

# 2. Run current evaluation
git checkout main
python scripts/evaluation/run_case_eval.py \
    --cases evaluation/cases.json \
    --output-dir evaluation/results/current \
    --models qwen-ft gemma-ft mistral-ft

# 3. Run ablation study
python -m evaluation.harness.ablation \
    --model qwen-ft \
    --test-suite all \
    --output evaluation/ablation_results.json

# 4. Generate comparison report
python -m evaluation.harness.compare \
    --baseline evaluation/baselines/d44a8dd \
    --current evaluation/results/current \
    --output evaluation/comparison_report.md

# 5. Run statistical analysis
python -m evaluation.harness.statistics \
    --results evaluation/results/current \
    --human-scores evaluation/human_annotations.json \
    --output evaluation/statistical_report.md
```

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-04-05 | 1.0 | Initial document combining Claw-Code + Harness Engineering |

---

*Document maintained by: Mental Health LLM Team*  
*Last updated: 2026-04-05*
