# Implementation Plan: Improve `mentalHealth-LLM` (7B-Optimized)

## Objective

Enhance the current counseling stack with **retrieval grounding, lightweight orchestration, and evaluation-driven quality controls** — scoped for a 7B parameter model deployment.

## Scope Constraints

| Constraint | Implication |
|------------|-------------|
| Model size: ≤7B | Limited parametric knowledge → retrieval compensates |
| Project scale: Demo/research | No enterprise infra → skip complex connectors/MCP |
| Compute: Single GPU | No multi-model orchestration overhead |

## Current Baseline (Already Implemented)

| Component | Status | Files |
|-----------|--------|-------|
| Skill routing + crisis gating | ✅ Done | `skill_router.py`, `crisis_gate.py` |
| Tiered context (Hot/Warm/Cold) | ✅ Done | `tiered_context.py` |
| 5-layer compaction | ✅ Done | `compaction_layers.py` |
| Self-healing memory | ✅ Done | `memory_persistence.py` |
| Context integration | ✅ Done | `context_integration.py` |
| Response safety guard | ✅ Done | `response_guard.py` |
| Evaluation harness | ✅ Done | `evaluation/harness/*` |
| Telegram bot runtime | ✅ Done | `scripts/telegram_bot.py` |

**Key gaps:** Simple orchestration, retrieval grounding, practical tools, groundedness metrics.

## OSS References (Selective)

| Reference | What We Take | What We Skip |
|-----------|--------------|--------------|
| `microsoft/agent-framework` | State machine pattern | Full DAG, checkpointing |
| `onyx-dot-app/onyx` | Hybrid search concept | Connectors, multi-tenant |
| `block/goose` | Simple tool pattern | MCP bridge, full runtime |

**Excluded entirely:**
- `RAG-Anything` multimodal — overkill for text-only counseling
- `hermes-agent` learning loop — requires high query volume

## Target Architecture (Simplified)

```
┌─────────────────────────────────────────────────────────────┐
│                     TELEGRAM BOT                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              ORCHESTRATION (State Machine)                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Triage  │→ │Retrieve │→ │Generate │→ │ Guard   │→ Persist│
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│ Retrieval │  │  Tools    │  │  Harness  │
│  (Basic)  │  │ (2-3 only)│  │ (Extended)│
└───────────┘  └───────────┘  └───────────┘
```

---

## Phase A — Lightweight Orchestration

**Goal:** Deterministic 5-node state machine with trace logging.

**Build:**
- `mental_health_llm/orchestration/state.py` — Turn state dataclass
- `mental_health_llm/orchestration/pipeline.py` — Linear pipeline executor

**Nodes:**
1. **Triage** — Crisis detection, skill routing
2. **Retrieve** — KB lookup (if enabled)
3. **Generate** — Model inference with context
4. **Guard** — Response safety check
5. **Persist** — Save to memory/session

**Integration:**
- Replace ad-hoc flow in `telegram_bot.py` with pipeline call
- Add trace logging (node → duration → outcome)

**Outcome:**
- Clear execution path for debugging
- Per-turn trace metadata for analysis

**Effort:** ~2-3 days

---

## Phase B — Basic Retrieval

**Goal:** Ground responses with curated psychoeducation content.

**Build:**
- `mental_health_llm/retrieval/index.py` — Embed & store docs
- `mental_health_llm/retrieval/search.py` — Cosine similarity search
- `mental_health_llm/retrieval/citation.py` — Format citations

**Knowledge Base (curated, not crawled):**
```
data/kb/
├── crisis_resources.json      # Hotlines by region
├── coping_strategies.md       # Evidence-based techniques
├── psychoeducation/           # Anxiety, depression, etc.
│   ├── anxiety_basics.md
│   ├── depression_basics.md
│   └── sleep_hygiene.md
└── grounding_exercises.md     # 5-4-3-2-1, breathing, etc.
```

**Integration:**
- Inject top-3 chunks into context when query matches KB topics
- Add `[Source: filename]` citations to responses

**Outcome:**
- Factual grounding for psychoeducation queries
- Reduced hallucination of techniques/resources

**Effort:** ~3-4 days

---

## Phase C — Practical Tools (Minimal)

**Goal:** 2-3 hardcoded tools, no complex registry.

**Build:**
- `mental_health_llm/tools/crisis_lookup.py` — Return hotlines by locale
- `mental_health_llm/tools/journal_export.py` — Export session to markdown

**Tool Interface (simple):**
```python
def crisis_lookup(locale: str = "HK") -> str:
    """Return crisis hotlines for the given locale."""
    resources = {
        "HK": "Samaritans HK: 2389 2222\nSuicide Prevention: 2382 0000",
        "US": "988 Suicide & Crisis Lifeline: 988",
        "UK": "Samaritans: 116 123",
    }
    return resources.get(locale.upper(), resources["HK"])
```

**Integration:**
- Call from orchestration when crisis detected
- No approval workflow (always safe to provide hotlines)

**Outcome:**
- Actionable crisis resources without hallucination
- User data export capability

**Effort:** ~1-2 days

---

## Phase D — Extended Evaluation Metrics

**Goal:** Add groundedness and crisis-specific metrics to existing harness.

**Extend `evaluation/harness/`:**
- `metrics.py` — Add `groundedness_score()`, `citation_rate()`
- `baseline.py` — Track crisis recall/precision

**New Metrics:**

| Metric | Definition | Target |
|--------|------------|--------|
| Groundedness | % responses with KB citation when applicable | >60% |
| Citation accuracy | Citations actually support claims | >90% |
| Crisis recall | % crisis cases correctly detected | >95% |
| Crisis precision | % crisis alerts that are true crises | >80% |
| False alarm rate | Non-crisis flagged as crisis | <10% |

**Integration:**
- Run on 50+ test cases (mix of crisis/non-crisis/psychoed)
- Add to ablation study (with/without retrieval)

**Outcome:**
- Quantified improvement from retrieval
- Safety regression gates

**Effort:** ~2 days

---

## Implementation Priority

| Priority | Phase | Effort | Impact |
|----------|-------|--------|--------|
| 1 | Phase A — Orchestration | 2-3 days | Debugging clarity, trace logging |
| 2 | Phase B — Retrieval | 3-4 days | Factual grounding, reduce hallucination |
| 3 | Phase D — Eval Metrics | 2 days | Measure improvement, safety gates |
| 4 | Phase C — Tools | 1-2 days | Crisis resources, data export |

**Total estimated effort:** ~10 days

---

## Explicitly Excluded

| Feature | Reason |
|---------|--------|
| Multimodal ingestion | Text-only counseling, 7B can't process images well |
| MCP/tool registry | Overkill for 2-3 tools |
| Learning loop / skill mining | Requires high volume, manual iteration sufficient |
| Connector framework | No external data sources needed |
| Full DAG orchestration | Linear pipeline sufficient |
| Checkpointing / recovery | Single-turn latency low enough to retry |

---

## Success Criteria

| Metric | Baseline (current) | Target |
|--------|-------------------|--------|
| Groundedness | ~0% (no retrieval) | >60% |
| Crisis recall | ~85% | >95% |
| Hallucination rate | Unknown | <15% |
| Response latency | ~2s | <3s (with retrieval) |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Retrieval adds latency | Cache embeddings, limit to top-3 |
| KB becomes stale | Version control KB, review quarterly |
| Over-reliance on citations | Guard still runs post-retrieval |
| Complexity creep | Feature flags, harness gating |

---

## Deliverables

- [ ] Orchestration pipeline with trace logging
- [ ] Retrieval index + search (50+ KB documents)
- [ ] Crisis resource lookup tool
- [ ] Journal export tool
- [ ] Extended harness metrics (groundedness, crisis recall)
- [ ] Baseline comparison report (before/after retrieval)

