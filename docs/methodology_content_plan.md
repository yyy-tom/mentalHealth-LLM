# Content Plan: Methodology Section (Term 2 FYP Report)

**Course:** SZL2504 – LLM for Mental Healthcare
**Report:** Final Year Project Report (2025-26 Term 2)

---

## Section 2.1 — Fine-Tuning

**Purpose:** Explain the model training approach, covering base model selection, datasets, LoRA strategy, and infrastructure.

### What to reuse from Sem1 (§3.1–3.5)
- Dataset sources (7 datasets, 623K samples, Cactus 73%)
- Rationale for intentional class imbalance
- LoRA theory (rank, alpha, target modules)
- 4-bit NF4 quantization rationale
- Distributed training on 8× RTX 2080 Ti
- Reliability findings (OOM mitigations, NCCL timeouts)

### New content to add for Sem2
- **Expanded model set**: 3 base models fine-tuned:
  - Qwen2.5-7B-Instruct (primary)
  - Gemma-2-9B-it
  - Mistral-7B-Instruct-v0.3
- **Skill-specific LoRA adapters** (second-stage training on top of full fine-tune):
  - 6 adapters: crisis-intervention, cbt-therapy, psychoeducation, empathetic-listening, professional-counseling, general-support
  - Each: LoRA r=32, α=64, ~20K skill-specific samples, 1 epoch, max_steps=2000
  - Adapter weights saved separately, loaded dynamically at inference
- **Key hyperparameter table:**

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 32 (standard), 64 (high-VRAM) |
| LoRA alpha | 64 (= 2×r) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Dropout | 0.05 |
| Learning rate | 1e-4 (single GPU), 2e-5 (full FT) |
| Effective batch size | 16–128 depending on hardware |
| LR scheduler | Cosine with 3% warmup |
| Quantization | 4-bit NF4, double quant, bfloat16 compute |

---

## Section 2.2 — Prompt Engineering

**Purpose:** Explain how the system dynamically selects and constructs prompts at inference time using skill routing.

### 2.2.1 Skill Taxonomy

6 therapeutic skills defined with priority levels:

| Priority | Skill |
|----------|-------|
| 100 | crisis-intervention |
| 50 | cbt-therapy |
| 40 | psychoeducation |
| 30 | empathetic-listening |
| 20 | professional-counseling |
| 0 (default) | general-support |

- Each skill has a dedicated system prompt
- Crisis skill embeds HK-specific hotlines: Samaritan Befrienders (2389 2222), Suicide Prevention Services (2382 0000)

### 2.2.2 Dual-Backend Skill Router

Two routing backends with automatic fallback:

**Embedding Router (primary):**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)
- Pre-computed skill centroids (`skill_centroids.npz`)
- History blending: 80% current message + 20% weighted recent history (3-turn window, exponential decay)
- Skill selected by cosine similarity argmax across centroids

**Keyword Router (fallback — no ML dependencies):**
- Regex + substring matching per skill
- Score = keyword hits × 0.3 + pattern hits × 0.4 (capped at 1.0)
- Returns default skill if best score < 0.3

**Auto mode:** tries embedding router first, silently falls back to keyword if unavailable.

### 2.2.3 Crisis Gate (Pre-classifier)

Runs **before** general skill routing — dedicated safety layer:

- **Keyword signal:** 22+ crisis keywords (e.g., "want to die", "kill myself", "self-harm") + 7 regex patterns
- **Embedding signal:** cosine similarity ≥ 0.45 to crisis centroid
- **Adaptive threshold:** lowers to 0.38 if conversation shows upward crisis trend (avg_increase > 0.02)
- Either signal triggers immediate return of "crisis-intervention" with confidence 0.95

### 2.2.4 Prompt Composition

Dynamic assembly order (`TherapyPromptBuilder`):

1. Base skill system prompt
2. Crisis context block (injected only when crisis gate fires; 3 levels: high/medium/low)
3. Session summary (compacted conversation history)
4. Locale block (region-specific resources: HK/US/UK/CA/AU)

---

## Section 2.3 — Telegram Interface

**Purpose:** Describe the user-facing chatbot interface, multi-model selection, and session management.

### 2.3.1 Bot Architecture

- Built with `python-telegram-bot`
- Supports **6 model variants** (3 fine-tuned + 3 base), switchable per user
- Per-user state: model selection, adapter toggle, streaming toggle, language preference

| Model Key | Description |
|-----------|-------------|
| qwen-ft | Qwen 2.5 7B (fine-tuned) |
| qwen-base | Qwen 2.5 7B (base) |
| gemma-ft | Gemma 2 9B (fine-tuned) |
| gemma-base | Gemma 2 9B (base) |
| mistral-ft | Mistral 7B (fine-tuned) |
| mistral-base | Mistral 7B (base) |

### 2.3.2 Commands Overview

| Command | Function |
|---------|----------|
| `/start` | Initialize session, show welcome |
| `/model` | Switch model via inline keyboard |
| `/clear` | Reset conversation history |
| `/language` | Set ASR language (auto/en/yue/zh) |
| `/adapters` | Toggle skill LoRA adapters on/off |
| `/streaming` | Toggle streaming response mode |
| `/score` | View per-turn quality scores |
| `/stats` | Session analytics |
| `/memory` | View cross-session memory |
| `/harness` | Access evaluation harness metrics |
| `/trace` | Show pipeline execution trace |

### 2.3.3 Session Persistence

- SQLite-backed (`SQLiteSessionStore`, `data/sessions.db`)
- In-memory fallback if SQLite unavailable
- **Context tiering:** hot layer (last 4 turns verbatim) + cold layer (older turns compacted via `EnhancedContextManager`)
- Cross-session memory stored separately via `memory_persistence.py`

### 2.3.4 LoRA Adapter Switching

- Dynamic per-turn adapter loading via PEFT
- Skill router selects skill → `AdapterCache` loads/caches appropriate LoRA weights
- Per-user toggle (enabled by default for fine-tuned models, disabled for base models)

### 2.3.5 Streaming Mode

- Token-by-token streaming via async generator (`mental_health_llm/streaming.py`)
- Delivered as progressive Telegram message edits
- User-configurable per session, disabled by default

---

## Section 2.4 — Voice Interface

**Purpose:** Describe the speech-to-text pipeline enabling voice message input.

### 2.4.1 ASR Model

- **Faster Whisper large-v3** (default, configurable via `--whisper_model`)
- Loaded on **last available GPU** (float16) to avoid conflict with LLM GPU assignment
- CPU fallback: int8 quantized

### 2.4.2 Voice Message Processing Pipeline

```
User sends voice OGG (≤120 sec)
        ↓
Telegram bot downloads file
        ↓
asyncio.to_thread(transcribe_audio, path, language)
        ↓
WhisperModel.transcribe(beam_size=5) → segments joined
        ↓
Empty result? → re-prompt user
        ↓
Transcript → Skill Router → Response generation
        ↓
Text response sent back to user
```

### 2.4.3 Language Support

| Code | Language |
|------|----------|
| None (default) | Auto-detect |
| en | English |
| yue | Cantonese |
| zh | Mandarin Chinese |

- Per-user preference stored in session, set via `/language` command
- **Note:** TTS (text-to-speech) is NOT implemented; output is text-only

---

## Section 2.5 — Automatic Grading System

**Purpose:** Explain the LLM-as-Judge evaluation harness that automatically scores model responses.

### 2.5.1 Motivation

Manual scoring (Sem1: 10 cases × 8 turns, rated by hand) is not scalable for 6 model variants and multiple ablation configurations. The automatic grading system enables:
- Rapid A/B testing between model variants
- Ablation studies across feature flags
- Regression detection relative to a saved baseline

### 2.5.2 Evaluation Dimensions

**Turn-level (0–2 each → 0–8 total per response):**

| Dimension | Description |
|-----------|-------------|
| Empathy | Emotion acknowledgment and validation |
| Guided Discovery | Socratic questioning depth |
| CBT Techniques | Max score across 4 sub-techniques (Cognitive Reconstruction, Behavioural Activation, Positive Encouragement, Psychoeducation) |
| Safety Awareness | Crisis detection and resource provision |

**Multi-turn coherence (0–2 each, assessed over full conversation):**

| Dimension | Description |
|-----------|-------------|
| Memory & Continuity | References and builds on prior context |
| Therapeutic Arc | Session has direction and progression |
| Repetition Avoidance | Varied techniques, no verbatim repetition |

> These dimensions extend and align with the manual CTRS-derived criteria in §3.1.1.

### 2.5.3 LLM Judge

- **Models supported:** GPT-4o, Claude Sonnet 4.5, DeepSeek V3, Gemini 2.5 Flash
- Temperature: 0 (deterministic scoring)
- Retry logic: 5 retries with exponential backoff (2s → 32s, max 60s)
- **Input:** conversation history + user turn + model response
- **Output:** JSON with `risk_level`, dimension scores with evidence strings, `overall_comment`
- Rubric stored at `evaluation/llm_judge_prompt.md`

### 2.5.4 Statistical Analysis

- Bootstrap confidence intervals (1,000 samples, 95% confidence)
- **Significance tests:**
  - Paired data (equal n) → Wilcoxon signed-rank test
  - Unpaired data (unequal n) → Mann-Whitney U test
  - Fallback → Permutation test (1,000 iterations)
- Effect size: Cohen's d (pooled std dev)
- Multiple comparison correction: Bonferroni or FDR-BH

### 2.5.5 Baseline Comparison

- Baseline stored with: metrics, raw results, model name, feature set, timestamp, commit hash
- Dimension-wise report: `baseline_mean` vs. `current_mean`, p-value, Cohen's d, % change
- **Merge gate:** significant regression in `safety_awareness` or `clinical_appropriateness` → automatically blocks approval

### 2.5.6 Ablation Study

- Features tested: `compaction`, `response_guard`, `dynamic_prompts`
- Methodology: run full evaluation → disable one feature → compare delta per dimension
- Recommends disabling only if Cohen's d < -0.2 AND statistically significant

---

## Reuse Summary from Sem1

| Content | Sem1 Location | Action for Sem2 |
|---------|--------------|-----------------|
| Dataset sources & rationale | §3.2.1 | Copy + update: add crisis dataset, note 3-model expansion |
| LoRA theory + quantization rationale | §3.2–3.3 | Copy; update param values for 7B models |
| Training reliability findings | §3.5 | Copy directly |
| Evaluation criteria (CTRS adaptation) | §3.6.1 | Do NOT copy — Sem2 §3.1.1 is the upgraded version (adds Safety Handling + benchmark tables) |
| Test cases | §4.2 | Do NOT copy — Sem2 §3.1.4 adds 3 high-risk cases |
| Introduction | §1 | Copy; add 1–2 sentences about Telegram/Voice/Grading additions |
| Background (Depression, CBT, Existing LLMs) | §2.1–2.3 | Copy entirely |

---

## Critical Files for Reference

| Section | Key Files |
|---------|-----------|
| 2.1 Fine-Tuning | `scripts/train_qwen_counsel.py`, `scripts/train_skill_lora.py`, `configs/*.json` |
| 2.2 Prompt Engineering | `mental_health_llm/skill_router.py`, `mental_health_llm/prompt_builder.py`, `mental_health_llm/crisis_gate.py`, `mental_health_llm/embedding_router.py`, `mental_health_llm/keyword_router.py`, `mental_health_llm/skills_config.json` |
| 2.3 Telegram Interface | `scripts/telegram_bot.py`, `mental_health_llm/session_store.py`, `mental_health_llm/adapter_cache.py` |
| 2.4 Voice Interface | `scripts/telegram_bot.py` (functions: `transcribe_audio`, `handle_voice`) |
| 2.5 Grading System | `evaluation/harness/runner.py`, `evaluation/harness/metrics.py`, `evaluation/harness/ablation.py`, `evaluation/llm_judge_prompt.md` |
