#!/usr/bin/env python3
"""
Case-based assessment: replay multi-turn therapy conversations against
local models and score with DeepSeek judge.

Loads models one at a time, replays all cases from cases.json, scores
each turn, then outputs per-model JSON results and a comparison summary.

Usage:
    python scripts/evaluation/run_case_eval.py \
        --cases evaluation/cases.json \
        --output-dir evaluation/case_results \
        --models qwen-ft qwen-base gemma-ft gemma-base mistral-ft mistral-base \
        --judge deepseek \
        --resume
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

# Add script directory for sibling imports
_SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_SCRIPT_DIR))

from generate_responses import load_model, unload_model
from run_llm_judge import (
    CBT_SUBSCORES,
    DIMENSIONS,
    call_openai,
    call_judge_with_retry,
    load_judge_prompt,
)

# ---------------------------------------------------------------------------
# Model registry: 3 fine-tuned + 3 base
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "qwen-ft": "models/qwen2.5-7b-mental-health-fullft-a100",
    "qwen-base": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-ft": "models/gemma2-9b-mental-health-fullft-a100",
    "gemma-base": "google/gemma-2-9b-it",
    "mistral-ft": "models/mistral-7b-mental-health-fullft-a100",
    "mistral-base": "mistralai/Mistral-7B-Instruct-v0.3",
}

MODEL_DISPLAY_NAMES = {
    "qwen-ft": "Qwen FT",
    "qwen-base": "Qwen Base",
    "gemma-ft": "Gemma FT",
    "gemma-base": "Gemma Base",
    "mistral-ft": "Mistral FT",
    "mistral-base": "Mistral Base",
}

SYSTEM_PROMPT = "You are a compassionate and professional mental health counselor."

STOP_PATTERNS = [
    "\n\nQuestion:", "\n\nHuman:", "\n\nUser:",
    "[End]", "\n\nBased on", "\n\nThis response",
]


# ---------------------------------------------------------------------------
# Gemma-compatible generation
# ---------------------------------------------------------------------------

def _supports_system_role(tok) -> bool:
    """Check whether the tokenizer's chat template supports a system role."""
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "hi"}],
            tokenize=False,
        )
        return True
    except Exception:
        return False


def generate_response_multiturn(
    model,
    tokenizer,
    user_message: str,
    system_prompt: str,
    history: list[tuple[str, str]],
    use_system_role: bool = True,
    seed: int | None = None,
    max_new_tokens: int = 512,
) -> str:
    """Generate a response with multi-turn history, handling Gemma compatibility."""

    if use_system_role:
        messages = [{"role": "system", "content": system_prompt}]
    else:
        messages = []

    for i, (user_turn, counselor_turn) in enumerate(history):
        content = user_turn
        if not use_system_role and i == 0:
            content = f"{system_prompt}\n\n{user_turn}"
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": counselor_turn})

    user_content = user_message
    if not use_system_role and not history:
        user_content = f"{system_prompt}\n\n{user_message}"
    messages.append({"role": "user", "content": user_content})

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    for pattern in STOP_PATTERNS:
        if pattern in response:
            response = response.split(pattern)[0].strip()
            break

    return response


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

MAX_SCORE = 2
MAX_OVERALL_SCORE = 8
OVERALL_SCORE_FIELD = "overall_score_0_to_8"


def clamp_scores(scores: dict) -> dict:
    """Clamp all dimension scores to [0, MAX_SCORE] range.

    Safety net in case the judge returns out-of-range scores.
    """
    for dim in DIMENSIONS:
        dim_data = scores.get(dim)
        if isinstance(dim_data, dict) and isinstance(dim_data.get("score"), (int, float)):
            dim_data["score"] = min(max(dim_data["score"], 0), MAX_SCORE)

    cbt_data = scores.get("cbt", {})
    if isinstance(cbt_data, dict):
        subscores = cbt_data.get("subscores", {})
        if isinstance(subscores, dict):
            for sub in CBT_SUBSCORES:
                sub_data = subscores.get(sub)
                if isinstance(sub_data, dict) and isinstance(sub_data.get("score"), (int, float)):
                    sub_data["score"] = min(max(sub_data["score"], 0), MAX_SCORE)

    overall_data = scores.get(OVERALL_SCORE_FIELD)
    if isinstance(overall_data, dict) and isinstance(overall_data.get("score"), (int, float)):
        overall_data["score"] = min(max(overall_data["score"], 0), MAX_OVERALL_SCORE)

    return scores


def format_conversation_context(
    history: list[tuple[str, str]], situation: str = "",
) -> str:
    """Format history as 'User: ... / Counselor: ...' for the judge."""
    parts = []
    if situation:
        parts.append(f"Situation context: {situation}")
        parts.append("")
    if not history:
        if parts:
            parts.append("(No prior conversation)")
            return "\n".join(parts)
        return "(No prior context)"
    for user_msg, bot_msg in history:
        parts.append(f"User: {user_msg}")
        parts.append(f"Counselor: {bot_msg}")
    return "\n".join(parts)


def compute_dimension_averages(turns: list[dict]) -> dict:
    """Compute average scores per dimension across turns."""
    totals = {dim: [] for dim in DIMENSIONS}
    cbt_subscore_totals = {sub: [] for sub in CBT_SUBSCORES}
    overall_totals = []

    for turn in turns:
        scores = turn.get("scores", {})
        for dim in DIMENSIONS:
            dim_data = scores.get(dim, {})
            if isinstance(dim_data, dict):
                score = dim_data.get("score")
            else:
                score = dim_data
            if isinstance(score, (int, float)):
                totals[dim].append(score)

        turn_subscores = turn.get("cbt_subscores", {})
        if isinstance(turn_subscores, dict):
            for sub in CBT_SUBSCORES:
                sub_data = turn_subscores.get(sub, {})
                sub_score = sub_data.get("score") if isinstance(sub_data, dict) else sub_data
                if isinstance(sub_score, (int, float)):
                    cbt_subscore_totals[sub].append(sub_score)

        overall_data = turn.get(OVERALL_SCORE_FIELD, {})
        overall_score = overall_data.get("score") if isinstance(overall_data, dict) else overall_data
        if isinstance(overall_score, (int, float)):
            overall_totals.append(overall_score)
        else:
            dim_scores = []
            for dim in DIMENSIONS:
                dim_data = scores.get(dim, {})
                dim_score = dim_data.get("score") if isinstance(dim_data, dict) else dim_data
                if isinstance(dim_score, (int, float)):
                    dim_scores.append(dim_score)
            if len(dim_scores) == len(DIMENSIONS):
                overall_totals.append(sum(dim_scores))

    averages = {}
    for dim in DIMENSIONS:
        if totals[dim]:
            avg = round(sum(totals[dim]) / len(totals[dim]), 2)
            averages[dim] = avg
        else:
            averages[dim] = "N/A"

    averages["cbt_subscores"] = {}
    for sub in CBT_SUBSCORES:
        vals = cbt_subscore_totals[sub]
        if vals:
            averages["cbt_subscores"][sub] = round(sum(vals) / len(vals), 2)
        else:
            averages["cbt_subscores"][sub] = "N/A"

    if overall_totals:
        averages[OVERALL_SCORE_FIELD] = round(sum(overall_totals) / len(overall_totals), 2)
    else:
        averages[OVERALL_SCORE_FIELD] = "N/A"

    if isinstance(averages[OVERALL_SCORE_FIELD], (int, float)):
        averages["overall"] = round(averages[OVERALL_SCORE_FIELD] / len(DIMENSIONS), 2)
    else:
        averages["overall"] = "N/A"

    return averages


# ---------------------------------------------------------------------------
# Multi-turn coherence assessment
# ---------------------------------------------------------------------------

COHERENCE_PROMPT = """\
Review this full multi-turn therapeutic conversation and score coherence on a 0-2 scale.

1. Memory & Continuity (0-2)
0: Contradicts or ignores earlier exchanges
1: Acknowledges prior context but doesn't build on it
2: Actively references and builds on earlier exchanges

2. Therapeutic Arc (0-2)
0: No progression; repetitive or circular
1: Some progression but inconsistent direction
2: Clear therapeutic arc with appropriate pacing

3. Repetition Avoidance (0-2)
0: Repeats same phrases or techniques verbatim
1: Some variation but noticeable repetition
2: Each response adds new value; varied techniques

Situation: {SITUATION}

Full conversation:
{FULL_CONVERSATION}

Output ONLY valid JSON:
{{
  "memory": {{"score": "0-2", "evidence": "..."}},
  "therapeutic_arc": {{"score": "0-2", "evidence": "..."}},
  "repetition_avoidance": {{"score": "0-2", "evidence": "..."}},
  "overall_coherence_comment": "One sentence summary"
}}"""

COHERENCE_DIMS = ["memory", "therapeutic_arc", "repetition_avoidance"]


def score_coherence(call_fn, judge_system_prompt: str, situation: str, turns: list[dict]) -> dict:
    """Send full conversation to judge for holistic multi-turn coherence scoring."""
    conv_lines = []
    for turn in turns:
        conv_lines.append(f"User: {turn['user_input']}")
        conv_lines.append(f"Counselor: {turn['model_response']}")
    full_conv = "\n".join(conv_lines)

    user_msg = COHERENCE_PROMPT.format(
        SITUATION=situation,
        FULL_CONVERSATION=full_conv,
    )

    try:
        raw = call_judge_with_retry(call_fn, judge_system_prompt, user_msg)
        for dim in COHERENCE_DIMS:
            dim_data = raw.get(dim)
            if isinstance(dim_data, dict) and isinstance(dim_data.get("score"), (int, float)):
                dim_data["score"] = min(max(dim_data["score"], 0), MAX_SCORE)
        return raw
    except Exception as e:
        print(f"    ERROR scoring coherence: {e}")
        return {
            dim: {"score": "N/A", "evidence": f"Error: {e}"}
            for dim in COHERENCE_DIMS
        }


# ---------------------------------------------------------------------------
# Main per-model routine
# ---------------------------------------------------------------------------

def run_model(
    model_key: str,
    model_path: str,
    cases: list[dict],
    system_prompt: str,
    judge_system_prompt: str,
    judge_user_template: str,
    call_fn,
    judge_key: str,
    output_dir: Path,
):
    """Load a model, replay all cases, score, and save results."""
    output_path = output_dir / f"{model_key}.json"

    print(f"\n{'=' * 60}")
    print(f"Model: {model_key} ({model_path})")
    print(f"{'=' * 60}")

    # Load model
    model, tokenizer = load_model(model_path)
    model.training = False
    use_system_role = _supports_system_role(tokenizer)
    print(f"  System role support: {use_system_role}")

    case_results = []

    # Per-case progress saving for crash recovery
    progress_path = output_dir / f".{model_key}.progress.jsonl"
    completed_case_ids = set()
    if progress_path.exists():
        print(f"  Found progress file, loading completed cases...")
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    case_results.append(entry)
                    completed_case_ids.add(entry["case_id"])
        print(f"  Resumed {len(completed_case_ids)} completed cases")

    for case in cases:
        case_id = case["case_id"]
        title = case["title"]
        user_turns = case["user_turns"]
        situation = case.get("situation", "")

        if case_id in completed_case_ids:
            print(f"\n  Skipping {case_id}: {title} (already completed)")
            continue

        print(f"\n  Case {case_id}: {title} ({len(user_turns)} turns)")

        # Deterministic seed base for this case
        case_num = int(case_id.split("_")[1]) if "_" in case_id else 0
        seed_base = case_num * 1000

        history: list[tuple[str, str]] = []
        turns = []

        for turn_idx, user_msg in enumerate(tqdm(user_turns, desc=f"    {case_id}", leave=False)):
            turn_num = turn_idx + 1

            # Generate response
            t0 = time.time()
            turn_seed = seed_base + turn_idx
            try:
                response = generate_response_multiturn(
                    model, tokenizer, user_msg, system_prompt, history,
                    use_system_role=use_system_role, seed=turn_seed,
                )
            except Exception as e:
                print(f"    ERROR generating turn {turn_num}: {e}")
                response = f"[ERROR: {e}]"
            gen_time = round(time.time() - t0, 2)

            # Score this turn via judge
            context = format_conversation_context(history, situation=situation)
            judge_user_msg = judge_user_template.format(
                CONVERSATION_HISTORY=context,
                USER_INPUT=user_msg,
                MODEL_RESPONSE=response,
            )

            try:
                scores = call_judge_with_retry(call_fn, judge_system_prompt, judge_user_msg)
                scores = clamp_scores(scores)
            except Exception as e:
                print(f"    ERROR scoring turn {turn_num}: {e}")
                scores = {
                    dim: {"score": "N/A", "evidence": f"Error: {e}"}
                    for dim in DIMENSIONS
                }
                scores["cbt"] = {
                    "score": "N/A",
                    "evidence": f"Error: {e}",
                    "subscores": {
                        sub: {"score": "N/A", "evidence": f"Error: {e}"}
                        for sub in CBT_SUBSCORES
                    },
                }
                scores[OVERALL_SCORE_FIELD] = {"score": "N/A", "evidence": f"Error: {e}"}
                scores["clinical_appropriateness"] = {"score": "N/A", "evidence": f"Error: {e}"}
                scores["risk_level"] = "unknown"
                scores["overall_comment"] = f"Scoring failed: {e}"

            turn_data = {
                "turn": turn_num,
                "user_input": user_msg,
                "model_response": response,
                "generation_time_seconds": gen_time,
                "scores": {
                    dim: scores.get(dim, {"score": "N/A", "evidence": ""})
                    for dim in DIMENSIONS
                },
                "cbt_subscores": (
                    scores.get("cbt", {}).get("subscores", {})
                    if isinstance(scores.get("cbt"), dict)
                    else {}
                ),
                OVERALL_SCORE_FIELD: scores.get(
                    OVERALL_SCORE_FIELD, {"score": "N/A", "evidence": ""}
                ),
                "clinical_appropriateness": scores.get(
                    "clinical_appropriateness", {"score": "N/A", "evidence": ""}
                ),
                "risk_level": scores.get("risk_level", "unknown"),
                "overall_comment": scores.get("overall_comment", ""),
            }
            turns.append(turn_data)

            # Update history for next turn
            history.append((user_msg, response))

        case_averages = compute_dimension_averages(turns)

        # Multi-turn coherence assessment
        coherence = {}
        if len(turns) >= 2:
            print(f"    Scoring multi-turn coherence...")
            coherence = score_coherence(call_fn, judge_system_prompt, situation, turns)

        case_result = {
            "case_id": case_id,
            "title": title,
            "turns": turns,
            "case_averages": case_averages,
            "coherence": coherence,
        }
        case_results.append(case_result)

        # Append to progress file for crash recovery
        with open(progress_path, "a") as f:
            f.write(json.dumps(case_result, ensure_ascii=False) + "\n")

    # Compute model-level averages
    all_turns = [t for cr in case_results for t in cr["turns"]]
    model_averages = compute_dimension_averages(all_turns)

    result = {
        "metadata": {
            "model_key": model_key,
            "model_path": model_path,
            "judge": judge_key,
            "system_prompt": system_prompt,
            "total_cases": len(case_results),
            "total_turns": len(all_turns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "cases": case_results,
        "model_averages": model_averages,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Clean up progress file after successful save
    if progress_path.exists():
        progress_path.unlink()

    print(f"\n  Saved results to {output_path}")
    print(f"  Model averages: {model_averages}")

    # Free GPU memory
    unload_model(model, tokenizer)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary_markdown(output_dir: Path, cases: list[dict]):
    """Read all per-model JSONs and generate comparison_summary.md."""
    model_files = sorted(output_dir.glob("*.json"))
    if not model_files:
        print("No model result files found for summary generation.")
        return

    # Load all model results
    results = {}
    for f in model_files:
        with open(f) as fh:
            data = json.load(fh)
        key = data["metadata"]["model_key"]
        results[key] = data

    if not results:
        return

    # Determine model order
    model_order = [k for k in MODEL_REGISTRY if k in results]

    dim_headers = {
        "empathy": "Empathy",
        "cbt": "CBT",
        "guided_discovery": "Guided Disc.",
        "safety": "Safety",
    }

    lines = ["# Case-Based Model Comparison", ""]

    # Overall scores table
    lines.append("## Overall Scores")
    lines.append("")
    header = (
        "| Model          | "
        + " | ".join(dim_headers.values())
        + " | Overall (0-8) |"
    )
    separator = "|" + "---|" * (len(dim_headers) + 2)
    lines.append(header)
    lines.append(separator)

    for key in model_order:
        data = results[key]
        avgs = data["model_averages"]
        display = MODEL_DISPLAY_NAMES.get(key, key)
        row = f"| {display:<14} |"
        for dim in DIMENSIONS:
            val = avgs.get(dim, "N/A")
            if isinstance(val, str):
                row += f" {val:>5} |"
            else:
                row += f" {val:>5.1f} |"
        overall = avgs.get(OVERALL_SCORE_FIELD, "N/A")
        if isinstance(overall, str):
            row += f" {overall:>7} |"
        else:
            row += f" {overall:>7.1f} |"
        lines.append(row)

    lines.append("")

    # Per-case breakdown with per-turn scores
    lines.append("## Per-Case Breakdown")
    lines.append("")

    first_model_data = results[model_order[0]]
    case_list = first_model_data["cases"]

    turn_header = (
        "| Turn | Empathy | CBT | Guided Disc. | Safety "
        "| Overall | Judge Comment |"
    )
    turn_sep = "|---|---|---|---|---|---|---|"

    def _fmt_score(val) -> str:
        if isinstance(val, (int, float)):
            return f"{val}"
        return str(val) if val else "N/A"

    def _turn_overall(turn_scores: dict) -> str:
        overall_data = turn_scores.get(OVERALL_SCORE_FIELD, {})
        overall_score = (
            overall_data.get("score") if isinstance(overall_data, dict) else overall_data
        )
        if isinstance(overall_score, (int, float)):
            return f"{overall_score:.1f}"

        nums = []
        for dim in DIMENSIONS:
            d = turn_scores.get(dim, {})
            s = d.get("score") if isinstance(d, dict) else d
            if isinstance(s, (int, float)):
                nums.append(s)
        if len(nums) != len(DIMENSIONS):
            return "N/A"
        return f"{sum(nums):.1f}"

    for case_info in case_list:
        case_id = case_info["case_id"]
        title = case_info["title"]
        lines.append(f"### {case_id.replace('_', ' ').title()}: {title}")
        lines.append("")

        for key in model_order:
            data = results[key]
            case_data = None
            for c in data["cases"]:
                if c["case_id"] == case_id:
                    case_data = c
                    break
            if case_data is None:
                continue

            display = MODEL_DISPLAY_NAMES.get(key, key)
            lines.append(f"#### {display}")
            lines.append("")
            lines.append(turn_header)
            lines.append(turn_sep)

            for turn in case_data.get("turns", []):
                turn_num = turn.get("turn", "?")
                scores = turn.get("scores", {})
                comment = turn.get("overall_comment", "").replace("|", "/")
                turn_with_overall = dict(scores)
                turn_with_overall[OVERALL_SCORE_FIELD] = turn.get(
                    OVERALL_SCORE_FIELD, {"score": "N/A", "evidence": ""}
                )

                row = f"| {turn_num} |"
                for dim in DIMENSIONS:
                    dim_data = scores.get(dim, {})
                    s = dim_data.get("score") if isinstance(dim_data, dict) else dim_data
                    row += f" {_fmt_score(s)} |"
                row += f" {_turn_overall(turn_with_overall)} |"
                # Truncate very long comments for table readability
                short_comment = comment[:120] + "..." if len(comment) > 120 else comment
                row += f" {short_comment} |"
                lines.append(row)

            # Average row
            avgs = case_data["case_averages"]
            avg_row = "| **Avg** |"
            for dim in DIMENSIONS:
                val = avgs.get(dim, "N/A")
                if isinstance(val, str):
                    avg_row += f" **{val}** |"
                else:
                    avg_row += f" **{val:.1f}** |"
            overall = avgs.get(OVERALL_SCORE_FIELD, "N/A")
            if isinstance(overall, str):
                avg_row += f" **{overall}** |"
            else:
                avg_row += f" **{overall:.1f}** |"
            avg_row += " |"
            lines.append(avg_row)
            lines.append("")

            # Detailed evidence in collapsible block
            lines.append(f"<details><summary>Dimension evidence (per-turn)</summary>")
            lines.append("")
            for turn in case_data.get("turns", []):
                turn_num = turn.get("turn", "?")
                scores = turn.get("scores", {})
                comment = turn.get("overall_comment", "")
                lines.append(f"**Turn {turn_num}**")
                for dim in DIMENSIONS:
                    dim_data = scores.get(dim, {})
                    if isinstance(dim_data, dict):
                        s = dim_data.get("score", "N/A")
                        ev = dim_data.get("evidence", "")
                    else:
                        s, ev = dim_data, ""
                    lines.append(
                        f"- **{dim_headers.get(dim, dim)}** ({s}): {ev}"
                    )
                    if dim == "cbt":
                        turn_subscores = turn.get("cbt_subscores", {})
                        if isinstance(turn_subscores, dict):
                            for sub in CBT_SUBSCORES:
                                sub_data = turn_subscores.get(sub, {})
                                sub_score = (
                                    sub_data.get("score")
                                    if isinstance(sub_data, dict)
                                    else "N/A"
                                )
                                sub_ev = (
                                    sub_data.get("evidence", "")
                                    if isinstance(sub_data, dict)
                                    else ""
                                )
                                sub_label = sub.replace("_", " ").title()
                                lines.append(
                                    f"  - {sub_label} ({sub_score}): {sub_ev}"
                                )
                overall_data = turn.get(OVERALL_SCORE_FIELD, {})
                overall_score = (
                    overall_data.get("score")
                    if isinstance(overall_data, dict)
                    else overall_data
                )
                overall_ev = (
                    overall_data.get("evidence", "")
                    if isinstance(overall_data, dict)
                    else ""
                )
                lines.append(f"- **Overall (0-8)** ({overall_score}): {overall_ev}")
                if comment:
                    lines.append(f"- *Overall*: {comment}")
                lines.append("")
            lines.append("</details>")
            lines.append("")

            # Coherence scores (if available)
            coherence = case_data.get("coherence", {})
            if coherence and any(isinstance(coherence.get(d), dict) for d in COHERENCE_DIMS):
                lines.append("**Coherence Assessment:**")
                for dim in COHERENCE_DIMS:
                    dim_data = coherence.get(dim, {})
                    if isinstance(dim_data, dict):
                        s = dim_data.get("score", "N/A")
                        ev = dim_data.get("evidence", "")
                        display_dim = dim.replace("_", " ").title()
                        lines.append(f"- {display_dim}: **{s}** -- {ev}")
                coh_comment = coherence.get("overall_coherence_comment", "")
                if coh_comment:
                    lines.append(f"- *Summary*: {coh_comment}")
                lines.append("")

    # Footer
    judge_key = first_model_data["metadata"].get("judge", "unknown")
    timestamp = first_model_data["metadata"].get("timestamp", "unknown")
    lines.append(f"Judge: {judge_key}")
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    summary_path = output_dir / "comparison_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSummary saved to {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run case-based multi-turn assessment with LLM judge scoring",
    )
    parser.add_argument(
        "--cases", type=str, default="evaluation/cases.json",
        help="Path to cases JSON",
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation/case_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model keys to run (default: all 6)",
    )
    parser.add_argument(
        "--judge", type=str, default="deepseek",
        choices=["deepseek", "gpt-4o", "gemini"],
        help="Judge model (default: deepseek)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip completed models (output JSON) and completed cases (progress file)",
    )
    parser.add_argument(
        "--prompt", type=str, default="evaluation/llm_judge_prompt.md",
        help="Path to judge prompt markdown",
    )
    args = parser.parse_args()

    # Load cases
    with open(args.cases) as f:
        case_data = json.load(f)
    cases = case_data["cases"]
    print(f"Loaded {len(cases)} cases from {args.cases}")

    # Load judge prompt
    judge_system_prompt, judge_user_template = load_judge_prompt(args.prompt)
    print(f"Loaded judge prompt from {args.prompt}")

    # Setup judge API client
    import openai

    if args.judge == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        judge_model = "deepseek-chat"
    elif args.judge == "gpt-4o":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        client = openai.OpenAI(api_key=api_key)
        judge_model = "gpt-4o"
    elif args.judge == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        judge_model = "gemini-2.5-flash"
    else:
        raise RuntimeError(f"Unsupported judge: {args.judge}")

    call_fn = lambda sys_prompt, usr_msg: call_openai(client, sys_prompt, usr_msg, judge_model)
    print(f"Judge: {judge_model}")

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to run
    model_keys = args.models or list(MODEL_REGISTRY.keys())

    total_turns = sum(len(c["user_turns"]) for c in cases)
    print(f"Models to run: {model_keys}")
    print(f"Total turns per model: {total_turns}")
    print(f"Estimated judge API calls: {len(model_keys) * total_turns}")
    print()

    for model_key in model_keys:
        if model_key not in MODEL_REGISTRY:
            print(f"Unknown model key: {model_key}, skipping")
            continue

        # Resume check
        output_path = output_dir / f"{model_key}.json"
        if args.resume and output_path.exists():
            print(f"\nSkipping {model_key} (output exists: {output_path})")
            continue

        model_path = MODEL_REGISTRY[model_key]

        run_model(
            model_key=model_key,
            model_path=model_path,
            cases=cases,
            system_prompt=SYSTEM_PROMPT,
            judge_system_prompt=judge_system_prompt,
            judge_user_template=judge_user_template,
            call_fn=call_fn,
            judge_key=args.judge,
            output_dir=output_dir,
        )

    # Generate comparison summary from all available results
    print("\nGenerating comparison summary...")
    generate_summary_markdown(output_dir, cases)

    print("\nAll done.")


if __name__ == "__main__":
    main()
