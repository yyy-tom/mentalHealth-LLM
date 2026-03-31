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
    max_new_tokens: int = 512,
) -> str:
    """Generate a response with multi-turn history, handling Gemma compatibility."""
    use_system_role = _supports_system_role(tokenizer)

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

def format_conversation_context(history: list[tuple[str, str]]) -> str:
    """Format history as 'User: ... / Counselor: ...' for the judge."""
    if not history:
        return "(No prior context)"
    lines = []
    for user_msg, bot_msg in history:
        lines.append(f"User: {user_msg}")
        lines.append(f"Counselor: {bot_msg}")
    return "\n".join(lines)


def compute_dimension_averages(turns: list[dict]) -> dict:
    """Compute average scores per dimension across turns."""
    totals = {dim: [] for dim in DIMENSIONS}
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

    averages = {}
    all_scores = []
    for dim in DIMENSIONS:
        if totals[dim]:
            avg = round(sum(totals[dim]) / len(totals[dim]), 2)
            averages[dim] = avg
            all_scores.append(avg)
        else:
            averages[dim] = "N/A"

    averages["overall"] = round(sum(all_scores) / len(all_scores), 2) if all_scores else "N/A"
    return averages


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

    case_results = []

    for case in cases:
        case_id = case["case_id"]
        title = case["title"]
        user_turns = case["user_turns"]
        print(f"\n  Case {case_id}: {title} ({len(user_turns)} turns)")

        history: list[tuple[str, str]] = []
        turns = []

        for turn_idx, user_msg in enumerate(tqdm(user_turns, desc=f"    {case_id}", leave=False)):
            turn_num = turn_idx + 1

            # Generate response
            t0 = time.time()
            try:
                response = generate_response_multiturn(
                    model, tokenizer, user_msg, system_prompt, history,
                )
            except Exception as e:
                print(f"    ERROR generating turn {turn_num}: {e}")
                response = f"[ERROR: {e}]"
            gen_time = round(time.time() - t0, 2)

            # Score this turn via judge
            context = format_conversation_context(history)
            judge_user_msg = judge_user_template.format(
                CONVERSATION_HISTORY=context,
                USER_INPUT=user_msg,
                MODEL_RESPONSE=response,
            )

            try:
                scores = call_judge_with_retry(call_fn, judge_system_prompt, judge_user_msg)
            except Exception as e:
                print(f"    ERROR scoring turn {turn_num}: {e}")
                scores = {
                    dim: {"score": "N/A", "evidence": f"Error: {e}"}
                    for dim in DIMENSIONS
                }
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
                "risk_level": scores.get("risk_level", "unknown"),
                "overall_comment": scores.get("overall_comment", ""),
            }
            turns.append(turn_data)

            # Update history for next turn
            history.append((user_msg, response))

        case_averages = compute_dimension_averages(turns)
        case_results.append({
            "case_id": case_id,
            "title": title,
            "turns": turns,
            "case_averages": case_averages,
        })

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
        "clinical_appropriateness": "Clinical",
    }

    lines = ["# Case-Based Model Comparison", ""]

    # Overall scores table
    lines.append("## Overall Scores (0-2 scale)")
    lines.append("")
    header = "| Model          | " + " | ".join(dim_headers.values()) + " | Overall |"
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
        overall = avgs.get("overall", "N/A")
        if isinstance(overall, str):
            row += f" {overall:>7} |"
        else:
            row += f" {overall:>7.1f} |"
        lines.append(row)

    lines.append("")

    # Per-case breakdown
    lines.append("## Per-Case Breakdown")
    lines.append("")

    first_model_data = results[model_order[0]]
    case_list = first_model_data["cases"]

    for case_info in case_list:
        case_id = case_info["case_id"]
        title = case_info["title"]
        lines.append(f"### {case_id.replace('_', ' ').title()}: {title}")
        lines.append("")
        lines.append(header)
        lines.append(separator)

        for key in model_order:
            data = results[key]
            # Find matching case
            case_data = None
            for c in data["cases"]:
                if c["case_id"] == case_id:
                    case_data = c
                    break

            if case_data is None:
                continue

            avgs = case_data["case_averages"]
            display = MODEL_DISPLAY_NAMES.get(key, key)
            row = f"| {display:<14} |"
            for dim in DIMENSIONS:
                val = avgs.get(dim, "N/A")
                if isinstance(val, str):
                    row += f" {val:>5} |"
                else:
                    row += f" {val:>5.1f} |"
            overall = avgs.get("overall", "N/A")
            if isinstance(overall, str):
                row += f" {overall:>7} |"
            else:
                row += f" {overall:>7.1f} |"
            lines.append(row)

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
        help="Skip models that already have output JSON",
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
