#!/usr/bin/env python3
"""
Score model responses via LLM-as-a-judge on the v2 clinical rubric.

Supports GPT-4o, Claude, DeepSeek V3, and Gemini 2.5 Flash as judges.
Runs 3 times per sample for inter-run consistency measurement. Supports resume
via progress files for mid-run recovery.

Usage:
    python scripts/evaluation/run_llm_judge.py --judge gpt-4o --runs 3 --resume
    python scripts/evaluation/run_llm_judge.py --judge deepseek --runs 1 --models qwen2.5-7b
    python scripts/evaluation/run_llm_judge.py --judge gemini --runs 1 --models qwen2.5-7b
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

JUDGE_MODELS = {
    "gpt-4o": "gpt-4o",
    "claude": "claude-sonnet-4-5-20250929",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.5-flash",
}

DIMENSIONS = ["empathy", "cbt", "guided_discovery", "safety"]
CBT_SUBSCORES = [
    "cognitive_reconstruction",
    "behavioral_activation",
    "positive_encouragement",
    "psychoeducation",
]
RUBRIC_VERSION = "v2_2026_04"

# Retry settings
MAX_RETRIES = 5
BASE_DELAY = 2.0
MAX_DELAY = 60.0


def load_judge_prompt(prompt_path: str = "evaluation/llm_judge_prompt.md") -> tuple[str, str]:
    """Parse the judge prompt markdown to extract system prompt and user template.

    Returns:
        (system_prompt, user_template) where user_template has placeholders:
        {CONVERSATION_HISTORY}, {USER_INPUT}, {MODEL_RESPONSE}
    """
    with open(prompt_path) as f:
        content = f.read()

    # Extract the code block after "## System Prompt"
    code_match = re.search(r"```\n(.+?)```", content, re.DOTALL)
    if not code_match:
        raise ValueError(f"Could not find system prompt code block in {prompt_path}")

    full_prompt = code_match.group(1).strip()

    # Split at the placeholders section: everything before "Conversation context:"
    # is rubric/system instructions; everything after becomes the user template.
    parts = full_prompt.split("Conversation context:", 1)
    if len(parts) != 2:
        raise ValueError("Could not split prompt at 'Conversation context:' marker")

    system_prompt = parts[0].strip()
    user_template_raw = "Conversation context:" + parts[1]

    # Escape non-placeholder braces so str.format() can safely render JSON blocks.
    placeholders = {
        "{CONVERSATION_HISTORY}": "__PH_CONVERSATION_HISTORY__",
        "{USER_INPUT}": "__PH_USER_INPUT__",
        "{MODEL_RESPONSE}": "__PH_MODEL_RESPONSE__",
    }
    for placeholder, token in placeholders.items():
        user_template_raw = user_template_raw.replace(placeholder, token)
    user_template = user_template_raw.replace("{", "{{").replace("}", "}}")
    for placeholder, token in placeholders.items():
        user_template = user_template.replace(token, placeholder)
    user_template = user_template.strip()

    return system_prompt, user_template


def parse_score(value) -> int | str:
    """Parse a score value that could be int, string int, or 'N/A'."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip().upper()
        if stripped == "N/A":
            return "N/A"
        try:
            return int(stripped)
        except ValueError:
            return "N/A"
    if isinstance(value, float):
        return int(value)
    return "N/A"


def parse_judge_response(text: str) -> dict:
    """Parse JSON from judge response, handling markdown code fences."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    data = json.loads(cleaned)

    # Validate required keys
    required = ["risk_level", "empathy", "cbt", "guided_discovery", "safety", "overall_comment"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    # Normalize scores
    for dim in DIMENSIONS:
        if isinstance(data.get(dim), dict) and "score" in data[dim]:
            data[dim]["score"] = parse_score(data[dim]["score"])
        else:
            raise ValueError(f"Dimension {dim} missing score field")

    # Normalize CBT technique subscores (new rubric field)
    cbt_data = data.get("cbt", {})
    subscores_raw = cbt_data.get("subscores", {}) if isinstance(cbt_data, dict) else {}
    if not isinstance(subscores_raw, dict):
        subscores_raw = {}
    normalized_subscores = {}
    for sub in CBT_SUBSCORES:
        sub_data = subscores_raw.get(sub, {})
        if isinstance(sub_data, dict):
            normalized_subscores[sub] = {
                "score": parse_score(sub_data.get("score")),
                "evidence": sub_data.get("evidence", ""),
            }
        else:
            normalized_subscores[sub] = {"score": "N/A", "evidence": ""}
    data["cbt"]["subscores"] = normalized_subscores

    # Keep backward-compatibility field if present (or synthesize if missing)
    compat = data.get("clinical_appropriateness", {})
    compat_score = parse_score(compat.get("score")) if isinstance(compat, dict) else "N/A"
    compat_evidence = compat.get("evidence", "") if isinstance(compat, dict) else ""
    data["clinical_appropriateness"] = {
        "score": compat_score,
        "evidence": compat_evidence,
    }

    # Normalize rubric overall score (0-8). Derive if missing.
    derived_overall = "N/A"
    if all(isinstance(data[dim]["score"], int) for dim in DIMENSIONS):
        derived_overall = sum(int(data[dim]["score"]) for dim in DIMENSIONS)

    overall_data = data.get("overall_score_0_to_8", {})
    overall_score = parse_score(overall_data.get("score")) if isinstance(overall_data, dict) else derived_overall
    if isinstance(overall_score, int):
        overall_score = min(max(overall_score, 0), 8)
    if overall_score == "N/A":
        overall_score = derived_overall
    overall_evidence = (
        overall_data.get("evidence", "")
        if isinstance(overall_data, dict)
        else ""
    )
    if not overall_evidence and isinstance(overall_score, int):
        overall_evidence = "Derived from empathy + guided_discovery + cbt + safety."
    data["overall_score_0_to_8"] = {
        "score": overall_score,
        "evidence": overall_evidence,
    }

    return data


def call_openai(client, system_prompt: str, user_message: str, model: str) -> str:
    """Call OpenAI API with structured JSON output."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def call_anthropic(client, system_prompt: str, user_message: str, model: str) -> str:
    """Call Anthropic API."""
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
        ],
    )
    return response.content[0].text


def call_judge_with_retry(call_fn, system_prompt: str, user_message: str) -> dict:
    """Call the judge API with exponential backoff retry and JSON parsing."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            raw = call_fn(system_prompt, user_message)
            return parse_judge_response(raw)
        except json.JSONDecodeError as e:
            last_error = e
            print(f"    JSON parse error (attempt {attempt + 1}): {e}")
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            is_retryable = any(kw in error_str for kw in [
                "rate_limit", "rate limit", "429", "500", "502", "503", "overloaded",
            ])
            if not is_retryable and attempt > 0:
                raise
            print(f"    API error (attempt {attempt + 1}): {e}")

        delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
        print(f"    Retrying in {delay:.0f}s...")
        time.sleep(delay)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


def load_progress(progress_path: Path) -> set[str]:
    """Load completed sample IDs from progress file."""
    completed = set()
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        completed.add(entry["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


def score_model_run(
    model_key: str,
    run_id: int,
    responses: list[dict],
    test_samples: dict[str, dict],
    system_prompt: str,
    user_template: str,
    call_fn,
    judge_model: str,
    judge_key: str,
    scores_dir: Path,
    resume: bool,
):
    """Score all responses for one model in one run."""
    output_path = scores_dir / f"{model_key}_{judge_key}_run{run_id}.json"
    progress_path = scores_dir / f".{model_key}_{judge_key}_run{run_id}.progress.jsonl"

    # Check if already complete
    if resume and output_path.exists():
        print(f"  Run {run_id} already complete: {output_path}, skipping")
        return

    # Load progress for mid-run resume
    completed_ids = load_progress(progress_path) if resume else set()
    if completed_ids:
        print(f"  Resuming run {run_id}: {len(completed_ids)} already scored")

    scores = []
    # Reload previously scored items from progress
    if completed_ids and progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        scores.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    progress_file = open(progress_path, "a")

    try:
        for resp in responses:
            sid = resp["sample_id"]
            if sid in completed_ids:
                continue

            sample = test_samples.get(sid, {})
            context = sample.get("conversation_context", "")
            user_input = sample.get("user_message", resp.get("user_message", ""))

            user_msg = user_template.format(
                CONVERSATION_HISTORY=context if context else "(No prior context)",
                USER_INPUT=user_input,
                MODEL_RESPONSE=resp["model_response"],
            )

            try:
                result = call_judge_with_retry(call_fn, system_prompt, user_msg)
            except Exception as e:
                print(f"  FAILED {sid}: {e}")
                result = {
                    "risk_level": "unknown",
                    "empathy": {"score": "N/A", "evidence": f"Error: {e}"},
                    "cbt": {
                        "score": "N/A",
                        "evidence": f"Error: {e}",
                        "subscores": {
                            sub: {"score": "N/A", "evidence": f"Error: {e}"}
                            for sub in CBT_SUBSCORES
                        },
                    },
                    "guided_discovery": {"score": "N/A", "evidence": f"Error: {e}"},
                    "safety": {"score": "N/A", "evidence": f"Error: {e}"},
                    "clinical_appropriateness": {"score": "N/A", "evidence": f"Error: {e}"},
                    "overall_score_0_to_8": {"score": "N/A", "evidence": f"Error: {e}"},
                    "overall_comment": f"Scoring failed: {e}",
                }

            score_entry = {"sample_id": sid, "risk_level": result.get("risk_level", "unknown")}
            for dim in DIMENSIONS:
                score_entry[dim] = result.get(dim, {"score": "N/A", "evidence": ""})
            score_entry["cbt_subscores"] = (
                result.get("cbt", {}).get("subscores", {})
                if isinstance(result.get("cbt"), dict)
                else {}
            )
            score_entry["overall_score_0_to_8"] = result.get(
                "overall_score_0_to_8",
                {"score": "N/A", "evidence": ""},
            )
            score_entry["clinical_appropriateness"] = result.get(
                "clinical_appropriateness",
                {"score": "N/A", "evidence": ""},
            )
            score_entry["overall_comment"] = result.get("overall_comment", "")

            scores.append(score_entry)
            # Write progress
            progress_file.write(json.dumps(score_entry) + "\n")
            progress_file.flush()
    finally:
        progress_file.close()

    # Write final output
    final = {
        "metadata": {
            "model_key": model_key,
            "judge_model": judge_model,
            "run_id": run_id,
            "total_scores": len(scores),
            "rubric_version": RUBRIC_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "scores": scores,
    }

    with open(output_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    # Clean up progress file on successful completion
    if progress_path.exists():
        progress_path.unlink()

    print(f"  Run {run_id} complete: {len(scores)} scores saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge scoring")
    parser.add_argument("--judge", type=str, default="gpt-4o", choices=list(JUDGE_MODELS.keys()),
                        help="Judge model to use")
    parser.add_argument("--runs", type=int, default=3, help="Number of scoring runs per model")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--models", nargs="+", default=None, help="Model keys to score (default: all in responses/)")
    parser.add_argument("--responses-dir", type=str, default="evaluation/responses", help="Directory with model responses")
    parser.add_argument("--test-set", type=str, default="evaluation/test_set.json", help="Path to test set JSON")
    parser.add_argument("--scores-dir", type=str, default="evaluation/scores", help="Output directory for scores")
    parser.add_argument("--prompt", type=str, default="evaluation/llm_judge_prompt.md", help="Path to judge prompt")
    args = parser.parse_args()

    # Load judge prompt
    system_prompt, user_template = load_judge_prompt(args.prompt)
    print(f"Loaded judge prompt from {args.prompt}")

    # Load test set for context lookup
    with open(args.test_set) as f:
        test_data = json.load(f)
    test_samples = {s["sample_id"]: s for s in test_data["samples"]}
    print(f"Loaded {len(test_samples)} test samples")

    # Setup API client
    judge_model = JUDGE_MODELS[args.judge]
    if args.judge == "gpt-4o":
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        client = openai.OpenAI(api_key=api_key)
        call_fn = lambda sys, usr: call_openai(client, sys, usr, judge_model)
    elif args.judge == "deepseek":
        import openai
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        call_fn = lambda sys, usr: call_openai(client, sys, usr, judge_model)
    elif args.judge == "gemini":
        import openai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        call_fn = lambda sys, usr: call_openai(client, sys, usr, judge_model)
    elif args.judge == "claude":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
        call_fn = lambda sys, usr: call_anthropic(client, sys, usr, judge_model)
    else:
        raise RuntimeError(f"Unknown judge: {args.judge}")

    print(f"Judge model: {judge_model}")

    # Find response files
    responses_dir = Path(args.responses_dir)
    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        model_keys = args.models
    else:
        model_keys = sorted(
            p.stem for p in responses_dir.glob("*.json")
        )

    if not model_keys:
        print(f"No response files found in {responses_dir}")
        return

    print(f"Models to score: {model_keys}")
    print(f"Runs per model: {args.runs}")

    total_calls = len(model_keys) * args.runs * len(test_samples)
    print(f"Estimated API calls: {total_calls}")
    print()

    for model_key in model_keys:
        resp_path = responses_dir / f"{model_key}.json"
        if not resp_path.exists():
            print(f"Response file not found: {resp_path}, skipping")
            continue

        with open(resp_path) as f:
            resp_data = json.load(f)
        responses = resp_data["responses"]

        print(f"{'=' * 60}")
        print(f"Scoring: {model_key} ({len(responses)} responses)")
        print(f"{'=' * 60}")

        for run_id in range(1, args.runs + 1):
            score_model_run(
                model_key=model_key,
                run_id=run_id,
                responses=responses,
                test_samples=test_samples,
                system_prompt=system_prompt,
                user_template=user_template,
                call_fn=call_fn,
                judge_model=judge_model,
                judge_key=args.judge,
                scores_dir=scores_dir,
                resume=args.resume,
            )

    print("\nAll scoring complete.")


if __name__ == "__main__":
    main()
