"""
Bot-facing interface for LLM-as-a-judge scoring.

Provides async scoring of bot responses using DeepSeek and/or Gemini judges,
with aggregation and user-facing report formatting.
"""

import asyncio
import logging
from pathlib import Path

from evaluation.run_llm_judge import (
    DIMENSIONS,
    call_judge_with_retry,
    call_openai,
    load_judge_prompt,
    parse_score,
)

logger = logging.getLogger(__name__)

# Module-level state set by init_judges()
_judge_clients: dict[str, dict] = {}
_system_prompt: str = ""
_user_template: str = ""

DIMENSION_LABELS = {
    "empathy": "Empathy",
    "cbt": "CBT Technique",
    "guided_discovery": "Guided Discovery",
    "safety": "Safety",
    "clinical_appropriateness": "Clinical Appropriateness",
}

_JUDGE_CONFIGS = {
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
    },
}


def init_judges(
    deepseek_key: str | None,
    gemini_key: str | None,
    prompt_path: str = "evaluation/llm_judge_prompt.md",
) -> list[str]:
    """Initialise judge clients for available API keys.

    Returns list of active judge names.
    """
    global _system_prompt, _user_template

    keys = {"deepseek": deepseek_key, "gemini": gemini_key}
    if not any(keys.values()):
        return []

    try:
        import openai
    except ModuleNotFoundError:
        logger.warning("openai package not installed — scoring disabled")
        return []

    active: list[str] = []

    for name, cfg in _JUDGE_CONFIGS.items():
        api_key = keys.get(name)
        if not api_key:
            continue
        client = openai.OpenAI(api_key=api_key, base_url=cfg["base_url"])
        _judge_clients[name] = {"client": client, "model": cfg["model"]}
        active.append(name)

    if active:
        _system_prompt, _user_template = load_judge_prompt(prompt_path)

    return active


def format_history_for_judge(history: list[tuple[str, str]]) -> str:
    """Convert conversation history into a string for the judge prompt."""
    if not history:
        return "(No prior context)"
    lines: list[str] = []
    for user_msg, bot_msg in history:
        lines.append(f"User: {user_msg}")
        lines.append(f"Counselor: {bot_msg}")
    return "\n".join(lines)


def score_single_judge(
    judge_name: str,
    user_input: str,
    model_response: str,
    context: str,
) -> dict | None:
    """Score one exchange with a single judge. Returns parsed result or None."""
    try:
        info = _judge_clients[judge_name]
        client, model = info["client"], info["model"]

        user_msg = _user_template.format(
            CONVERSATION_HISTORY=context,
            USER_INPUT=user_input,
            MODEL_RESPONSE=model_response,
        )

        call_fn = lambda sys, usr: call_openai(client, sys, usr, model)
        return call_judge_with_retry(call_fn, _system_prompt, user_msg)
    except Exception:
        logger.exception("Judge %s failed", judge_name)
        return None


async def score_exchange_async(
    user_input: str,
    model_response: str,
    history_before: list[tuple[str, str]],
) -> dict[str, dict]:
    """Score an exchange with all active judges in parallel.

    Returns ``{"deepseek": {…}, "gemini": {…}}`` with only successful results.
    """
    context = format_history_for_judge(history_before)

    async def _run(name: str) -> tuple[str, dict | None]:
        result = await asyncio.to_thread(
            score_single_judge, name, user_input, model_response, context
        )
        return name, result

    tasks = [_run(name) for name in _judge_clients]
    pairs = await asyncio.gather(*tasks, return_exceptions=True)

    results: dict[str, dict] = {}
    for item in pairs:
        if isinstance(item, Exception):
            logger.error("Judge task raised: %s", item)
            continue
        name, result = item
        if result is not None:
            results[name] = result
    return results


def aggregate_scores(score_list: list[dict]) -> dict:
    """Aggregate per-exchange score dicts into mean scores per dimension.

    Each entry in *score_list* is a dict mapping judge names to parsed judge
    results (as returned by ``score_exchange_async``).
    """
    dim_values: dict[str, list[float]] = {d: [] for d in DIMENSIONS}

    for exchange in score_list:
        for _judge_name, result in exchange.items():
            for dim in DIMENSIONS:
                entry = result.get(dim)
                if not isinstance(entry, dict):
                    continue
                score = parse_score(entry.get("score"))
                if isinstance(score, int):
                    dim_values[dim].append(float(score))

    agg: dict[str, float | int | list[str]] = {}
    all_scores: list[float] = []
    for dim in DIMENSIONS:
        vals = dim_values[dim]
        avg = sum(vals) / len(vals) if vals else 0.0
        agg[dim] = round(avg, 1)
        all_scores.extend(vals)

    agg["overall"] = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0
    agg["count"] = len(score_list)
    agg["judge_names"] = list(_judge_clients.keys())
    return agg


def format_score_report(
    score_list: list[dict],
    total_exchanges: int,
    pending_count: int,
) -> str:
    """Build a user-facing quality report string."""
    scored = len(score_list)
    if scored == 0:
        return "No scores available yet. Send a few messages first, then try /score again."

    agg = aggregate_scores(score_list)

    lines: list[str] = []
    lines.append(f"Conversation Quality Report ({scored} exchange{'s' if scored != 1 else ''} scored)\n")
    lines.append(f"{'Dimension':<26} Avg Score")
    lines.append("-" * 38)
    for dim in DIMENSIONS:
        label = DIMENSION_LABELS.get(dim, dim)
        val = agg.get(dim, 0.0)
        lines.append(f"{label:<26} {val:>4.1f} / 5")
    lines.append("-" * 38)
    lines.append(f"{'Overall':<26} {agg['overall']:>4.1f} / 5\n")

    judge_str = ", ".join(n.capitalize() for n in agg.get("judge_names", []))
    lines.append(f"Judges: {judge_str}")

    if pending_count > 0:
        lines.append(f"({pending_count} exchange{'s' if pending_count != 1 else ''} still being scored...)")

    return "\n".join(lines)
