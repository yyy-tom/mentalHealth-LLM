"""
Streaming event architecture for mental health counseling responses.

Provides an event-based generation pipeline that yields structured events
during the response lifecycle:

  CrisisCheckEvent  — result of the pre-routing crisis safety gate
  SkillRoutedEvent  — which skill was selected and confidence
  TextDeltaEvent    — incremental text chunks during generation
  MessageStopEvent  — final event with metadata (safety score, token count)

Usage with the Telegram bot:

    async for event in stream_response(question, model_key, history):
        if isinstance(event, TextDeltaEvent):
            buffer += event.text
            # flush to Telegram when buffer reaches threshold

Source pattern: claw-code AssistantEvent enum (TextDelta, ToolUse, MessageStop).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    CRISIS_CHECK = "crisis_check"
    SKILL_ROUTED = "skill_routed"
    TEXT_DELTA = "text_delta"
    MESSAGE_STOP = "message_stop"


@dataclass(frozen=True, slots=True)
class CrisisCheckEvent:
    """Emitted after the crisis safety gate runs."""

    type: EventType = field(default=EventType.CRISIS_CHECK, init=False)
    is_crisis: bool = False
    keyword_triggered: bool = False
    embedding_triggered: bool = False
    embedding_score: float = 0.0


@dataclass(frozen=True, slots=True)
class SkillRoutedEvent:
    """Emitted after skill routing completes."""

    type: EventType = field(default=EventType.SKILL_ROUTED, init=False)
    skill: str = "general-support"
    confidence: float = 0.0
    router_backend: str = "keyword"


@dataclass(frozen=True, slots=True)
class TextDeltaEvent:
    """Emitted for each chunk of generated text."""

    type: EventType = field(default=EventType.TEXT_DELTA, init=False)
    text: str = ""


@dataclass(frozen=True, slots=True)
class MessageStopEvent:
    """Emitted when generation is complete."""

    type: EventType = field(default=EventType.MESSAGE_STOP, init=False)
    full_response: str = ""
    skill: str = ""
    token_count: int = 0
    generation_time_ms: int = 0


StreamEvent = CrisisCheckEvent | SkillRoutedEvent | TextDeltaEvent | MessageStopEvent


# ---------------------------------------------------------------------------
# Sentence-boundary chunking
# ---------------------------------------------------------------------------

_SENTENCE_ENDS = frozenset(".!?\n")


def _split_into_chunks(text: str, min_chunk_len: int = 60) -> list[str]:
    """Split generated text into sentence-aligned chunks for streaming.

    Tries to break on sentence boundaries (.!?\\n). Falls back to the full
    text as a single chunk if no suitable break point is found.
    """
    if len(text) <= min_chunk_len:
        return [text] if text else []

    chunks: list[str] = []
    start = 0

    for i, ch in enumerate(text):
        if ch in _SENTENCE_ENDS and (i - start + 1) >= min_chunk_len:
            chunk = text[start : i + 1].strip()
            if chunk:
                chunks.append(chunk)
            start = i + 1

    # Remaining tail
    tail = text[start:].strip()
    if tail:
        if chunks:
            # Merge short tail into last chunk
            if len(tail) < min_chunk_len // 2:
                chunks[-1] = chunks[-1] + " " + tail
            else:
                chunks.append(tail)
        else:
            chunks.append(tail)

    return chunks


# ---------------------------------------------------------------------------
# Core streaming pipeline
# ---------------------------------------------------------------------------


async def stream_response(
    question: str,
    model_key: str,
    history: list[tuple[str, str]] | None = None,
    adapters_enabled: bool = True,
    prompt_enabled: bool = True,
    *,
    skill_router,
    model_manager,
    chunk_min_len: int = 60,
) -> AsyncIterator[StreamEvent]:
    """Async generator that yields streaming events during response generation.

    This wraps the existing synchronous ``route_and_generate`` logic and
    breaks the result into sentence-aligned chunks so the Telegram bot can
    deliver incremental updates with typing indicators.

    Args:
        question: User message text.
        model_key: Key into ModelManager (e.g. "qwen-ft").
        history: Conversation history as (user, assistant) tuples.
        adapters_enabled: Whether LoRA adapters are active.
        prompt_enabled: Whether to use skill-specific system prompts.
        skill_router: A ``SkillRouter`` instance.
        model_manager: A ``ModelManager`` instance.
        chunk_min_len: Minimum characters per TextDeltaEvent chunk.

    Yields:
        StreamEvent instances in order:
        CrisisCheckEvent -> SkillRoutedEvent -> TextDeltaEvent* -> MessageStopEvent
    """

    # ── 1. Crisis check ──────────────────────────────────────────
    crisis_result = _run_crisis_check(question, history, skill_router)
    yield CrisisCheckEvent(
        is_crisis=crisis_result.get("is_crisis", False),
        keyword_triggered=crisis_result.get("keyword_triggered", False),
        embedding_triggered=crisis_result.get("embedding_triggered", False),
        embedding_score=crisis_result.get("embedding_score", 0.0),
    )

    # ── 2. Skill routing ─────────────────────────────────────────
    skill_name, confidence, details = skill_router.route_with_confidence(
        question, history
    )

    # Crisis gate overrides routing
    if crisis_result.get("is_crisis", False):
        skill_name = "crisis-intervention"
        confidence = 1.0

    yield SkillRoutedEvent(
        skill=skill_name,
        confidence=confidence,
        router_backend=details.get("router_type", skill_router.backend),
    )

    # ── 3. Generation (offloaded to thread) ──────────────────────
    t0 = time.monotonic()

    response, actual_skill = await asyncio.to_thread(
        _generate_sync,
        question=question,
        skill_name=skill_name,
        model_key=model_key,
        history=history,
        adapters_enabled=adapters_enabled,
        prompt_enabled=prompt_enabled,
        skill_router=skill_router,
        model_manager=model_manager,
    )

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # ── 4. Emit text deltas ──────────────────────────────────────
    chunks = _split_into_chunks(response, min_chunk_len=chunk_min_len)
    for chunk in chunks:
        yield TextDeltaEvent(text=chunk)

    # ── 5. Message stop ──────────────────────────────────────────
    yield MessageStopEvent(
        full_response=response,
        skill=actual_skill,
        token_count=len(response.split()),  # rough word-count proxy
        generation_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Helpers (run in thread pool)
# ---------------------------------------------------------------------------


def _run_crisis_check(
    question: str,
    history: list[tuple[str, str]] | None,
    skill_router,
) -> dict:
    """Run crisis gate if the router's delegate exposes one."""
    delegate = getattr(skill_router, "_delegate", None)
    crisis_gate = getattr(delegate, "crisis_gate", None)
    if crisis_gate is not None:
        try:
            return crisis_gate.check(question, history)
        except Exception:
            logger.exception("Crisis gate check failed")
    return {"is_crisis": False}


def _generate_sync(
    *,
    question: str,
    skill_name: str,
    model_key: str,
    history: list[tuple[str, str]] | None,
    adapters_enabled: bool,
    prompt_enabled: bool = True,
    skill_router,
    model_manager,
) -> tuple[str, str]:
    """Synchronous generation — mirrors ``route_and_generate`` in telegram_bot.py."""

    mdl, tok, loaded_skills = model_manager.get(model_key)

    # Activate or disable LoRA adapters
    if loaded_skills:
        if adapters_enabled:
            mdl.enable_adapter_layers()
            if skill_name in loaded_skills:
                mdl.set_adapter(skill_name)
            else:
                fallback = (
                    "general-support"
                    if "general-support" in loaded_skills
                    else loaded_skills[0]
                )
                mdl.set_adapter(fallback)
        else:
            mdl.disable_adapter_layers()

    # Get system prompt only if prompt is enabled
    system_prompt = skill_router.get_system_prompt(skill_name) if prompt_enabled else ""

    # Import generation function lazily to avoid circular imports
    from scripts.telegram_bot import generate_response

    try:
        response = generate_response(question, system_prompt, mdl, tok, history)
    except Exception:
        logger.exception(
            "Generation failed for model=%s skill=%s", model_key, skill_name
        )
        raise
    finally:
        if loaded_skills and not adapters_enabled:
            mdl.enable_adapter_layers()

    if not response or not response.strip():
        response = (
            "I'm not sure how to respond to that. "
            "Could you tell me more about what's on your mind?"
        )

    return response, skill_name


# ---------------------------------------------------------------------------
# Telegram integration helper
# ---------------------------------------------------------------------------


async def send_streaming_response(
    update,
    context,
    question: str,
    model_key: str,
    history: list[tuple[str, str]] | None,
    adapters_enabled: bool,
    *,
    prompt_enabled: bool = True,
    skill_router,
    model_manager,
    typing_interval: float = 4.0,
) -> tuple[str, str, CrisisCheckEvent | None]:
    """High-level helper that streams a response to a Telegram chat.

    Sends typing indicators while generating, then delivers the response
    in sentence-aligned chunks (editing a single message for a smooth UX).

    Args:
        update: Telegram Update object.
        context: Telegram bot context.
        question: User message.
        model_key: Model key string.
        history: Conversation history.
        adapters_enabled: LoRA adapter toggle.
        prompt_enabled: Whether to use skill-specific system prompts.
        skill_router: SkillRouter instance.
        model_manager: ModelManager instance.
        typing_interval: Seconds between typing action refreshes.

    Returns:
        (full_response, skill_name, crisis_event) tuple.
    """
    from telegram.constants import ChatAction

    chat_id = update.effective_chat.id
    skill_name = "general-support"
    full_response = ""
    crisis_event: CrisisCheckEvent | None = None
    sent_message = None

    # Keep sending typing action while generating
    typing_task = asyncio.create_task(
        _keep_typing(context.bot, chat_id, typing_interval)
    )

    try:
        chunks_buffer: list[str] = []

        async for event in stream_response(
            question,
            model_key,
            history,
            adapters_enabled,
            prompt_enabled,
            skill_router=skill_router,
            model_manager=model_manager,
        ):
            if isinstance(event, CrisisCheckEvent):
                crisis_event = event
                logger.info(
                    "Crisis check: is_crisis=%s keyword=%s embedding=%.3f",
                    event.is_crisis,
                    event.keyword_triggered,
                    event.embedding_score,
                )

            elif isinstance(event, SkillRoutedEvent):
                skill_name = event.skill
                logger.info(
                    "Routed to [%s] (confidence=%.2f, backend=%s)",
                    event.skill,
                    event.confidence,
                    event.router_backend,
                )

            elif isinstance(event, TextDeltaEvent):
                chunks_buffer.append(event.text)
                current_text = " ".join(chunks_buffer)

                if sent_message is None:
                    sent_message = await update.message.reply_text(current_text)
                else:
                    try:
                        await sent_message.edit_text(current_text)
                    except Exception:
                        pass  # Telegram may reject edits if text unchanged

            elif isinstance(event, MessageStopEvent):
                full_response = event.full_response
                skill_name = event.skill
                logger.info(
                    "Generation complete: %d tokens, %dms",
                    event.token_count,
                    event.generation_time_ms,
                )
                # Final edit to ensure full text is displayed
                if sent_message and full_response:
                    try:
                        await sent_message.edit_text(full_response)
                    except Exception:
                        pass

    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    # Fallback if no message was sent (e.g. empty response)
    if sent_message is None and full_response:
        await update.message.reply_text(full_response)
    elif sent_message is None:
        fallback = (
            "I'm not sure how to respond to that. "
            "Could you tell me more about what's on your mind?"
        )
        await update.message.reply_text(fallback)
        full_response = fallback

    return full_response, skill_name, crisis_event


async def _keep_typing(bot, chat_id: int, interval: float) -> None:
    """Send typing indicators periodically until cancelled."""
    from telegram.constants import ChatAction

    try:
        while True:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
