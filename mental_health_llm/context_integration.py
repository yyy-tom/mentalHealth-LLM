"""
Context management integration for telegram bot.

Integrates the claw-code components (TieredContextManager, MultiLayerCompactor,
CompactionMemoryStore) with the existing telegram bot session management.

This module provides:
1. EnhancedContextManager - unified interface for context + compaction
2. Session lifecycle hooks for memory persistence
3. Crisis turn tracking and preservation

Usage:
    from mental_health_llm.context_integration import EnhancedContextManager

    # Initialize once at bot startup
    context_mgr = EnhancedContextManager(
        db_path="data/sessions.db",
        memory_db_path="data/memory.db",
        hot_size=4,
        target_tokens=1024,
    )

    # On each message
    history = context_mgr.get_history(user_id)
    # ... generate response ...
    context_mgr.save_turn(user_id, user_msg, response, skill=skill, is_crisis=is_crisis)

    # On session end
    context_mgr.end_session(user_id)
"""

from __future__ import annotations

import logging
from typing import Optional, Callable

from mental_health_llm.tiered_context import TieredContextManager, Tier, ContextSnapshot
from mental_health_llm.compaction_layers import MultiLayerCompactor, CompactionResult
from mental_health_llm.memory_persistence import CompactionMemoryStore
from mental_health_llm.session_store import SQLiteSessionStore

logger = logging.getLogger(__name__)


class EnhancedContextManager:
    """Unified context management with tiered context, compaction, and memory persistence.

    Replaces the simple HISTORY_LIMIT trimming with:
    - Hot/Warm/Cold tier management
    - 5-layer progressive compaction when context exceeds budget
    - Cross-session memory recall for relevant past context
    - Crisis turn preservation across compaction
    """

    def __init__(
        self,
        db_path: str = "data/sessions.db",
        memory_db_path: str = "data/memory.db",
        hot_size: int = 4,
        warm_size: int = 6,
        target_tokens: int = 1024,
        llm_summarizer: Optional[Callable[[str], str]] = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ):
        """Initialize the enhanced context manager.

        Args:
            db_path: Path to session SQLite database.
            memory_db_path: Path to compaction memory database.
            hot_size: Max turns in HOT tier (always verbatim).
            warm_size: Max turns in WARM tier (important, detailed).
            target_tokens: Target token budget for context.
            llm_summarizer: Optional LLM function for session summaries.
            embedding_fn: Optional embedding function for semantic search.
        """
        # Core components
        self._session_store = SQLiteSessionStore(db_path)
        self._memory_store = CompactionMemoryStore(memory_db_path, embedding_fn=embedding_fn)
        self._compactor = MultiLayerCompactor(
            llm_summarizer=llm_summarizer,
            embedding_fn=embedding_fn,
        )

        # Per-user tiered context managers
        self._user_contexts: dict[int, TieredContextManager] = {}

        # Configuration
        self._hot_size = hot_size
        self._warm_size = warm_size
        self._target_tokens = target_tokens

    def _get_context_manager(self, user_id: int) -> TieredContextManager:
        """Get or create a tiered context manager for a user."""
        if user_id not in self._user_contexts:
            self._user_contexts[user_id] = TieredContextManager(
                hot_size=self._hot_size,
                warm_size=self._warm_size,
            )
            # Restore from session store
            self._restore_user_context(user_id)
        return self._user_contexts[user_id]

    def _restore_user_context(self, user_id: int) -> None:
        """Restore user context from session store into tiered manager."""
        mgr = self._user_contexts[user_id]

        # Get raw history from session store
        history = self._session_store.restore_history(user_id)
        crisis_indices = self._session_store.restore_crisis_flags(user_id)

        # Add turns to tiered manager
        for i, (user_msg, asst_msg) in enumerate(history):
            is_crisis = i in crisis_indices
            mgr.add_turn("user", user_msg, is_crisis=is_crisis)
            mgr.add_turn("assistant", asst_msg, is_crisis=is_crisis)

    def get_history(
        self,
        user_id: int,
        include_memory: bool = True,
    ) -> list[tuple[str, str]]:
        """Get conversation history for a user, with optional cross-session memory.

        Returns history as (user_msg, assistant_msg) pairs, ready for generation.
        Applies compaction if context exceeds token budget.
        """
        mgr = self._get_context_manager(user_id)

        # Get current context snapshot
        snapshot = mgr.get_context(max_tokens=self._target_tokens)

        # Convert to history pairs
        history_pairs = snapshot.to_history_pairs()

        # Check if we need additional compaction
        total_tokens = sum(len(u) + len(a) for u, a in history_pairs) // 4

        if total_tokens > self._target_tokens:
            # Apply multi-layer compaction
            crisis_indices = self._get_crisis_indices(user_id)
            result = self._compactor.compact(
                history=history_pairs,
                target_tokens=self._target_tokens,
                preserve_indices=crisis_indices,
                preserve_recent=self._hot_size,
            )

            # If we have a session summary, include past memory context
            if result.session_summary and include_memory:
                # Recall relevant past sessions
                relevant = self._memory_store.recall_relevant(
                    user_id=user_id,
                    current_context=result.session_summary,
                    top_k=2,
                    min_similarity=0.3,
                )
                if relevant:
                    # Prepend relevant memory as context
                    memory_context = " ".join(r.summary for r in relevant[:2])
                    logger.debug(
                        "User %d: Including %d past memories",
                        user_id,
                        len(relevant),
                    )

            # Convert compacted result back to pairs
            return self._messages_to_pairs(result.compacted)

        return history_pairs

    def get_context_messages(
        self,
        user_id: int,
        system_prompt: str = "",
    ) -> list[dict]:
        """Get context as chat messages (for direct use with chat template).

        Returns messages in [{"role": "...", "content": "..."}] format.
        """
        mgr = self._get_context_manager(user_id)
        snapshot = mgr.get_context(max_tokens=self._target_tokens)
        messages = snapshot.to_messages()

        # Prepend system prompt if provided
        if system_prompt:
            # Check if we have a cold summary to include
            if snapshot.cold_summary:
                combined = f"{system_prompt}\n\nPrior context: {snapshot.cold_summary}"
                messages.insert(0, {"role": "system", "content": combined})
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})
        elif snapshot.cold_summary:
            # No system prompt, but we have cold summary
            messages.insert(0, {"role": "system", "content": f"Prior context: {snapshot.cold_summary}"})

        return messages

    def save_turn(
        self,
        user_id: int,
        user_msg: str,
        assistant_msg: str,
        *,
        skill: str = "",
        is_crisis: bool = False,
        model_key: str = "",
    ) -> None:
        """Save a conversation turn to both tiered context and session store."""
        # Save to session store (persistence)
        self._session_store.save_turn(
            user_id=user_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            skill=skill,
            is_crisis=is_crisis,
            model_key=model_key,
        )

        # Add to tiered context manager
        mgr = self._get_context_manager(user_id)
        mgr.add_turn("user", user_msg, is_crisis=is_crisis)
        mgr.add_turn("assistant", assistant_msg, is_crisis=is_crisis)

        # Check if compaction needed
        snapshot = mgr.get_context()
        if snapshot.truncated:
            logger.info("User %d: Context truncated, triggering compaction", user_id)
            self._trigger_compaction(user_id)

    def end_session(
        self,
        user_id: int,
        persist_memory: bool = True,
    ) -> None:
        """End a session, optionally persisting compaction to memory.

        Call this on /clear, /start, or session timeout.
        """
        if user_id not in self._user_contexts:
            return

        mgr = self._user_contexts[user_id]

        if persist_memory:
            # Get full history for compaction
            snapshot = mgr.get_context(max_tokens=10000)  # Get all
            history_pairs = snapshot.to_history_pairs()

            if history_pairs:
                # Compact for memory storage
                crisis_indices = self._get_crisis_indices(user_id)
                result = self._compactor.compact(
                    history=history_pairs,
                    target_tokens=200,  # Short summary for memory
                    preserve_indices=crisis_indices,
                    preserve_recent=0,  # Summarize everything
                )

                # Extract key facts from crisis turns
                key_facts = []
                for msg in result.preserved_crisis:
                    content = msg.get("content", "")
                    if content:
                        # Extract first sentence as key fact
                        first_sentence = content.split(".")[0]
                        if len(first_sentence) < 100:
                            key_facts.append(first_sentence)

                # Save to memory store
                summary = result.session_summary or self._extract_summary(history_pairs)
                if summary:
                    self._memory_store.save_compaction(
                        user_id=user_id,
                        session_id=f"session-{user_id}-{len(history_pairs)}",
                        summary=summary,
                        crisis_turns=[m for m in result.preserved_crisis],
                        key_facts=key_facts[:5],  # Limit key facts
                        original_turn_count=len(history_pairs),
                    )
                    logger.info(
                        "User %d: Session saved to memory (%d turns → %d chars)",
                        user_id,
                        len(history_pairs),
                        len(summary),
                    )

        # Clear the tiered context
        del self._user_contexts[user_id]

    def clear_session(self, user_id: int) -> None:
        """Clear session without persisting to memory."""
        if user_id in self._user_contexts:
            del self._user_contexts[user_id]
        self._session_store.delete_session(user_id)

    def _trigger_compaction(self, user_id: int) -> None:
        """Trigger compaction when context overflows."""
        mgr = self._user_contexts.get(user_id)
        if not mgr:
            return

        snapshot = mgr.get_context(max_tokens=10000)
        history_pairs = snapshot.to_history_pairs()

        if not history_pairs:
            return

        crisis_indices = self._get_crisis_indices(user_id)
        result = self._compactor.compact(
            history=history_pairs,
            target_tokens=self._target_tokens,
            preserve_indices=crisis_indices,
            preserve_recent=self._hot_size,
        )

        logger.info(
            "User %d: Compaction applied layers=%s reduction=%.1f%%",
            user_id,
            result.layers_applied,
            result.token_reduction * 100,
        )

    def _get_crisis_indices(self, user_id: int) -> set[int]:
        """Get crisis turn indices for a user."""
        mgr = self._user_contexts.get(user_id)
        if not mgr:
            return set()

        # Find crisis turns
        crisis_indices = set()
        pair_idx = 0
        for i in range(0, len(mgr._turns), 2):
            if i + 1 < len(mgr._turns):
                if mgr._turns[i].is_crisis or mgr._turns[i + 1].is_crisis:
                    crisis_indices.add(pair_idx)
                pair_idx += 1

        return crisis_indices

    def _messages_to_pairs(self, messages: list[dict]) -> list[tuple[str, str]]:
        """Convert chat messages back to (user, assistant) pairs."""
        pairs = []
        i = 0
        while i < len(messages) - 1:
            if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                pairs.append((messages[i]["content"], messages[i + 1]["content"]))
                i += 2
            else:
                i += 1
        return pairs

    def _extract_summary(self, history: list[tuple[str, str]]) -> str:
        """Extract a simple summary from history when LLM summarizer unavailable."""
        if not history:
            return ""

        # Take first sentence from each user message
        summaries = []
        for user_msg, _ in history[:5]:  # First 5 turns
            first_sentence = user_msg.split(".")[0].strip()
            if first_sentence:
                summaries.append(f"User: {first_sentence}")

        return ". ".join(summaries)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_user_key_facts(self, user_id: int, limit: int = 10) -> list[str]:
        """Get key facts about a user from memory."""
        return self._memory_store.get_key_facts(user_id, limit=limit)

    def recall_relevant_memories(
        self,
        user_id: int,
        context: str,
        limit: int = 3,
    ) -> list[str]:
        """Recall relevant past session summaries."""
        records = self._memory_store.recall_relevant(
            user_id=user_id,
            current_context=context,
            top_k=limit,
            min_similarity=0.2,
        )
        return [r.summary for r in records]
