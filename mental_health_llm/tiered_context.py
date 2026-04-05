"""
Tiered context management with Hot/Warm/Cold hierarchy.

Implements dynamic context tiering inspired by claw-code architecture:
- Hot: Recent turns, always in context (verbatim)
- Warm: Important turns retained with detail (crisis, referenced)
- Cold: Old turns summarized/archived (retrievable)

Source pattern: claw-code context_manager.rs tier system.

Usage:
    from mental_health_llm.tiered_context import TieredContextManager

    manager = TieredContextManager(hot_size=4, warm_size=6)

    # Add turns as conversation progresses
    manager.add_turn({"role": "user", "content": "I feel anxious"})
    manager.add_turn({"role": "assistant", "content": "I hear you..."})

    # Mark important turns
    manager.promote_to_warm(turn_idx=2, reason="crisis")

    # Get context within token budget
    context = manager.get_context(max_tokens=1024)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Tier(Enum):
    """Context tier levels."""

    HOT = "hot"  # Full verbatim, always included
    WARM = "warm"  # Key sentences + metadata
    COLD = "cold"  # Summarized, retrievable on demand


@dataclass
class TieredTurn:
    """A conversation turn with tier metadata."""

    role: str  # "user" or "assistant"
    content: str
    turn_index: int
    tier: Tier = Tier.HOT
    is_crisis: bool = False
    importance_score: float = 0.0
    promotion_reason: Optional[str] = None
    summary: Optional[str] = None  # Used when tier is COLD

    @property
    def display_content(self) -> str:
        """Get content appropriate for current tier."""
        if self.tier == Tier.COLD and self.summary:
            return self.summary
        return self.content

    def token_estimate(self) -> int:
        """Estimate token count using len/4 heuristic."""
        return len(self.display_content) // 4


@dataclass
class ContextSnapshot:
    """A snapshot of the assembled context."""

    hot_turns: list[TieredTurn] = field(default_factory=list)
    warm_turns: list[TieredTurn] = field(default_factory=list)
    cold_summary: str = ""
    total_tokens: int = 0
    truncated: bool = False

    def to_messages(self) -> list[dict]:
        """Convert to chat message format."""
        messages = []

        # Cold summary as system context
        if self.cold_summary:
            messages.append({
                "role": "system",
                "content": f"Earlier context: {self.cold_summary}",
            })

        # Warm turns (important older turns)
        for turn in self.warm_turns:
            messages.append({
                "role": turn.role,
                "content": turn.display_content,
            })

        # Hot turns (recent verbatim)
        for turn in self.hot_turns:
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })

        return messages

    def to_history_pairs(self) -> list[tuple[str, str]]:
        """Convert to (user, assistant) pairs."""
        all_turns = self.warm_turns + self.hot_turns
        pairs = []
        i = 0
        while i < len(all_turns) - 1:
            if all_turns[i].role == "user" and all_turns[i + 1].role == "assistant":
                pairs.append((all_turns[i].display_content, all_turns[i + 1].display_content))
                i += 2
            else:
                i += 1
        return pairs


class TieredContextManager:
    """Dynamic Hot/Warm/Cold context management.

    Manages conversation context with automatic tier transitions:
    - New turns enter as HOT
    - As HOT overflows, turns move to WARM (if important) or COLD
    - WARM overflow moves to COLD with summarization
    - COLD turns are summarized and can be retrieved on demand
    """

    def __init__(
        self,
        hot_size: int = 4,
        warm_size: int = 6,
        cold_threshold: int = 20,
        importance_threshold: float = 0.5,
    ):
        """Initialize the tiered context manager.

        Args:
            hot_size: Maximum turns in HOT tier (always verbatim).
            warm_size: Maximum turns in WARM tier (important, detailed).
            cold_threshold: Archive cold turns after this total count.
            importance_threshold: Score threshold for auto-promotion to WARM.
        """
        self.hot_size = hot_size
        self.warm_size = warm_size
        self.cold_threshold = cold_threshold
        self.importance_threshold = importance_threshold

        self._turns: list[TieredTurn] = []
        self._cold_archive: list[TieredTurn] = []
        self._cold_summary: str = ""

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_turn(
        self,
        role: str,
        content: str,
        is_crisis: bool = False,
        importance_score: float = 0.0,
    ) -> TieredTurn:
        """Add a new turn to the context.

        New turns always start in HOT tier. Crisis turns are auto-promoted
        to WARM when they would otherwise be demoted.

        Args:
            role: "user" or "assistant"
            content: Turn content
            is_crisis: Whether this is a crisis turn (never fully demoted)
            importance_score: Importance score for tier decisions

        Returns:
            The created TieredTurn
        """
        turn = TieredTurn(
            role=role,
            content=content,
            turn_index=len(self._turns),
            tier=Tier.HOT,
            is_crisis=is_crisis,
            importance_score=importance_score,
        )

        self._turns.append(turn)
        self._rebalance()

        return turn

    def add_turn_pair(
        self,
        user_msg: str,
        assistant_msg: str,
        is_crisis: bool = False,
    ) -> tuple[TieredTurn, TieredTurn]:
        """Add a user-assistant turn pair."""
        user_turn = self.add_turn("user", user_msg, is_crisis=is_crisis)
        asst_turn = self.add_turn("assistant", assistant_msg, is_crisis=is_crisis)
        return user_turn, asst_turn

    def promote_to_warm(self, turn_idx: int, reason: str = "referenced") -> bool:
        """Promote a turn to WARM tier.

        Use this when a cold/archived turn is referenced again in conversation.

        Args:
            turn_idx: Index of the turn to promote
            reason: Why the turn is being promoted

        Returns:
            True if promotion succeeded
        """
        # Check main turns
        for turn in self._turns:
            if turn.turn_index == turn_idx:
                if turn.tier != Tier.HOT:
                    turn.tier = Tier.WARM
                    turn.promotion_reason = reason
                    turn.importance_score = max(turn.importance_score, self.importance_threshold)
                    logger.debug("Promoted turn %d to WARM: %s", turn_idx, reason)
                return True

        # Check cold archive
        for i, turn in enumerate(self._cold_archive):
            if turn.turn_index == turn_idx:
                turn.tier = Tier.WARM
                turn.promotion_reason = reason
                # Move back to main turns
                self._turns.append(turn)
                self._cold_archive.pop(i)
                self._rebalance()
                logger.debug("Restored turn %d from COLD to WARM: %s", turn_idx, reason)
                return True

        return False

    def mark_crisis(self, turn_idx: int) -> bool:
        """Mark a turn as crisis-related (never fully demoted)."""
        for turn in self._turns:
            if turn.turn_index == turn_idx:
                turn.is_crisis = True
                if turn.tier == Tier.COLD:
                    turn.tier = Tier.WARM
                    turn.promotion_reason = "crisis"
                return True
        return False

    # ------------------------------------------------------------------
    # Tier rebalancing
    # ------------------------------------------------------------------

    def _rebalance(self) -> None:
        """Rebalance turns across tiers based on configuration."""
        if not self._turns:
            return

        # Sort by turn_index to maintain order
        self._turns.sort(key=lambda t: t.turn_index)

        # Identify tier boundaries
        total = len(self._turns)
        hot_start = max(0, total - self.hot_size * 2)  # *2 for pairs

        # Assign tiers based on position and importance
        for i, turn in enumerate(self._turns):
            if i >= hot_start:
                # Recent turns stay HOT
                turn.tier = Tier.HOT
            elif turn.is_crisis or turn.importance_score >= self.importance_threshold:
                # Important/crisis turns go to WARM
                turn.tier = Tier.WARM
            else:
                # Old unimportant turns go to COLD
                turn.tier = Tier.COLD

        # Enforce WARM size limit
        warm_turns = [t for t in self._turns if t.tier == Tier.WARM]
        if len(warm_turns) > self.warm_size * 2:
            # Sort WARM by importance, demote lowest
            warm_turns.sort(key=lambda t: (t.is_crisis, t.importance_score), reverse=True)
            for turn in warm_turns[self.warm_size * 2 :]:
                if not turn.is_crisis:  # Never demote crisis
                    turn.tier = Tier.COLD

        # Archive old COLD turns
        if total > self.cold_threshold:
            cold_turns = [t for t in self._turns if t.tier == Tier.COLD]
            archive_count = len(cold_turns) - (self.cold_threshold - self.hot_size * 2 - self.warm_size * 2)

            if archive_count > 0:
                # Archive oldest COLD turns
                to_archive = cold_turns[:archive_count]
                for turn in to_archive:
                    turn.summary = self._summarize_turn(turn)
                    self._cold_archive.append(turn)
                    self._turns.remove(turn)

                # Rebuild cold summary
                self._rebuild_cold_summary()

    def _summarize_turn(self, turn: TieredTurn) -> str:
        """Create a brief summary of a turn for COLD tier."""
        text = turn.content.strip()
        if not text:
            return ""

        # Extract first sentence
        first_sentence = text
        for sep in [".", "!", "?"]:
            idx = text.find(sep)
            if idx != -1 and idx < len(first_sentence):
                first_sentence = text[: idx + 1]

        # Truncate if needed
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:97] + "..."

        prefix = "User" if turn.role == "user" else "Counselor"
        return f"{prefix}: {first_sentence}"

    def _rebuild_cold_summary(self) -> None:
        """Rebuild the aggregated cold summary."""
        if not self._cold_archive:
            self._cold_summary = ""
            return

        summaries = []
        token_budget = 150  # ~150 tokens for cold summary
        current_tokens = 0

        for turn in reversed(self._cold_archive):  # Most recent first
            summary = turn.summary or self._summarize_turn(turn)
            tokens = len(summary) // 4
            if current_tokens + tokens > token_budget:
                break
            summaries.append(summary)
            current_tokens += tokens

        summaries.reverse()  # Back to chronological order
        self._cold_summary = " ".join(summaries)

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------

    def get_context(self, max_tokens: int = 2048) -> ContextSnapshot:
        """Assemble context from all tiers within token budget.

        Args:
            max_tokens: Maximum tokens for the assembled context.

        Returns:
            ContextSnapshot with turns organized by tier.
        """
        snapshot = ContextSnapshot()
        remaining_tokens = max_tokens

        # Always include cold summary if available
        if self._cold_summary:
            cold_tokens = len(self._cold_summary) // 4
            if cold_tokens < remaining_tokens:
                snapshot.cold_summary = self._cold_summary
                remaining_tokens -= cold_tokens

        # Collect turns by tier
        hot_turns = [t for t in self._turns if t.tier == Tier.HOT]
        warm_turns = [t for t in self._turns if t.tier == Tier.WARM]
        cold_turns = [t for t in self._turns if t.tier == Tier.COLD]

        # HOT turns are always included (most recent, required for coherence)
        for turn in hot_turns:
            tokens = turn.token_estimate()
            if tokens < remaining_tokens:
                snapshot.hot_turns.append(turn)
                remaining_tokens -= tokens
            else:
                snapshot.truncated = True

        # WARM turns (important older turns)
        for turn in sorted(warm_turns, key=lambda t: t.turn_index):
            tokens = turn.token_estimate()
            if tokens < remaining_tokens:
                snapshot.warm_turns.append(turn)
                remaining_tokens -= tokens
            else:
                snapshot.truncated = True

        # COLD turns only if space remains (use summaries)
        for turn in sorted(cold_turns, key=lambda t: t.turn_index):
            summary = turn.summary or self._summarize_turn(turn)
            tokens = len(summary) // 4
            if tokens < remaining_tokens:
                turn.summary = summary
                # Include as extension of cold_summary
                if snapshot.cold_summary:
                    snapshot.cold_summary += f" {summary}"
                else:
                    snapshot.cold_summary = summary
                remaining_tokens -= tokens
            else:
                snapshot.truncated = True

        snapshot.total_tokens = max_tokens - remaining_tokens
        return snapshot

    def get_full_history(self) -> list[TieredTurn]:
        """Get all turns including archived ones."""
        return self._cold_archive + self._turns

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all context."""
        self._turns.clear()
        self._cold_archive.clear()
        self._cold_summary = ""

    def get_stats(self) -> dict:
        """Get tier statistics."""
        hot = [t for t in self._turns if t.tier == Tier.HOT]
        warm = [t for t in self._turns if t.tier == Tier.WARM]
        cold = [t for t in self._turns if t.tier == Tier.COLD]

        return {
            "total_turns": len(self._turns) + len(self._cold_archive),
            "hot_count": len(hot),
            "warm_count": len(warm),
            "cold_count": len(cold),
            "archived_count": len(self._cold_archive),
            "cold_summary_tokens": len(self._cold_summary) // 4,
            "crisis_turns": sum(1 for t in self._turns if t.is_crisis),
        }

    def restore_from_pairs(
        self,
        history: list[tuple[str, str]],
        crisis_indices: set[int],
    ) -> None:
        """Restore context from (user, assistant) pairs.

        Used for session restoration from session_store.

        Args:
            history: List of (user_msg, assistant_msg) tuples.
            crisis_indices: Set of turn indices where crisis was detected.
        """
        self.clear()
        for i, (user_msg, asst_msg) in enumerate(history):
            is_crisis = i in crisis_indices
            self.add_turn_pair(user_msg, asst_msg, is_crisis=is_crisis)
