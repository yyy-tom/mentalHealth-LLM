"""
Conversation compaction for mental health counseling sessions.

Replaces the naive pop(0) history trimming with extractive summarization
that preserves crisis-flagged turns verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    """A single conversation turn."""

    role: str  # "user" or "assistant"
    content: str
    is_crisis: bool = False
    turn_index: int = 0


@dataclass
class CompactedHistory:
    """Result of compacting a conversation history."""

    summary: str  # Extractive summary of old turns
    crisis_turns: list[Turn] = field(default_factory=list)  # Never compacted
    recent_turns: list[Turn] = field(default_factory=list)  # Last N preserved verbatim
    total_original_turns: int = 0
    was_compacted: bool = False

    def to_messages(self) -> list[dict]:
        """Convert to chat message format for tokenizer.apply_chat_template()."""
        messages: list[dict] = []
        if self.summary:
            messages.append(
                {"role": "system", "content": f"Session context: {self.summary}"}
            )
        for turn in self.crisis_turns:
            messages.append({"role": turn.role, "content": turn.content})
        for turn in self.recent_turns:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def to_history_pairs(self) -> list[tuple[str, str]]:
        """Convert back to (user_msg, assistant_msg) pairs for generate_response()."""
        all_turns = self.crisis_turns + self.recent_turns
        pairs: list[tuple[str, str]] = []
        i = 0
        while i < len(all_turns) - 1:
            if all_turns[i].role == "user" and all_turns[i + 1].role == "assistant":
                pairs.append((all_turns[i].content, all_turns[i + 1].content))
                i += 2
            else:
                i += 1
        return pairs


class ConversationCompactor:
    """Compacts conversation history while preserving crisis turns.

    Instead of hard-dropping old turns with ``pop(0)``, this extracts key
    sentences from older turns into a summary and always keeps crisis-flagged
    turns verbatim.
    """

    def __init__(
        self,
        max_tokens: int = 768,
        trigger_threshold: int = 600,
        preserve_recent: int = 4,
    ):
        self.max_tokens = max_tokens
        self.trigger_threshold = trigger_threshold
        self.preserve_recent = preserve_recent

    def estimate_tokens(self, history: list[tuple[str, str]], system_prompt: str = "") -> int:
        """Estimate token count using len(text)//4 heuristic."""
        total = len(system_prompt) // 4
        for user_msg, assistant_msg in history:
            total += len(user_msg) // 4 + len(assistant_msg) // 4
        return total

    def compact(
        self,
        history: list[tuple[str, str]],
        crisis_turn_indices: set[int],
    ) -> CompactedHistory:
        """Compact conversation history, preserving crisis turns.

        Args:
            history: List of (user_msg, assistant_msg) tuples.
            crisis_turn_indices: Set of turn indices where crisis was detected.

        Returns:
            CompactedHistory with summary, crisis turns, and recent turns.
        """
        total_turns = len(history)

        if total_turns == 0:
            return CompactedHistory(
                summary="",
                total_original_turns=0,
                was_compacted=False,
            )

        # Check if compaction is needed
        if self.estimate_tokens(history) < self.trigger_threshold:
            # No compaction needed — return all turns as recent
            recent = []
            for i, (user_msg, asst_msg) in enumerate(history):
                is_crisis = i in crisis_turn_indices
                recent.append(Turn("user", user_msg, is_crisis=is_crisis, turn_index=i * 2))
                recent.append(Turn("assistant", asst_msg, is_crisis=is_crisis, turn_index=i * 2 + 1))
            return CompactedHistory(
                summary="",
                recent_turns=recent,
                total_original_turns=total_turns,
                was_compacted=False,
            )

        # Split into old and recent
        split_point = max(0, total_turns - self.preserve_recent)

        # Build all turns as Turn objects
        all_turns: list[Turn] = []
        for i, (user_msg, asst_msg) in enumerate(history):
            is_crisis = i in crisis_turn_indices
            all_turns.append(Turn("user", user_msg, is_crisis=is_crisis, turn_index=i * 2))
            all_turns.append(Turn("assistant", asst_msg, is_crisis=is_crisis, turn_index=i * 2 + 1))

        old_turns = all_turns[: split_point * 2]
        recent_turns = all_turns[split_point * 2 :]

        # Extract crisis turns from old turns (these are never summarized)
        crisis_turns: list[Turn] = []
        non_crisis_old: list[Turn] = []
        for turn in old_turns:
            if turn.is_crisis:
                crisis_turns.append(turn)
            else:
                non_crisis_old.append(turn)

        # Deduplicate crisis turns that are also in recent
        recent_indices = {t.turn_index for t in recent_turns}
        crisis_turns = [t for t in crisis_turns if t.turn_index not in recent_indices]

        # Summarize non-crisis old turns
        summary = self._extractive_summarize(non_crisis_old)

        return CompactedHistory(
            summary=summary,
            crisis_turns=crisis_turns,
            recent_turns=recent_turns,
            total_original_turns=total_turns,
            was_compacted=True,
        )

    def _extractive_summarize(self, turns: list[Turn]) -> str:
        """Extract key sentences from old turns to create a summary.

        Takes the first sentence of each turn and combines them into a
        compact narrative prefixed with 'Earlier in conversation:'.
        """
        if not turns:
            return ""

        sentences: list[str] = []
        for turn in turns:
            text = turn.content.strip()
            if not text:
                continue
            # Extract first sentence
            first_sentence = text
            for sep in [".", "!", "?"]:
                idx = text.find(sep)
                if idx != -1 and idx < len(first_sentence):
                    first_sentence = text[: idx + 1]
            # Keep it concise
            if len(first_sentence) > 150:
                first_sentence = first_sentence[:147] + "..."
            prefix = "User mentioned" if turn.role == "user" else "Counselor discussed"
            sentences.append(f"{prefix}: {first_sentence}")

        if not sentences:
            return ""

        # Cap at ~100 tokens worth
        summary_parts: list[str] = []
        token_est = 0
        for s in sentences:
            s_tokens = len(s) // 4
            if token_est + s_tokens > 100:
                break
            summary_parts.append(s)
            token_est += s_tokens

        return "Earlier in conversation: " + " ".join(summary_parts)
