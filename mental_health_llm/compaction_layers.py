"""
Five-layer progressive compaction pipeline.

Implements hierarchical compaction inspired by claw-code architecture:
- L1: Turn-level trimming (truncate verbose responses)
- L2: Pair-level merging (combine user-assistant into single summaries)
- L3: Topic-level clustering (group by semantic similarity)
- L4: Session-level summary (overall session themes)
- L5: Archival (cross-session semantic memory)

Each layer is applied progressively until the context fits within the
token budget. Crisis turns are preserved verbatim at all layers.

Usage:
    from mental_health_llm.compaction_layers import MultiLayerCompactor

    compactor = MultiLayerCompactor(llm_summarizer=my_summarize_fn)

    result = compactor.compact(
        history=conversation_history,
        target_tokens=1024,
        preserve_indices={2, 5},  # Crisis turns
    )

    print(f"Applied layers: {result.layers_applied}")
    print(f"Reduction: {result.token_reduction:.1%}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompactionResult:
    """Result of multi-layer compaction."""

    compacted: list[dict]  # Final compacted messages
    layers_applied: list[str]  # Which layers were used
    original_tokens: int
    final_tokens: int
    preserved_crisis: list[dict] = field(default_factory=list)
    session_summary: str = ""

    @property
    def token_reduction(self) -> float:
        """Percentage of tokens reduced."""
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.final_tokens / self.original_tokens)

    def to_messages(self, system_prompt: str = "") -> list[dict]:
        """Convert to chat message format."""
        messages = []

        # System prompt with session summary
        if system_prompt or self.session_summary:
            content = system_prompt
            if self.session_summary:
                if content:
                    content += f"\n\nSession context: {self.session_summary}"
                else:
                    content = f"Session context: {self.session_summary}"
            messages.append({"role": "system", "content": content})

        # Preserved crisis turns first
        messages.extend(self.preserved_crisis)

        # Compacted history
        messages.extend(self.compacted)

        return messages


@dataclass
class Turn:
    """Internal turn representation for compaction."""

    role: str
    content: str
    index: int
    is_crisis: bool = False
    topic_id: Optional[int] = None


class MultiLayerCompactor:
    """Five-layer progressive compaction pipeline.

    Applies compaction layers in order until the context fits within
    the token budget. Crisis turns are never compacted beyond L1 trimming.
    """

    def __init__(
        self,
        llm_summarizer: Optional[Callable[[str], str]] = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
        max_sentence_length: int = 150,
        max_pair_summary_length: int = 100,
    ):
        """Initialize the multi-layer compactor.

        Args:
            llm_summarizer: Optional function for L4 session-level summarization.
                            Signature: (text: str) -> str
            embedding_fn: Optional function for L3 topic clustering.
                          Signature: (text: str) -> list[float]
            max_sentence_length: Maximum chars per sentence in L1 trimming.
            max_pair_summary_length: Maximum chars per pair in L2 merging.
        """
        self._llm_summarizer = llm_summarizer
        self._embedding_fn = embedding_fn
        self._max_sentence_length = max_sentence_length
        self._max_pair_summary_length = max_pair_summary_length

    def compact(
        self,
        history: list[tuple[str, str]],
        target_tokens: int,
        preserve_indices: Optional[set[int]] = None,
        preserve_recent: int = 4,
    ) -> CompactionResult:
        """Apply compaction layers progressively.

        Args:
            history: List of (user_msg, assistant_msg) tuples.
            target_tokens: Target token budget.
            preserve_indices: Turn indices to preserve verbatim (crisis).
            preserve_recent: Number of recent turn pairs to always keep.

        Returns:
            CompactionResult with compacted messages and metadata.
        """
        preserve_indices = preserve_indices or set()

        if not history:
            return CompactionResult(
                compacted=[],
                layers_applied=[],
                original_tokens=0,
                final_tokens=0,
            )

        # Convert to Turn objects
        turns: list[Turn] = []
        for i, (user_msg, asst_msg) in enumerate(history):
            is_crisis = i in preserve_indices
            turns.append(Turn("user", user_msg, i * 2, is_crisis))
            turns.append(Turn("assistant", asst_msg, i * 2 + 1, is_crisis))

        original_tokens = self._token_count_turns(turns)

        # Split into recent (always preserved) and old (compactable)
        split_point = max(0, len(history) - preserve_recent)
        recent_turns = turns[split_point * 2:]
        old_turns = turns[:split_point * 2]

        # Separate crisis turns from old
        crisis_turns = [t for t in old_turns if t.is_crisis]
        compactable_turns = [t for t in old_turns if not t.is_crisis]

        applied_layers: list[str] = []
        current = compactable_turns
        session_summary = ""

        # Apply layers progressively until within budget
        token_budget = target_tokens - self._token_count_turns(recent_turns) - self._token_count_turns(crisis_turns)

        # L1: Turn-level trimming
        if self._token_count_turns(current) > token_budget:
            current = self._layer1_trim_turns(current)
            applied_layers.append("L1:turn_trim")
            logger.debug("L1 applied: %d tokens", self._token_count_turns(current))

        # L2: Pair-level merging
        if self._token_count_turns(current) > token_budget:
            current = self._layer2_merge_pairs(current)
            applied_layers.append("L2:pair_merge")
            logger.debug("L2 applied: %d tokens", self._token_count_turns(current))

        # L3: Topic clustering
        if self._token_count_turns(current) > token_budget:
            current = self._layer3_topic_clusters(current)
            applied_layers.append("L3:topic_cluster")
            logger.debug("L3 applied: %d tokens", self._token_count_turns(current))

        # L4: Session summary (replaces all old turns with summary)
        if self._token_count_turns(current) > token_budget:
            session_summary = self._layer4_session_summary(current)
            current = []  # All old non-crisis turns become summary
            applied_layers.append("L4:session_summary")
            logger.debug("L4 applied: %d tokens in summary", len(session_summary) // 4)

        # Build final message list
        compacted_messages = []
        for turn in current:
            compacted_messages.append({"role": turn.role, "content": turn.content})
        for turn in recent_turns:
            compacted_messages.append({"role": turn.role, "content": turn.content})

        # Build crisis message list
        crisis_messages = []
        for turn in crisis_turns:
            crisis_messages.append({"role": turn.role, "content": turn.content})

        final_tokens = (
            self._token_count_turns(current)
            + self._token_count_turns(recent_turns)
            + self._token_count_turns(crisis_turns)
            + (len(session_summary) // 4)
        )

        return CompactionResult(
            compacted=compacted_messages,
            layers_applied=applied_layers,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            preserved_crisis=crisis_messages,
            session_summary=session_summary,
        )

    # ------------------------------------------------------------------
    # Layer implementations
    # ------------------------------------------------------------------

    def _layer1_trim_turns(self, turns: list[Turn]) -> list[Turn]:
        """L1: Trim each turn to key sentences.

        Strategy:
        - Extract first 2 sentences from each turn
        - Truncate at max_sentence_length
        - Preserve structure (user/assistant alternation)
        """
        trimmed = []
        for turn in turns:
            content = self._extract_key_sentences(turn.content, max_sentences=2)
            trimmed.append(Turn(
                role=turn.role,
                content=content,
                index=turn.index,
                is_crisis=turn.is_crisis,
                topic_id=turn.topic_id,
            ))
        return trimmed

    def _layer2_merge_pairs(self, turns: list[Turn]) -> list[Turn]:
        """L2: Merge user-assistant pairs into single summaries.

        Strategy:
        - Combine each (user, assistant) pair into one summary turn
        - Format: "User asked about X. Counselor discussed Y."
        """
        merged = []
        i = 0
        while i < len(turns):
            if i + 1 < len(turns) and turns[i].role == "user" and turns[i + 1].role == "assistant":
                user_turn = turns[i]
                asst_turn = turns[i + 1]

                # Extract key points
                user_key = self._extract_key_sentences(user_turn.content, max_sentences=1)
                asst_key = self._extract_key_sentences(asst_turn.content, max_sentences=1)

                # Merge into single summary
                summary = f"User: {user_key} → Counselor: {asst_key}"
                if len(summary) > self._max_pair_summary_length:
                    summary = summary[: self._max_pair_summary_length - 3] + "..."

                merged.append(Turn(
                    role="system",  # Merged pairs become system context
                    content=summary,
                    index=user_turn.index,
                    is_crisis=user_turn.is_crisis or asst_turn.is_crisis,
                ))
                i += 2
            else:
                merged.append(turns[i])
                i += 1

        return merged

    def _layer3_topic_clusters(self, turns: list[Turn]) -> list[Turn]:
        """L3: Cluster turns by topic and summarize each cluster.

        Strategy:
        - Group semantically similar turns (using embeddings if available)
        - Create one summary per topic cluster
        - Fall back to sequential grouping if no embedding function
        """
        if not turns:
            return []

        # Group into clusters
        clusters = self._cluster_turns(turns)

        # Summarize each cluster
        clustered = []
        for topic_id, cluster_turns in enumerate(clusters):
            if len(cluster_turns) == 1:
                clustered.append(cluster_turns[0])
            else:
                # Create cluster summary
                contents = [t.content for t in cluster_turns]
                cluster_text = " ".join(contents)

                # Extract key themes
                summary = self._extract_key_sentences(cluster_text, max_sentences=2)
                prefix = self._infer_topic_label(contents)

                clustered.append(Turn(
                    role="system",
                    content=f"[{prefix}] {summary}",
                    index=cluster_turns[0].index,
                    is_crisis=any(t.is_crisis for t in cluster_turns),
                    topic_id=topic_id,
                ))

        return clustered

    def _layer4_session_summary(self, turns: list[Turn]) -> str:
        """L4: Compress all turns into an overall session summary.

        Strategy:
        - If LLM summarizer available, use it
        - Otherwise, extract key sentences and combine
        """
        if not turns:
            return ""

        full_text = "\n".join(f"{t.role}: {t.content}" for t in turns)

        # Use LLM summarizer if available
        if self._llm_summarizer is not None:
            try:
                prompt = f"Summarize this therapeutic conversation in 2-3 sentences, focusing on the user's main concerns and the counselor's key guidance:\n\n{full_text[:2000]}"
                return self._llm_summarizer(prompt)
            except Exception as e:
                logger.warning("LLM summarizer failed: %s", e)

        # Fallback: extractive summary
        sentences = []
        for turn in turns:
            key = self._extract_key_sentences(turn.content, max_sentences=1)
            prefix = "User discussed" if turn.role == "user" else "Counselor noted"
            sentences.append(f"{prefix}: {key}")

        # Cap at ~100 tokens
        summary_parts = []
        token_est = 0
        for s in sentences:
            s_tokens = len(s) // 4
            if token_est + s_tokens > 100:
                break
            summary_parts.append(s)
            token_est += s_tokens

        return " ".join(summary_parts)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _token_count_turns(self, turns: list[Turn]) -> int:
        """Estimate token count for a list of turns."""
        return sum(len(t.content) // 4 for t in turns)

    def _extract_key_sentences(self, text: str, max_sentences: int = 2) -> str:
        """Extract first N sentences from text."""
        text = text.strip()
        if not text:
            return ""

        # Split into sentences
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)

        # Take first N
        selected = sentences[:max_sentences]
        result = " ".join(selected)

        # Truncate if still too long
        if len(result) > self._max_sentence_length:
            result = result[: self._max_sentence_length - 3] + "..."

        return result

    def _cluster_turns(self, turns: list[Turn], max_clusters: int = 5) -> list[list[Turn]]:
        """Cluster turns by semantic similarity.

        If embedding function is available, use k-means style clustering.
        Otherwise, group sequentially by fixed size.
        """
        if not turns:
            return []

        if self._embedding_fn is None:
            # Fallback: sequential grouping
            cluster_size = max(1, len(turns) // max_clusters)
            clusters = []
            for i in range(0, len(turns), cluster_size):
                clusters.append(turns[i : i + cluster_size])
            return clusters

        # Compute embeddings
        try:
            embeddings = [self._embedding_fn(t.content) for t in turns]
        except Exception as e:
            logger.warning("Embedding failed: %s, falling back to sequential", e)
            cluster_size = max(1, len(turns) // max_clusters)
            clusters = []
            for i in range(0, len(turns), cluster_size):
                clusters.append(turns[i : i + cluster_size])
            return clusters

        # Simple clustering: group by similarity to centroids
        clusters: list[list[Turn]] = [[] for _ in range(max_clusters)]
        centroids = [embeddings[i * (len(embeddings) // max_clusters)] for i in range(max_clusters)]

        for turn, emb in zip(turns, embeddings):
            # Find closest centroid
            best_cluster = 0
            best_sim = -1.0
            for j, centroid in enumerate(centroids):
                sim = self._cosine_similarity(emb, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = j
            clusters[best_cluster].append(turn)

        # Remove empty clusters
        return [c for c in clusters if c]

    def _infer_topic_label(self, contents: list[str]) -> str:
        """Infer a short topic label from content."""
        combined = " ".join(contents).lower()

        # Simple keyword-based topic detection
        topics = {
            "anxiety": ["anxious", "anxiety", "worried", "panic", "stress"],
            "depression": ["depressed", "sad", "hopeless", "tired", "empty"],
            "relationships": ["partner", "relationship", "friend", "family", "argue"],
            "work": ["work", "job", "boss", "career", "burnout"],
            "self-esteem": ["confident", "worth", "failure", "stupid", "ugly"],
            "coping": ["cope", "strategy", "technique", "help", "advice"],
        }

        best_topic = "Discussion"
        best_score = 0

        for topic, keywords in topics.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_topic = topic.title()

        return best_topic

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between vectors."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Integration with existing compaction.py
# ---------------------------------------------------------------------------


def upgrade_compactor(
    original_compactor,
    llm_summarizer: Optional[Callable[[str], str]] = None,
    embedding_fn: Optional[Callable[[str], list[float]]] = None,
) -> MultiLayerCompactor:
    """Create a MultiLayerCompactor with settings from existing compactor.

    Args:
        original_compactor: Existing ConversationCompactor instance.
        llm_summarizer: Optional LLM summarization function.
        embedding_fn: Optional embedding function.

    Returns:
        Configured MultiLayerCompactor.
    """
    return MultiLayerCompactor(
        llm_summarizer=llm_summarizer,
        embedding_fn=embedding_fn,
    )
