"""
Binary crisis/non-crisis pre-classifier.

Runs BEFORE general skill routing to ensure crisis messages are never missed.
Uses a dual-signal approach: embedding cosine similarity AND keyword regex
matching. If either signal triggers, the message is routed to crisis-intervention.

Requires: sentence-transformers (optional — gate degrades to keyword-only if
unavailable).
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


class CrisisGate:
    """Binary crisis detector using embedding similarity + keyword matching."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        threshold: float = 0.45,
        keyword_boost: bool = True,
        crisis_centroid: Optional[np.ndarray] = None,
        encode_fn=None,
        history_window: int = 3,
    ):
        """
        Args:
            config_path: Path to skills_config.json for loading crisis keywords/patterns.
            threshold: Cosine similarity threshold for embedding-based detection.
            keyword_boost: If True, keyword matches also trigger crisis detection.
            crisis_centroid: Pre-computed crisis centroid embedding (unit-norm).
            encode_fn: Function that encodes text to a unit-norm embedding vector.
            history_window: Number of recent turns to consider for escalation.
        """
        self.threshold = threshold
        self.keyword_boost = keyword_boost
        self.crisis_centroid = crisis_centroid
        self._encode = encode_fn
        self.history_window = history_window

        # Load crisis keywords and patterns from config
        if config_path is None:
            config_path = str(Path(__file__).parent / "skills_config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        self._keywords = []
        self._patterns = []

        for skill_def in config["skills"]:
            if skill_def["name"] == "crisis-intervention":
                self._keywords = [kw.lower() for kw in skill_def.get("keywords", [])]
                for p in skill_def.get("patterns", []):
                    try:
                        self._patterns.append(re.compile(p, re.IGNORECASE))
                    except re.error:
                        pass
                break

    def check(
        self, message: str, history: Optional[list] = None
    ) -> dict:
        """
        Check whether a message indicates crisis.

        Args:
            message: Current user message.
            history: Optional list of (user_msg, assistant_msg) tuples.

        Returns:
            Dict with keys:
                - is_crisis (bool): Whether crisis was detected.
                - keyword_triggered (bool): Whether keywords/patterns matched.
                - embedding_triggered (bool): Whether embedding exceeded threshold.
                - embedding_score (float): Cosine similarity to crisis centroid.
                - escalation_detected (bool): Whether history shows escalation.
                - effective_threshold (float): Threshold used (may be lowered by escalation).
        """
        result = {
            "is_crisis": False,
            "keyword_triggered": False,
            "embedding_triggered": False,
            "embedding_score": 0.0,
            "escalation_detected": False,
            "effective_threshold": self.threshold,
        }

        # --- Signal 1: Keyword/pattern matching ---
        if self.keyword_boost:
            msg_lower = message.lower()
            for kw in self._keywords:
                if kw in msg_lower:
                    result["keyword_triggered"] = True
                    break

            if not result["keyword_triggered"]:
                for pattern in self._patterns:
                    if pattern.search(message):
                        result["keyword_triggered"] = True
                        break

        # --- Signal 2: Embedding similarity ---
        effective_threshold = self.threshold

        if self._encode is not None and self.crisis_centroid is not None:
            msg_emb = self._encode(message)
            score = float(np.dot(msg_emb, self.crisis_centroid))
            result["embedding_score"] = score

            # History escalation: check if recent turns trend toward crisis
            if history and len(history) > 0:
                recent = history[-self.history_window:]
                history_scores = []
                for user_msg, _ in recent:
                    h_emb = self._encode(user_msg)
                    history_scores.append(float(np.dot(h_emb, self.crisis_centroid)))

                if len(history_scores) >= 2:
                    # Check for increasing trend toward crisis centroid
                    diffs = [
                        history_scores[i + 1] - history_scores[i]
                        for i in range(len(history_scores) - 1)
                    ]
                    avg_increase = sum(diffs) / len(diffs)
                    if avg_increase > 0.02:  # consistent upward trend
                        result["escalation_detected"] = True
                        # Lower threshold by 15% to catch borderline cases
                        effective_threshold = self.threshold * 0.85
                        result["effective_threshold"] = effective_threshold

            if score >= effective_threshold:
                result["embedding_triggered"] = True

        # --- Final decision: either signal triggers crisis ---
        result["is_crisis"] = result["keyword_triggered"] or result["embedding_triggered"]

        return result
