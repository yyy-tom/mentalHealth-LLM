"""
Embedding-based skill router using sentence-transformers.

Routes user messages by encoding them with all-MiniLM-L6-v2 (~5ms CPU) and
comparing cosine similarity to pre-computed skill centroids. Includes a
CrisisGate pre-classifier for safety-critical messages and optional
conversation-aware routing via history blending.

Requires: sentence-transformers>=2.2.0
          pip install mental-health-llm[router]
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from mental_health_llm.crisis_gate import CrisisGate
from mental_health_llm.router_base import BaseRouter

logger = logging.getLogger(__name__)


class EmbeddingRouter(BaseRouter):
    """Routes messages to skills via embedding cosine similarity."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        centroid_path: Optional[str] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        history_window: int = 3,
        history_weight: float = 0.2,
        crisis_threshold: float = 0.45,
        crisis_keyword_boost: bool = True,
    ):
        """
        Args:
            config_path: Path to skills_config.json.
            centroid_path: Path to skill_centroids.npz. Defaults to
                           mental_health_llm/centroids/skill_centroids.npz.
            model_name: Sentence-transformer model name or path.
            history_window: Number of recent turns for history blending.
            history_weight: Weight of history in blended embedding (0-1).
            crisis_threshold: Cosine similarity threshold for crisis gate.
            crisis_keyword_boost: Whether crisis gate also uses keyword matching.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / "skills_config.json")

        if centroid_path is None:
            centroid_path = str(
                Path(__file__).parent / "centroids" / "skill_centroids.npz"
            )

        # Load skills config for system prompts, adapter paths, etc.
        with open(config_path, "r") as f:
            config = json.load(f)

        self.default_skill = config.get("default_skill", "general-support")

        self.skills = []
        for skill_def in config["skills"]:
            self.skills.append({
                "name": skill_def["name"],
                "description": skill_def.get("description", ""),
                "priority": skill_def.get("priority", 0),
                "adapter_path": skill_def.get("adapter_path", ""),
                "system_prompt": skill_def.get("system_prompt", ""),
            })
        self.skills.sort(key=lambda s: s["priority"], reverse=True)
        self._by_name = {s["name"]: s for s in self.skills}

        # Load sentence transformer
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._history_window = history_window
        self._history_weight = history_weight

        # Load centroids
        centroid_file = Path(centroid_path)
        if not centroid_file.exists():
            raise FileNotFoundError(
                f"Centroid file not found: {centroid_path}. "
                "Run `python scripts/build_skill_centroids.py` first."
            )

        data = np.load(centroid_path)
        self._skill_names_order = list(data["skill_names"])
        self._centroids = data["centroids"]  # shape: (n_skills, embed_dim)

        # Verify centroids are unit-norm
        norms = np.linalg.norm(self._centroids, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            warnings.warn("Centroids are not unit-normalized, normalizing now.")
            self._centroids = self._centroids / norms[:, np.newaxis]

        # Crisis centroid for the gate
        crisis_idx = (
            self._skill_names_order.index("crisis-intervention")
            if "crisis-intervention" in self._skill_names_order
            else None
        )
        crisis_centroid = self._centroids[crisis_idx] if crisis_idx is not None else None

        # Initialize crisis gate
        self._crisis_gate = CrisisGate(
            config_path=config_path,
            threshold=crisis_threshold,
            keyword_boost=crisis_keyword_boost,
            crisis_centroid=crisis_centroid,
            encode_fn=self._encode,
            history_window=history_window,
        )

        logger.info(
            "EmbeddingRouter initialized: %d centroids, model=%s",
            len(self._skill_names_order),
            model_name,
        )

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to a unit-norm embedding vector."""
        emb = self._model.encode(text, normalize_embeddings=True)
        return emb.astype(np.float32)

    def _blend_with_history(
        self, current_emb: np.ndarray, history: Optional[list]
    ) -> np.ndarray:
        """Blend current message embedding with history using exponential decay.

        Args:
            current_emb: Unit-norm embedding of the current message.
            history: List of (user_msg, assistant_msg) tuples.

        Returns:
            Unit-norm blended embedding.
        """
        if not history:
            return current_emb

        recent = history[-self._history_window:]
        if not recent:
            return current_emb

        # Encode history user messages with exponential decay weights
        # Most recent history turn gets highest weight
        history_embs = []
        decay_weights = []
        for i, (user_msg, _) in enumerate(recent):
            h_emb = self._encode(user_msg)
            history_embs.append(h_emb)
            # Exponential decay: more recent = higher weight
            decay_weights.append(0.5 ** (len(recent) - 1 - i))

        # Weighted average of history embeddings
        decay_weights = np.array(decay_weights, dtype=np.float32)
        decay_weights /= decay_weights.sum()
        history_avg = np.zeros_like(current_emb)
        for emb, w in zip(history_embs, decay_weights):
            history_avg += w * emb

        # Blend: current dominates
        current_weight = 1.0 - self._history_weight
        blended = current_weight * current_emb + self._history_weight * history_avg

        # Re-normalize to unit norm
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm

        return blended

    def route(self, message: str, history: Optional[list] = None) -> str:
        """Route a user message to the best-matching skill."""
        skill, _, _ = self.route_with_confidence(message, history)
        return skill

    def route_with_confidence(
        self, message: str, history: Optional[list] = None
    ) -> tuple:
        """
        Route a message with full confidence details.

        Returns:
            Tuple of (skill_name, confidence, details_dict).
        """
        details = {
            "scores": {},
            "router_type": "embedding",
            "crisis_gate": None,
            "history_used": history is not None and len(history or []) > 0,
        }

        # Step 1: Crisis gate — always runs first
        crisis_result = self._crisis_gate.check(message, history)
        details["crisis_gate"] = crisis_result

        if crisis_result["is_crisis"]:
            # Immediately route to crisis-intervention
            details["scores"]["crisis-intervention"] = {
                "score": max(crisis_result["embedding_score"], 0.95),
                "crisis_gate_triggered": True,
            }
            return "crisis-intervention", 0.95, details

        # Step 2: Encode and optionally blend with history
        msg_emb = self._encode(message)
        blended_emb = self._blend_with_history(msg_emb, history)

        # Step 3: Cosine similarity to all centroids
        similarities = np.dot(self._centroids, blended_emb)

        best_skill = self.default_skill
        best_score = 0.0

        for i, skill_name in enumerate(self._skill_names_order):
            score = float(similarities[i])
            details["scores"][skill_name] = {"score": score}

            if score > best_score:
                best_score = score
                best_skill = skill_name

        return best_skill, best_score, details

    def get_system_prompt(self, skill_name: str) -> str:
        """Get the system prompt for a given skill."""
        skill = self._by_name.get(skill_name)
        if skill:
            return skill["system_prompt"]
        default = self._by_name.get(self.default_skill)
        return default["system_prompt"] if default else ""

    def get_adapter_path(self, skill_name: str) -> str:
        """Get the adapter path for a given skill."""
        skill = self._by_name.get(skill_name)
        if skill:
            return skill["adapter_path"]
        return ""

    def list_skills(self) -> list:
        """Return list of all skill names in priority order."""
        return [s["name"] for s in self.skills]
