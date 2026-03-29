"""
Skill Router for Mental Health LLM.

Facade that delegates to the best available routing backend:
  - "embedding" → EmbeddingRouter (requires sentence-transformers + centroids)
  - "keyword"   → KeywordRouter (CPU-only, zero dependencies)
  - "auto"      → try embedding, fall back to keyword

Usage:
    from mental_health_llm.skill_router import SkillRouter

    router = SkillRouter()
    skill = router.route("I want to kill myself")
    # -> "crisis-intervention"

    skill, confidence, details = router.route_with_confidence("What is anxiety?")
    # -> ("psychoeducation", 0.8, {...})

    # Explicitly select backend:
    router = SkillRouter(backend="keyword")    # force keyword-only
    router = SkillRouter(backend="embedding")  # force embedding (raises if unavailable)
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SkillRouter:
    """Routes user messages to skill-specific LoRA adapters.

    Thin facade that auto-selects the best available routing backend.
    All existing callers work unchanged — the constructor signature is
    backward-compatible, and new parameters are keyword-only with defaults.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        backend: str = "auto",
    ):
        """
        Initialize the router with backend selection.

        Args:
            config_path: Path to skills_config.json. Defaults to the one
                         bundled with the mental_health_llm package.
            backend: Routing backend — "keyword", "embedding", or "auto".
                     "auto" tries embedding first, falls back to keyword.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / "skills_config.json")

        # Read optional router config from skills_config.json
        with open(config_path, "r") as f:
            config = json.load(f)

        if backend == "auto":
            backend = config.get("router_backend", "auto")

        self._backend_name = backend
        self._delegate = None

        if backend in ("auto", "embedding"):
            self._delegate = self._try_embedding(config_path, config)

        if self._delegate is None:
            if backend == "embedding":
                raise RuntimeError(
                    "Embedding router requested but could not be initialized. "
                    "Install sentence-transformers and build centroids first."
                )
            # Fall back to keyword
            from mental_health_llm.keyword_router import KeywordRouter

            self._delegate = KeywordRouter(config_path=config_path)
            if backend == "auto":
                logger.info("Using keyword router (embedding backend unavailable)")
            self._backend_name = "keyword"

    @staticmethod
    def _try_embedding(config_path: str, config: dict):
        """Attempt to create an EmbeddingRouter. Returns None on failure."""
        try:
            from mental_health_llm.embedding_router import EmbeddingRouter

            emb_config = config.get("embedding_router", {})
            crisis_config = config.get("crisis_gate", {})

            router = EmbeddingRouter(
                config_path=config_path,
                model_name=emb_config.get(
                    "model_name", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                history_window=emb_config.get("history_window", 3),
                history_weight=emb_config.get("history_weight", 0.2),
                crisis_threshold=crisis_config.get("threshold", 0.45),
                crisis_keyword_boost=crisis_config.get("keyword_boost", True),
            )
            logger.info("Using embedding router")
            return router

        except ImportError:
            warnings.warn(
                "sentence-transformers not installed. "
                "Install with: pip install mental-health-llm[router]",
                stacklevel=3,
            )
            return None
        except FileNotFoundError as e:
            warnings.warn(str(e), stacklevel=3)
            return None
        except Exception as e:
            warnings.warn(
                f"Failed to initialize embedding router: {e}",
                stacklevel=3,
            )
            return None

    @property
    def backend(self) -> str:
        """Return the name of the active routing backend."""
        return self._backend_name

    # --- Delegated interface (backward-compatible signatures) ---

    def route(self, message: str, history: Optional[list] = None) -> str:
        """
        Route a user message to the best-matching skill.

        Args:
            message: The user's input message.
            history: Optional list of (user_msg, assistant_msg) tuples.

        Returns:
            Skill name string (e.g. "crisis-intervention").
        """
        return self._delegate.route(message, history)

    def route_with_confidence(
        self, message: str, history: Optional[list] = None
    ) -> tuple:
        """
        Route a message and return confidence details for debugging.

        Args:
            message: The user's input message.
            history: Optional conversation history.

        Returns:
            Tuple of (skill_name, confidence, details_dict).
        """
        return self._delegate.route_with_confidence(message, history)

    def get_system_prompt(self, skill_name: str) -> str:
        """Get the system prompt for a given skill."""
        return self._delegate.get_system_prompt(skill_name)

    def get_adapter_path(self, skill_name: str) -> str:
        """Get the adapter path for a given skill."""
        return self._delegate.get_adapter_path(skill_name)

    def list_skills(self) -> list:
        """Return list of all skill names in priority order."""
        return self._delegate.list_skills()
