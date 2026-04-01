"""
LRU adapter cache for lazy loading of LoRA skill adapters.

Instead of loading all 6 LoRA adapters at startup (consuming VRAM for
rarely-used skills), this loads adapters on first route and evicts the
least-recently-used adapter when the cache exceeds ``max_size``.

The ``crisis-intervention`` adapter is always pinned and never evicted.

Source pattern: claw-code deferred_init.py (trust-gated deferred
initialization).

Usage:

    from mental_health_llm.adapter_cache import AdapterCache

    cache = AdapterCache(max_size=3, adapters_dir="adapters")

    # Attach to a base model (before any adapters are loaded)
    cache.attach(base_model)

    # On each request — loads lazily, evicts LRU if needed
    if cache.ensure_loaded("cbt-therapy"):
        cache.model.set_adapter("cbt-therapy")
        # ... generate ...
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

PINNED_ADAPTER = "crisis-intervention"


class AdapterCache:
    """LRU cache for PEFT LoRA adapters with a pinned crisis adapter.

    Attributes:
        model: The PEFT-wrapped model (set via :meth:`attach`).
        loaded_skills: List of currently loaded adapter names.
    """

    def __init__(
        self,
        max_size: int = 3,
        adapters_dir: str = "adapters",
        pinned: str = PINNED_ADAPTER,
    ) -> None:
        """
        Args:
            max_size: Maximum number of adapters to keep in memory.
                      Must be >= 2 (pinned + at least one other).
            adapters_dir: Directory containing skill adapter subdirectories.
            pinned: Adapter name that is never evicted.
        """
        if max_size < 2:
            raise ValueError("max_size must be >= 2 (pinned + at least one other)")
        self._max_size = max_size
        self._adapters_dir = adapters_dir
        self._pinned = pinned
        # OrderedDict tracks LRU order: last item = most recently used
        self._cache: OrderedDict[str, bool] = OrderedDict()
        self._model = None
        self._is_peft = False  # True once first adapter is loaded

    def attach(self, model, *, project_root: str | Path | None = None) -> None:
        """Attach the base model that adapters will be loaded onto.

        Args:
            model: A HuggingFace model (pre-PEFT wrapping).
            project_root: If set, resolve relative adapters_dir against this.
        """
        self._model = model
        if project_root and not os.path.isabs(self._adapters_dir):
            self._adapters_dir = str(Path(project_root) / self._adapters_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_loaded(self, skill_name: str) -> bool:
        """Ensure the adapter for ``skill_name`` is loaded.

        Loads lazily on first access.  Evicts the least-recently-used
        non-pinned adapter if the cache is full.

        Returns:
            True if the adapter is available, False if loading failed
            (e.g. adapter files missing).
        """
        if self._model is None:
            raise RuntimeError("Call attach(model) before ensure_loaded()")

        if skill_name in self._cache:
            # Move to end = most recently used
            self._cache.move_to_end(skill_name)
            return True

        # Attempt to load
        if not self._load_adapter(skill_name):
            return False

        # Evict if over capacity
        while len(self._cache) > self._max_size:
            self._evict_lru()

        return True

    def preload_pinned(self) -> bool:
        """Pre-load the pinned adapter (e.g. at startup).

        Returns True if loaded successfully.
        """
        return self.ensure_loaded(self._pinned)

    @property
    def model(self):
        """The underlying model (PEFT-wrapped once an adapter is loaded)."""
        return self._model

    @property
    def loaded_skills(self) -> list[str]:
        """Currently loaded adapter names."""
        return list(self._cache.keys())

    @property
    def is_peft(self) -> bool:
        """Whether the model has been wrapped with PeftModel."""
        return self._is_peft

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_adapter(self, skill_name: str) -> bool:
        """Load a single adapter from disk."""
        adapter_path = os.path.join(self._adapters_dir, skill_name)
        if not self._has_adapter_files(adapter_path):
            logger.warning(
                "Adapter not found: %s — skipping %s", adapter_path, skill_name
            )
            return False

        try:
            if not self._is_peft:
                # First adapter: wrap the model with PeftModel
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(
                    self._model, adapter_path, adapter_name=skill_name
                )
                self._is_peft = True
            else:
                self._model.load_adapter(adapter_path, adapter_name=skill_name)

            self._cache[skill_name] = True
            logger.info("Loaded adapter: %s (%d/%d in cache)",
                        skill_name, len(self._cache), self._max_size)
            return True

        except Exception:
            logger.exception("Failed to load adapter: %s", skill_name)
            return False

    def _evict_lru(self) -> None:
        """Evict the least-recently-used non-pinned adapter."""
        # Iterate from oldest (front) to newest
        for name in list(self._cache.keys()):
            if name == self._pinned:
                continue
            self._cache.pop(name)
            try:
                self._model.delete_adapter(name)
                logger.info("Evicted adapter: %s (%d/%d in cache)",
                            name, len(self._cache), self._max_size)
            except Exception:
                logger.exception("Failed to evict adapter: %s", name)
            return

        # Should not reach here if max_size >= 2
        logger.warning("Cannot evict — only pinned adapter in cache")

    @staticmethod
    def _has_adapter_files(adapter_path: str) -> bool:
        """Check whether an adapter directory has the expected files."""
        return (
            os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
            or os.path.exists(
                os.path.join(adapter_path, "adapter_model.safetensors")
            )
            or os.path.exists(
                os.path.join(adapter_path, "adapter_model.bin")
            )
        )
