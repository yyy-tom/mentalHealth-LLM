"""
Configuration management for evaluation harness.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FeatureFlags:
    """Feature toggles for ablation testing."""
    compaction: bool = True
    response_guard: bool = True
    dynamic_prompts: bool = True
    session_store: bool = True
    tiered_context: bool = True
    multi_layer_compaction: bool = True
    memory_persistence: bool = True
    
    def to_dict(self) -> dict[str, bool]:
        return {
            "compaction": self.compaction,
            "response_guard": self.response_guard,
            "dynamic_prompts": self.dynamic_prompts,
            "session_store": self.session_store,
            "tiered_context": self.tiered_context,
            "multi_layer_compaction": self.multi_layer_compaction,
            "memory_persistence": self.memory_persistence,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, bool]) -> "FeatureFlags":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
    
    @classmethod
    def all_off(cls) -> "FeatureFlags":
        return cls(
            compaction=False,
            response_guard=False,
            dynamic_prompts=False,
            session_store=False,
            tiered_context=False,
            multi_layer_compaction=False,
            memory_persistence=False,
        )
    
    @classmethod
    def all_on(cls) -> "FeatureFlags":
        return cls()  # All defaults are True
    
    def with_only(self, feature: str) -> "FeatureFlags":
        """Return new FeatureFlags with only one feature enabled."""
        flags = self.all_off()
        if hasattr(flags, feature):
            setattr(flags, feature, True)
        return flags


@dataclass
class JudgeConfig:
    """Configuration for LLM judge."""
    name: str = "deepseek"
    model: str = "deepseek-chat"
    api_base: str = "https://api.deepseek.com"
    temperature: float = 0.0
    max_retries: int = 3
    timeout: float = 60.0


@dataclass
class HarnessConfig:
    """Main configuration for evaluation harness."""
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    cases_dir: Path = field(default_factory=lambda: Path("evaluation/cases"))
    baselines_dir: Path = field(default_factory=lambda: Path("evaluation/baselines"))
    results_dir: Path = field(default_factory=lambda: Path("evaluation/results"))
    
    # Evaluation settings
    seed: int = 42
    max_new_tokens: int = 512
    temperature: float = 0.7
    num_workers: int = 1
    
    # Statistical settings
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    min_sample_size: int = 30  # For statistical significance
    multiple_comparison_correction: str = "none"  # "none" | "bonferroni" | "fdr_bh"
    
    # Runtime safety
    allow_placeholder_fallback: bool = True
    
    # Feature flags
    features: FeatureFlags = field(default_factory=FeatureFlags)
    
    # Judge configuration
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    
    # Model registry
    model_registry: dict[str, str] = field(default_factory=lambda: {
        "qwen-ft": "models/qwen2.5-7b-mental-health-fullft-a100",
        "qwen-base": "Qwen/Qwen2.5-7B-Instruct",
        "gemma-ft": "models/gemma2-9b-mental-health-fullft-a100",
        "gemma-base": "google/gemma-2-9b-it",
        "mistral-ft": "models/mistral-7b-mental-health-fullft-a100",
        "mistral-base": "mistralai/Mistral-7B-Instruct-v0.3",
    })
    
    def __post_init__(self):
        # Convert string paths to Path objects
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)
        if isinstance(self.cases_dir, str):
            self.cases_dir = Path(self.cases_dir)
        if isinstance(self.baselines_dir, str):
            self.baselines_dir = Path(self.baselines_dir)
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        
        # Make paths absolute
        if not self.cases_dir.is_absolute():
            self.cases_dir = self.project_root / self.cases_dir
        if not self.baselines_dir.is_absolute():
            self.baselines_dir = self.project_root / self.baselines_dir
        if not self.results_dir.is_absolute():
            self.results_dir = self.project_root / self.results_dir
        
        # Ensure directories exist
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate statistical config
        valid_corrections = {"none", "bonferroni", "fdr_bh"}
        if self.multiple_comparison_correction not in valid_corrections:
            raise ValueError(
                "multiple_comparison_correction must be one of: "
                f"{', '.join(sorted(valid_corrections))}"
            )
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "HarnessConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def from_json(cls, path: str | Path) -> "HarnessConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "HarnessConfig":
        """Create config from dictionary."""
        # Handle nested configs
        if "features" in data and isinstance(data["features"], dict):
            data["features"] = FeatureFlags.from_dict(data["features"])
        if "judge" in data and isinstance(data["judge"], dict):
            data["judge"] = JudgeConfig(**data["judge"])
        
        # Filter to known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        
        return cls(**filtered)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "project_root": str(self.project_root),
            "cases_dir": str(self.cases_dir),
            "baselines_dir": str(self.baselines_dir),
            "results_dir": str(self.results_dir),
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "num_workers": self.num_workers,
            "bootstrap_samples": self.bootstrap_samples,
            "confidence_level": self.confidence_level,
            "min_sample_size": self.min_sample_size,
            "multiple_comparison_correction": self.multiple_comparison_correction,
            "allow_placeholder_fallback": self.allow_placeholder_fallback,
            "features": self.features.to_dict(),
            "judge": {
                "name": self.judge.name,
                "model": self.judge.model,
                "api_base": self.judge.api_base,
                "temperature": self.judge.temperature,
            },
            "model_registry": self.model_registry,
        }
    
    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def save_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
