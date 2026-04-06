"""
Baseline management for evaluation harness.

Captures, stores, and loads evaluation baselines for comparison.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import HarnessConfig, FeatureFlags
from .metrics import EvalMetrics


@dataclass
class Baseline:
    """Captured evaluation baseline."""
    id: str
    commit: str
    timestamp: str
    model: str
    test_suite: str
    features: dict[str, bool]
    metrics: dict[str, Any]
    raw_results: list[dict] | None = None
    description: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "commit": self.commit,
            "timestamp": self.timestamp,
            "model": self.model,
            "test_suite": self.test_suite,
            "features": self.features,
            "metrics": self.metrics,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Baseline":
        return cls(
            id=data["id"],
            commit=data["commit"],
            timestamp=data["timestamp"],
            model=data["model"],
            test_suite=data["test_suite"],
            features=data.get("features", {}),
            metrics=data["metrics"],
            description=data.get("description", ""),
        )
    
    def save(self, path: Path) -> None:
        """Save baseline to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Baseline":
        """Load baseline from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class BaselineManager:
    """Manage evaluation baselines."""
    
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.baselines_dir = config.baselines_dir
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
    
    def capture(
        self,
        baseline_id: str,
        model: str,
        test_suite: str,
        metrics: EvalMetrics,
        features: FeatureFlags | None = None,
        raw_results: list[dict] | None = None,
        description: str = "",
    ) -> Baseline:
        """
        Capture a new baseline from evaluation results.
        
        Args:
            baseline_id: Unique identifier for this baseline
            model: Model identifier used for evaluation
            test_suite: Test suite identifier used
            metrics: Aggregated evaluation metrics
            features: Feature flags active during evaluation
            raw_results: Optional raw evaluation results
            description: Human-readable description
        
        Returns:
            Captured baseline
        """
        commit = self._get_current_commit()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        baseline = Baseline(
            id=baseline_id,
            commit=commit,
            timestamp=timestamp,
            model=model,
            test_suite=test_suite,
            features=features.to_dict() if features else {},
            metrics=metrics.to_dict(),
            raw_results=raw_results,
            description=description,
        )
        
        # Save to file
        path = self.baselines_dir / f"{baseline_id}.json"
        baseline.save(path)
        
        # Save raw results separately if provided
        if raw_results:
            raw_path = self.baselines_dir / f"{baseline_id}_raw.json"
            with open(raw_path, "w") as f:
                json.dump(raw_results, f, indent=2)
        
        return baseline
    
    def load(self, baseline_id: str) -> Baseline:
        """Load a baseline by ID."""
        path = self.baselines_dir / f"{baseline_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Baseline not found: {baseline_id}")
        return Baseline.load(path)
    
    def load_raw_results(self, baseline_id: str) -> list[dict] | None:
        """Load raw results for a baseline."""
        raw_path = self.baselines_dir / f"{baseline_id}_raw.json"
        if not raw_path.exists():
            return None
        with open(raw_path) as f:
            return json.load(f)
    
    def list_baselines(self) -> list[str]:
        """List all available baseline IDs."""
        baselines = []
        for path in self.baselines_dir.glob("*.json"):
            if not path.name.endswith("_raw.json"):
                baselines.append(path.stem)
        return sorted(baselines)
    
    def get_latest(self) -> Baseline | None:
        """Get the most recent baseline."""
        baselines = self.list_baselines()
        if not baselines:
            return None
        
        # Load all and sort by timestamp
        loaded = []
        for bid in baselines:
            try:
                b = self.load(bid)
                loaded.append(b)
            except Exception:
                continue
        
        if not loaded:
            return None
        
        loaded.sort(key=lambda b: b.timestamp, reverse=True)
        return loaded[0]
    
    def delete(self, baseline_id: str) -> bool:
        """Delete a baseline and its raw results."""
        path = self.baselines_dir / f"{baseline_id}.json"
        raw_path = self.baselines_dir / f"{baseline_id}_raw.json"
        
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        if raw_path.exists():
            raw_path.unlink()
        
        return deleted
    
    def _get_current_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def compare_baselines(
        self,
        baseline_id_1: str,
        baseline_id_2: str,
    ) -> dict[str, Any]:
        """
        Compare two baselines.
        
        Returns a comparison report with differences.
        """
        b1 = self.load(baseline_id_1)
        b2 = self.load(baseline_id_2)
        
        comparison = {
            "baseline_1": {"id": b1.id, "commit": b1.commit, "timestamp": b1.timestamp},
            "baseline_2": {"id": b2.id, "commit": b2.commit, "timestamp": b2.timestamp},
            "dimension_changes": {},
            "feature_changes": {},
        }
        
        # Compare dimensions
        dims1 = b1.metrics.get("dimensions", {})
        dims2 = b2.metrics.get("dimensions", {})
        
        for dim in set(dims1.keys()) | set(dims2.keys()):
            d1 = dims1.get(dim, {})
            d2 = dims2.get(dim, {})
            
            mean1 = d1.get("mean", 0) if isinstance(d1, dict) else 0
            mean2 = d2.get("mean", 0) if isinstance(d2, dict) else 0
            
            diff = mean2 - mean1
            pct = (diff / mean1 * 100) if mean1 != 0 else 0
            
            comparison["dimension_changes"][dim] = {
                "baseline_1": mean1,
                "baseline_2": mean2,
                "difference": round(diff, 3),
                "percent_change": round(pct, 2),
            }
        
        # Compare features
        for feat in set(b1.features.keys()) | set(b2.features.keys()):
            f1 = b1.features.get(feat)
            f2 = b2.features.get(feat)
            if f1 != f2:
                comparison["feature_changes"][feat] = {"baseline_1": f1, "baseline_2": f2}
        
        return comparison


def capture_baseline_at_commit(
    commit: str,
    baseline_id: str,
    model: str = "qwen-ft",
    test_suite: str = "all",
    config: HarnessConfig | None = None,
) -> str:
    """
    Capture a baseline at a specific git commit.
    
    This checks out the commit, runs evaluation, and captures the baseline,
    then returns to the original branch.
    
    Returns the path to the captured baseline.
    """
    import os
    
    config = config or HarnessConfig()
    project_root = config.project_root
    
    # Get current branch
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    original_branch = result.stdout.strip()
    
    try:
        # Stash any changes
        subprocess.run(["git", "stash"], cwd=project_root, check=True)
        
        # Checkout target commit
        subprocess.run(["git", "checkout", commit], cwd=project_root, check=True)
        
        # Run evaluation (this would need the evaluation runner)
        # For now, just create a placeholder
        print(f"Would run evaluation at commit {commit}")
        print(f"Model: {model}, Test suite: {test_suite}")
        
        # Return to original branch
        subprocess.run(["git", "checkout", original_branch], cwd=project_root, check=True)
        
        # Pop stash
        subprocess.run(["git", "stash", "pop"], cwd=project_root, check=False)
        
        return str(config.baselines_dir / f"{baseline_id}.json")
    
    except Exception as e:
        # Ensure we return to original branch
        subprocess.run(["git", "checkout", original_branch], cwd=project_root, check=False)
        subprocess.run(["git", "stash", "pop"], cwd=project_root, check=False)
        raise e
