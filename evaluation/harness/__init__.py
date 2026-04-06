# Evaluation Harness for Mental Health LLM
"""
Harness Engineering module for systematic evaluation with:
- Baseline capture and comparison
- Feature ablation testing
- Statistical analysis (Wilcoxon, Bootstrap CI)
- Multi-judge validation
"""

from evaluation.harness.config import HarnessConfig
from evaluation.harness.runner import EvaluationHarness
from evaluation.harness.metrics import MetricsAggregator, StatisticalAnalyzer
from evaluation.harness.baseline import BaselineManager
from evaluation.harness.ablation import AblationRunner

__all__ = [
    "HarnessConfig",
    "EvaluationHarness",
    "MetricsAggregator",
    "StatisticalAnalyzer",
    "BaselineManager",
    "AblationRunner",
]
