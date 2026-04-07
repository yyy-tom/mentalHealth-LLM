"""
Orchestration module for mental health counseling pipeline.

Provides a lightweight 5-node state machine:
  Triage → Retrieve → Generate → Guard → Persist
"""

from .state import (
    TurnState,
    NodeResult,
    PipelineTrace,
    TriageResult,
    RetrievalResult,
    GenerationResult,
    GuardResult,
    PersistResult,
)
from .pipeline import (
    CounselingPipeline,
    PipelineConfig,
    run_pipeline,
)

__all__ = [
    # State
    "TurnState",
    "NodeResult",
    "PipelineTrace",
    "TriageResult",
    "RetrievalResult",
    "GenerationResult",
    "GuardResult",
    "PersistResult",
    # Pipeline
    "CounselingPipeline",
    "PipelineConfig",
    "run_pipeline",
]
