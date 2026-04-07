"""
State definitions for the counseling orchestration pipeline.

Each node in the pipeline produces a typed result that feeds into the next node.
The TurnState accumulates results as the turn progresses through the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class CrisisLevel(str, Enum):
    """Crisis severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardAction(str, Enum):
    """Response guard actions."""
    PASS = "pass"
    MODIFY = "modify"
    BLOCK = "block"


@dataclass
class NodeResult:
    """Base result from a pipeline node."""
    node_name: str
    success: bool
    duration_ms: float
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node_name,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


@dataclass
class TriageResult(NodeResult):
    """Result from triage node (crisis detection + skill routing)."""
    crisis_level: CrisisLevel = CrisisLevel.NONE
    crisis_keywords: list[str] = field(default_factory=list)
    detected_skill: str = "general-support"
    confidence: float = 0.0
    should_retrieve: bool = False
    
    def __post_init__(self):
        self.node_name = "triage"
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "crisis_level": self.crisis_level.value,
            "crisis_keywords": self.crisis_keywords,
            "detected_skill": self.detected_skill,
            "confidence": round(self.confidence, 3),
            "should_retrieve": self.should_retrieve,
        })
        return base


@dataclass
class RetrievalResult(NodeResult):
    """Result from retrieval node (KB lookup)."""
    retrieved_chunks: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    query_used: str = ""
    
    def __post_init__(self):
        self.node_name = "retrieve"
    
    @property
    def has_context(self) -> bool:
        return len(self.retrieved_chunks) > 0
    
    def format_context(self, max_chunks: int = 3) -> str:
        """Format retrieved chunks for injection into prompt."""
        if not self.retrieved_chunks:
            return ""
        
        parts = ["[Relevant Context]"]
        for chunk in self.retrieved_chunks[:max_chunks]:
            source = chunk.get("source", "unknown")
            content = chunk.get("content", "")
            parts.append(f"• {content[:200]}... [Source: {source}]")
        return "\n".join(parts)
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "num_chunks": len(self.retrieved_chunks),
            "citations": self.citations,
            "query_used": self.query_used,
        })
        return base


@dataclass
class GenerationResult(NodeResult):
    """Result from generation node (model inference)."""
    response: str = ""
    model_id: str = ""
    adapter_used: str | None = None
    tokens_generated: int = 0
    
    def __post_init__(self):
        self.node_name = "generate"
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "response_length": len(self.response),
            "model_id": self.model_id,
            "adapter_used": self.adapter_used,
            "tokens_generated": self.tokens_generated,
        })
        return base


@dataclass
class GuardResult(NodeResult):
    """Result from guard node (response safety check)."""
    action: GuardAction = GuardAction.PASS
    original_response: str = ""
    final_response: str = ""
    flags: list[str] = field(default_factory=list)
    modifications: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_name = "guard"
    
    @property
    def was_modified(self) -> bool:
        return self.action in (GuardAction.MODIFY, GuardAction.BLOCK)
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "action": self.action.value,
            "flags": self.flags,
            "was_modified": self.was_modified,
        })
        return base


@dataclass
class PersistResult(NodeResult):
    """Result from persist node (save to memory/session)."""
    session_saved: bool = False
    memory_saved: bool = False
    importance_score: float = 0.0
    
    def __post_init__(self):
        self.node_name = "persist"
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "session_saved": self.session_saved,
            "memory_saved": self.memory_saved,
            "importance_score": round(self.importance_score, 3),
        })
        return base


@dataclass
class TurnState:
    """
    Accumulated state for a single conversation turn.
    
    Flows through the pipeline, accumulating results from each node.
    """
    # Input
    user_id: int
    user_message: str
    conversation_history: list[tuple[str, str]] = field(default_factory=list)
    model_id: str = "qwen-ft"
    
    # Pipeline results (populated as turn progresses)
    triage: TriageResult | None = None
    retrieval: RetrievalResult | None = None
    generation: GenerationResult | None = None
    guard: GuardResult | None = None
    persist: PersistResult | None = None
    
    # Metadata
    turn_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    
    def __post_init__(self):
        if not self.turn_id:
            self.turn_id = f"{self.user_id}_{int(time.time() * 1000)}"
    
    @property
    def final_response(self) -> str:
        """Get the final response to send to user."""
        if self.guard and self.guard.final_response:
            return self.guard.final_response
        if self.generation and self.generation.response:
            return self.generation.response
        return ""
    
    @property
    def is_crisis(self) -> bool:
        """Check if this turn involves a crisis."""
        if not self.triage:
            return False
        return self.triage.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)
    
    @property
    def total_duration_ms(self) -> float:
        """Total pipeline duration in milliseconds."""
        if not self.completed_at:
            return 0.0
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000
    
    def complete(self) -> None:
        """Mark the turn as complete."""
        self.completed_at = datetime.now(timezone.utc)
    
    def to_trace(self) -> "PipelineTrace":
        """Convert to a trace for logging/debugging."""
        return PipelineTrace(
            turn_id=self.turn_id,
            user_id=self.user_id,
            started_at=self.started_at.isoformat(),
            completed_at=self.completed_at.isoformat() if self.completed_at else None,
            total_duration_ms=self.total_duration_ms,
            nodes=[
                r.to_dict() for r in [
                    self.triage, self.retrieval, self.generation, 
                    self.guard, self.persist
                ] if r is not None
            ],
            final_response_length=len(self.final_response),
            is_crisis=self.is_crisis,
        )


@dataclass
class PipelineTrace:
    """
    Trace of a pipeline execution for logging and debugging.
    
    Serializable for storage and analysis.
    """
    turn_id: str
    user_id: int
    started_at: str
    completed_at: str | None
    total_duration_ms: float
    nodes: list[dict]
    final_response_length: int
    is_crisis: bool
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_id": self.user_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "nodes": self.nodes,
            "final_response_length": self.final_response_length,
            "is_crisis": self.is_crisis,
        }
    
    def summary(self) -> str:
        """One-line summary for logging."""
        node_times = ", ".join(
            f"{n['node']}={n['duration_ms']:.0f}ms" 
            for n in self.nodes
        )
        crisis_flag = " [CRISIS]" if self.is_crisis else ""
        return f"Turn {self.turn_id}: {self.total_duration_ms:.0f}ms total ({node_times}){crisis_flag}"
