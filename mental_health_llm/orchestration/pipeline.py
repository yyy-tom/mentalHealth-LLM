"""
Counseling pipeline executor.

Implements a 5-node linear pipeline:
  Triage → Retrieve → Generate → Guard → Persist

Each node is a function that takes TurnState and returns a result.
The pipeline executes nodes sequentially, timing each step.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .state import (
    TurnState,
    TriageResult,
    RetrievalResult,
    GenerationResult,
    GuardResult,
    PersistResult,
    PipelineTrace,
    CrisisLevel,
    GuardAction,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the counseling pipeline."""
    # Feature toggles
    enable_retrieval: bool = False  # KB lookup (Phase B)
    enable_guard: bool = True
    enable_memory_persist: bool = True
    
    # Thresholds
    crisis_escalation_threshold: float = 0.8
    retrieval_confidence_threshold: float = 0.5
    
    # Limits
    max_retrieved_chunks: int = 3
    max_response_length: int = 2000
    
    # Logging
    trace_logging: bool = True
    log_responses: bool = False  # Privacy: don't log actual responses by default


class TriageNode(Protocol):
    """Protocol for triage node implementations."""
    def __call__(self, state: TurnState) -> TriageResult: ...


class RetrievalNode(Protocol):
    """Protocol for retrieval node implementations."""
    def __call__(self, state: TurnState, triage: TriageResult) -> RetrievalResult: ...


class GenerationNode(Protocol):
    """Protocol for generation node implementations."""
    def __call__(
        self, 
        state: TurnState, 
        triage: TriageResult,
        retrieval: RetrievalResult | None,
    ) -> GenerationResult: ...


class GuardNode(Protocol):
    """Protocol for guard node implementations."""
    def __call__(
        self, 
        state: TurnState,
        generation: GenerationResult,
        triage: TriageResult,
    ) -> GuardResult: ...


class PersistNode(Protocol):
    """Protocol for persist node implementations."""
    def __call__(
        self,
        state: TurnState,
        generation: GenerationResult,
        guard: GuardResult,
        triage: TriageResult,
    ) -> PersistResult: ...


def _timed_call(fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Call a function and return (result, duration_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000
    return result, duration_ms


class CounselingPipeline:
    """
    Main pipeline executor for counseling conversations.
    
    Executes 5 nodes in sequence:
    1. Triage - Crisis detection and skill routing
    2. Retrieve - KB lookup (optional)
    3. Generate - Model inference
    4. Guard - Response safety check
    5. Persist - Save to session/memory
    """
    
    def __init__(
        self,
        config: PipelineConfig | None = None,
        triage_fn: TriageNode | None = None,
        retrieval_fn: RetrievalNode | None = None,
        generation_fn: GenerationNode | None = None,
        guard_fn: GuardNode | None = None,
        persist_fn: PersistNode | None = None,
    ):
        self.config = config or PipelineConfig()
        
        # Node implementations (use defaults if not provided)
        self._triage = triage_fn or self._default_triage
        self._retrieval = retrieval_fn or self._default_retrieval
        self._generation = generation_fn or self._default_generation
        self._guard = guard_fn or self._default_guard
        self._persist = persist_fn or self._default_persist
    
    def run(self, state: TurnState) -> TurnState:
        """
        Run the full pipeline on a turn state.
        
        Args:
            state: Input turn state with user message and history
        
        Returns:
            Updated turn state with all node results
        """
        try:
            # Node 1: Triage
            state.triage, duration = _timed_call(self._triage, state)
            state.triage.duration_ms = duration
            logger.debug(f"Triage: {state.triage.detected_skill}, crisis={state.triage.crisis_level.value}")
            
            # Node 2: Retrieve (optional)
            if self.config.enable_retrieval and state.triage.should_retrieve:
                state.retrieval, duration = _timed_call(
                    self._retrieval, state, state.triage
                )
                state.retrieval.duration_ms = duration
                logger.debug(f"Retrieval: {len(state.retrieval.retrieved_chunks)} chunks")
            else:
                state.retrieval = RetrievalResult(
                    node_name="retrieve",
                    success=True,
                    duration_ms=0.0,
                )
            
            # Node 3: Generate
            state.generation, duration = _timed_call(
                self._generation, state, state.triage, state.retrieval
            )
            state.generation.duration_ms = duration
            logger.debug(f"Generation: {len(state.generation.response)} chars")
            
            # Node 4: Guard
            if self.config.enable_guard:
                state.guard, duration = _timed_call(
                    self._guard, state, state.generation, state.triage
                )
                state.guard.duration_ms = duration
                if state.guard.was_modified:
                    logger.info(f"Guard modified response: {state.guard.flags}")
            else:
                state.guard = GuardResult(
                    node_name="guard",
                    success=True,
                    duration_ms=0.0,
                    action=GuardAction.PASS,
                    original_response=state.generation.response,
                    final_response=state.generation.response,
                )
            
            # Node 5: Persist
            if self.config.enable_memory_persist:
                state.persist, duration = _timed_call(
                    self._persist, state, state.generation, state.guard, state.triage
                )
                state.persist.duration_ms = duration
            else:
                state.persist = PersistResult(
                    node_name="persist",
                    success=True,
                    duration_ms=0.0,
                )
            
            state.complete()
            
            # Log trace if enabled
            if self.config.trace_logging:
                trace = state.to_trace()
                logger.info(trace.summary())
            
            return state
            
        except Exception as e:
            logger.exception(f"Pipeline error for user {state.user_id}: {e}")
            state.complete()
            raise
    
    # ─────────────────────────────────────────────────────────────────
    # Default node implementations (can be overridden)
    # ─────────────────────────────────────────────────────────────────
    
    def _default_triage(self, state: TurnState) -> TriageResult:
        """Default triage using existing crisis gate and skill router."""
        try:
            from mental_health_llm import CrisisGate, SkillRouter
            
            # Crisis detection
            gate = CrisisGate()
            crisis_result = gate.check(state.user_message)
            
            # Map crisis level
            level_map = {
                "none": CrisisLevel.NONE,
                "low": CrisisLevel.LOW,
                "medium": CrisisLevel.MEDIUM,
                "high": CrisisLevel.HIGH,
                "critical": CrisisLevel.CRITICAL,
            }
            crisis_level = level_map.get(
                crisis_result.level.lower(), CrisisLevel.NONE
            )
            
            # Skill routing
            router = SkillRouter()
            skill_match = router.route(state.user_message)
            detected_skill = skill_match.skill if skill_match else "general-support"
            confidence = skill_match.confidence if skill_match else 0.0
            
            # Decide if retrieval is needed
            # (enable for psychoeducation queries when retrieval is implemented)
            should_retrieve = detected_skill in (
                "psychoeducation", "coping-strategies", "grounding"
            )
            
            return TriageResult(
                node_name="triage",
                success=True,
                duration_ms=0.0,
                crisis_level=crisis_level,
                crisis_keywords=crisis_result.keywords if hasattr(crisis_result, 'keywords') else [],
                detected_skill=detected_skill,
                confidence=confidence,
                should_retrieve=should_retrieve,
            )
        except ImportError:
            # Fallback if modules not available
            return TriageResult(
                node_name="triage",
                success=True,
                duration_ms=0.0,
                crisis_level=CrisisLevel.NONE,
                detected_skill="general-support",
                confidence=0.5,
                should_retrieve=False,
            )
    
    def _default_retrieval(
        self, state: TurnState, triage: TriageResult
    ) -> RetrievalResult:
        """Default retrieval using psychoeducation knowledge base."""
        if not self.config.enable_retrieval:
            return RetrievalResult(
                node_name="retrieve",
                success=True,
                duration_ms=0.0,
                retrieved_chunks=[],
                citations=[],
                query_used=state.user_message,
                skipped=True,
            )
        
        try:
            from mental_health_llm.retrieval import KnowledgeIndex, RetrievalEngine
            from pathlib import Path
            
            # Use project-level KB path
            kb_db = Path("data/kb/index.db")
            if not kb_db.exists():
                # Index not built yet
                return RetrievalResult(
                    node_name="retrieve",
                    success=True,
                    duration_ms=0.0,
                    retrieved_chunks=[],
                    citations=[],
                    query_used=state.user_message,
                    skipped=True,
                    error="Knowledge base not indexed",
                )
            
            index = KnowledgeIndex(db_path=kb_db)
            engine = RetrievalEngine(
                index=index,
                default_top_k=self.config.max_retrieved_chunks,
                relevance_threshold=0.3,
            )
            
            # Skill-aware search
            skill = triage.detected_skill if triage else "general-support"
            result = engine.search_for_skill(
                query=state.user_message,
                skill=skill,
                top_k=self.config.max_retrieved_chunks,
            )
            
            # Convert to serializable format
            chunks = []
            citations = []
            for r in result.results:
                chunks.append({
                    "source": r.document.source,
                    "title": r.document.title,
                    "content": r.document.content[:500],
                    "score": r.score,
                })
                citations.append(r.document.source)
            
            return RetrievalResult(
                node_name="retrieve",
                success=True,
                duration_ms=result.search_time_ms,
                retrieved_chunks=chunks,
                citations=citations,
                query_used=state.user_message,
            )
            
        except ImportError:
            # sentence-transformers not installed
            return RetrievalResult(
                node_name="retrieve",
                success=True,
                duration_ms=0.0,
                retrieved_chunks=[],
                citations=[],
                query_used=state.user_message,
                skipped=True,
                error="Retrieval module not available",
            )
    
    def _default_generation(
        self,
        state: TurnState,
        triage: TriageResult,
        retrieval: RetrievalResult | None,
    ) -> GenerationResult:
        """
        Default generation - delegates to caller.
        
        This is a placeholder. The actual telegram bot will inject its own
        generation function that calls the model manager.
        """
        # This should be overridden by the caller
        return GenerationResult(
            node_name="generate",
            success=False,
            duration_ms=0.0,
            error="Generation function not configured",
            response="I apologize, but I'm unable to respond right now. Please try again.",
            model_id=state.model_id,
        )
    
    def _default_guard(
        self,
        state: TurnState,
        generation: GenerationResult,
        triage: TriageResult,
    ) -> GuardResult:
        """Default guard using ResponseGuard."""
        try:
            from mental_health_llm import ResponseGuard
            
            guard = ResponseGuard()
            result = guard.validate(
                response=generation.response,
                skill=triage.detected_skill,
                crisis_level=triage.crisis_level.value,
            )
            
            action_map = {
                "pass": GuardAction.PASS,
                "modified": GuardAction.MODIFY,
                "blocked": GuardAction.BLOCK,
            }
            action = action_map.get(result.action, GuardAction.PASS)
            
            return GuardResult(
                node_name="guard",
                success=True,
                duration_ms=0.0,
                action=action,
                original_response=generation.response,
                final_response=result.response,
                flags=result.flags if hasattr(result, 'flags') else [],
            )
        except ImportError:
            # Fallback: pass through
            return GuardResult(
                node_name="guard",
                success=True,
                duration_ms=0.0,
                action=GuardAction.PASS,
                original_response=generation.response,
                final_response=generation.response,
            )
    
    def _default_persist(
        self,
        state: TurnState,
        generation: GenerationResult,
        guard: GuardResult,
        triage: TriageResult,
    ) -> PersistResult:
        """
        Default persist - delegates to EnhancedContextManager.
        
        This is a placeholder. The actual telegram bot will inject its own
        persist function that uses the context manager.
        """
        # Calculate importance based on crisis level
        importance_map = {
            CrisisLevel.NONE: 0.3,
            CrisisLevel.LOW: 0.5,
            CrisisLevel.MEDIUM: 0.7,
            CrisisLevel.HIGH: 0.9,
            CrisisLevel.CRITICAL: 1.0,
        }
        importance = importance_map.get(triage.crisis_level, 0.3)
        
        # This should be overridden by the caller
        return PersistResult(
            node_name="persist",
            success=True,
            duration_ms=0.0,
            session_saved=False,
            memory_saved=False,
            importance_score=importance,
        )


def run_pipeline(
    user_id: int,
    user_message: str,
    history: list[tuple[str, str]],
    model_id: str = "qwen-ft",
    config: PipelineConfig | None = None,
    generation_fn: GenerationNode | None = None,
    persist_fn: PersistNode | None = None,
) -> TurnState:
    """
    Convenience function to run the pipeline.
    
    Args:
        user_id: User identifier
        user_message: Current user message
        history: Conversation history as (user, assistant) tuples
        model_id: Model to use for generation
        config: Pipeline configuration
        generation_fn: Custom generation function
        persist_fn: Custom persist function
    
    Returns:
        Completed turn state
    """
    state = TurnState(
        user_id=user_id,
        user_message=user_message,
        conversation_history=history,
        model_id=model_id,
    )
    
    pipeline = CounselingPipeline(
        config=config,
        generation_fn=generation_fn,
        persist_fn=persist_fn,
    )
    
    return pipeline.run(state)
