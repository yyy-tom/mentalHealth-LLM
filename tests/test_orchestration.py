"""
Unit tests for the orchestration pipeline.
"""

import pytest
from datetime import datetime, timezone

from mental_health_llm.orchestration import (
    TurnState,
    PipelineConfig,
    CounselingPipeline,
    PipelineTrace,
    run_pipeline,
)
from mental_health_llm.orchestration.state import (
    CrisisLevel,
    GuardAction,
    TriageResult,
    RetrievalResult,
    GenerationResult,
    GuardResult,
    PersistResult,
)


class TestTurnState:
    """Tests for TurnState dataclass."""
    
    def test_create_turn_state(self):
        state = TurnState(
            user_id=123,
            user_message="I'm feeling anxious",
            model_id="qwen-ft",
        )
        assert state.user_id == 123
        assert state.user_message == "I'm feeling anxious"
        assert state.turn_id.startswith("123_")
        assert state.triage is None
        assert state.final_response == ""
    
    def test_turn_state_with_history(self):
        history = [("Hi", "Hello!"), ("How are you?", "I'm here to help.")]
        state = TurnState(
            user_id=456,
            user_message="I need help",
            conversation_history=history,
        )
        assert len(state.conversation_history) == 2
    
    def test_is_crisis_false_by_default(self):
        state = TurnState(user_id=1, user_message="test")
        assert state.is_crisis is False
    
    def test_is_crisis_true_when_high(self):
        state = TurnState(user_id=1, user_message="test")
        state.triage = TriageResult(
            node_name="triage",
            success=True,
            duration_ms=10.0,
            crisis_level=CrisisLevel.HIGH,
        )
        assert state.is_crisis is True
    
    def test_final_response_from_guard(self):
        state = TurnState(user_id=1, user_message="test")
        state.generation = GenerationResult(
            node_name="generate",
            success=True,
            duration_ms=100.0,
            response="Original response",
        )
        state.guard = GuardResult(
            node_name="guard",
            success=True,
            duration_ms=5.0,
            action=GuardAction.MODIFY,
            original_response="Original response",
            final_response="Modified response",
        )
        assert state.final_response == "Modified response"
    
    def test_complete_sets_timestamp(self):
        state = TurnState(user_id=1, user_message="test")
        assert state.completed_at is None
        state.complete()
        assert state.completed_at is not None
        assert state.total_duration_ms > 0
    
    def test_to_trace(self):
        state = TurnState(user_id=1, user_message="test")
        state.triage = TriageResult(
            node_name="triage",
            success=True,
            duration_ms=10.0,
        )
        state.complete()
        
        trace = state.to_trace()
        assert trace.user_id == 1
        assert trace.turn_id == state.turn_id
        assert len(trace.nodes) == 1
        assert trace.nodes[0]["node"] == "triage"


class TestNodeResults:
    """Tests for individual node result types."""
    
    def test_triage_result(self):
        result = TriageResult(
            node_name="triage",
            success=True,
            duration_ms=15.5,
            crisis_level=CrisisLevel.MEDIUM,
            detected_skill="anxiety-support",
            confidence=0.85,
        )
        
        d = result.to_dict()
        assert d["crisis_level"] == "medium"
        assert d["detected_skill"] == "anxiety-support"
        assert d["confidence"] == 0.85
    
    def test_retrieval_result_format_context(self):
        result = RetrievalResult(
            node_name="retrieve",
            success=True,
            duration_ms=50.0,
            retrieved_chunks=[
                {"source": "anxiety.md", "content": "Deep breathing helps..."},
                {"source": "coping.md", "content": "Grounding techniques..."},
            ],
        )
        
        assert result.has_context is True
        context = result.format_context()
        assert "[Relevant Context]" in context
        assert "anxiety.md" in context
    
    def test_retrieval_result_empty(self):
        result = RetrievalResult(
            node_name="retrieve",
            success=True,
            duration_ms=5.0,
        )
        assert result.has_context is False
        assert result.format_context() == ""
    
    def test_guard_result_was_modified(self):
        blocked = GuardResult(
            node_name="guard",
            success=True,
            duration_ms=3.0,
            action=GuardAction.BLOCK,
            original_response="bad",
            final_response="safe",
        )
        assert blocked.was_modified is True
        
        passed = GuardResult(
            node_name="guard",
            success=True,
            duration_ms=3.0,
            action=GuardAction.PASS,
            original_response="good",
            final_response="good",
        )
        assert passed.was_modified is False


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        config = PipelineConfig()
        assert config.enable_retrieval is False
        assert config.enable_guard is True
        assert config.enable_memory_persist is True
        assert config.max_retrieved_chunks == 3
    
    def test_custom_config(self):
        config = PipelineConfig(
            enable_retrieval=True,
            enable_guard=False,
            crisis_escalation_threshold=0.9,
        )
        assert config.enable_retrieval is True
        assert config.enable_guard is False
        assert config.crisis_escalation_threshold == 0.9


class TestCounselingPipeline:
    """Tests for the main pipeline executor."""
    
    def test_pipeline_creation(self):
        pipeline = CounselingPipeline()
        assert pipeline.config is not None
    
    def test_pipeline_with_custom_config(self):
        config = PipelineConfig(enable_guard=False)
        pipeline = CounselingPipeline(config=config)
        assert pipeline.config.enable_guard is False
    
    def test_pipeline_run_with_mock_generation(self):
        """Test pipeline with injected generation function."""
        
        def mock_generate(state, triage, retrieval):
            return GenerationResult(
                node_name="generate",
                success=True,
                duration_ms=50.0,
                response="I understand you're feeling anxious. Let's work through this together.",
                model_id="mock",
            )
        
        def mock_persist(state, generation, guard, triage):
            return PersistResult(
                node_name="persist",
                success=True,
                duration_ms=5.0,
                session_saved=True,
                memory_saved=True,
            )
        
        config = PipelineConfig(enable_retrieval=False, enable_guard=False)
        pipeline = CounselingPipeline(
            config=config,
            generation_fn=mock_generate,
            persist_fn=mock_persist,
        )
        
        state = TurnState(
            user_id=123,
            user_message="I'm feeling anxious today",
        )
        
        result = pipeline.run(state)
        
        assert result.triage is not None
        assert result.generation is not None
        assert result.generation.success is True
        assert "anxious" in result.final_response.lower()
        assert result.completed_at is not None
    
    def test_pipeline_run_populates_all_nodes(self):
        """Verify all nodes get populated even with defaults."""
        
        def mock_generate(state, triage, retrieval):
            return GenerationResult(
                node_name="generate",
                success=True,
                duration_ms=50.0,
                response="Test response",
                model_id="mock",
            )
        
        config = PipelineConfig(
            enable_retrieval=False,
            enable_guard=False,
            enable_memory_persist=False,
        )
        pipeline = CounselingPipeline(
            config=config,
            generation_fn=mock_generate,
        )
        
        state = TurnState(user_id=1, user_message="test")
        result = pipeline.run(state)
        
        assert result.triage is not None
        assert result.retrieval is not None  # Skipped but still populated
        assert result.generation is not None
        assert result.guard is not None  # Disabled but still populated
        assert result.persist is not None


class TestPipelineTrace:
    """Tests for PipelineTrace."""
    
    def test_trace_to_dict(self):
        trace = PipelineTrace(
            turn_id="123_456",
            user_id=123,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:01Z",
            total_duration_ms=1000.0,
            nodes=[
                {"node": "triage", "duration_ms": 10.0, "success": True},
                {"node": "generate", "duration_ms": 900.0, "success": True},
            ],
            final_response_length=150,
            is_crisis=False,
        )
        
        d = trace.to_dict()
        assert d["turn_id"] == "123_456"
        assert d["total_duration_ms"] == 1000.0
        assert len(d["nodes"]) == 2
    
    def test_trace_summary(self):
        trace = PipelineTrace(
            turn_id="123_456",
            user_id=123,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:01Z",
            total_duration_ms=500.0,
            nodes=[
                {"node": "triage", "duration_ms": 10.0, "success": True},
                {"node": "generate", "duration_ms": 480.0, "success": True},
            ],
            final_response_length=100,
            is_crisis=False,
        )
        
        summary = trace.summary()
        assert "123_456" in summary
        assert "500ms" in summary
        assert "triage=10ms" in summary
    
    def test_trace_summary_crisis_flag(self):
        trace = PipelineTrace(
            turn_id="999_111",
            user_id=999,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:00:01Z",
            total_duration_ms=200.0,
            nodes=[],
            final_response_length=50,
            is_crisis=True,
        )
        
        summary = trace.summary()
        assert "[CRISIS]" in summary


class TestRunPipelineHelper:
    """Tests for the run_pipeline convenience function."""
    
    def test_run_pipeline_creates_state(self):
        def mock_gen(state, triage, retrieval):
            return GenerationResult(
                node_name="generate",
                success=True,
                duration_ms=10.0,
                response="Hello",
                model_id="mock",
            )
        
        config = PipelineConfig(enable_guard=False, enable_memory_persist=False)
        result = run_pipeline(
            user_id=42,
            user_message="Hi there",
            history=[],
            config=config,
            generation_fn=mock_gen,
        )
        
        assert result.user_id == 42
        assert result.user_message == "Hi there"
        assert result.final_response == "Hello"
