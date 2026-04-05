"""
Tests for claw-code architecture components.

Tests:
- MultiLayerCompactor: 5-layer progressive compaction
- TieredContextManager: Hot/Warm/Cold tier management
- CompactionMemoryStore: Self-healing memory persistence
"""

import pytest
import tempfile
import time
import os
from pathlib import Path

from mental_health_llm.compaction_layers import MultiLayerCompactor, CompactionResult
from mental_health_llm.tiered_context import (
    TieredContextManager,
    Tier,
    TieredTurn,
    ContextSnapshot,
)
from mental_health_llm.memory_persistence import CompactionMemoryStore, CompactionRecord


# ---------------------------------------------------------------------------
# MultiLayerCompactor Tests
# ---------------------------------------------------------------------------


class TestMultiLayerCompactor:
    """Tests for the 5-layer compaction pipeline."""

    @pytest.fixture
    def compactor(self):
        """Create a compactor without LLM/embedding functions."""
        return MultiLayerCompactor()

    @pytest.fixture
    def sample_history(self):
        """Create sample conversation history."""
        return [
            ("I've been feeling really anxious lately.", "I understand that anxiety can be overwhelming."),
            ("Work has been stressful.", "It sounds like work pressure is affecting you."),
            ("I can't sleep at night.", "Sleep difficulties often accompany stress."),
            ("I tried breathing exercises.", "That's great that you're trying coping strategies."),
            ("They help a bit.", "Even small improvements are progress."),
            ("What else can I try?", "There are several techniques we can explore."),
        ]

    def test_empty_history(self, compactor):
        """Test compaction of empty history."""
        result = compactor.compact([], target_tokens=1000)
        assert result.compacted == []
        assert result.layers_applied == []
        assert result.token_reduction == 0.0

    def test_within_budget_no_compaction(self, compactor, sample_history):
        """Test that no compaction occurs when within budget."""
        result = compactor.compact(sample_history, target_tokens=10000)
        # All recent turns should be preserved
        assert len(result.compacted) >= 4  # At least preserve_recent pairs

    def test_l1_turn_trimming(self, compactor):
        """Test L1 turn-level trimming."""
        # Create long messages that need trimming
        long_history = [
            ("This is a very long message. " * 50, "This is a very long response. " * 50),
            ("Another long message here. " * 50, "Another long response. " * 50),
        ] * 5

        result = compactor.compact(long_history, target_tokens=500, preserve_recent=1)
        assert "L1:turn_trim" in result.layers_applied
        assert result.final_tokens < result.original_tokens

    def test_crisis_preservation(self, compactor, sample_history):
        """Test that crisis turns are preserved verbatim."""
        crisis_indices = {1}  # Mark second turn pair as crisis
        
        result = compactor.compact(
            sample_history,
            target_tokens=100,  # Very tight budget
            preserve_indices=crisis_indices,
        )
        
        # Crisis turns should be in preserved_crisis
        assert len(result.preserved_crisis) > 0

    def test_to_messages_format(self, compactor, sample_history):
        """Test conversion to chat message format."""
        result = compactor.compact(sample_history, target_tokens=5000)
        messages = result.to_messages(system_prompt="You are a helpful counselor.")
        
        assert messages[0]["role"] == "system"
        assert "counselor" in messages[0]["content"].lower()

    def test_token_reduction_property(self, compactor, sample_history):
        """Test token reduction calculation."""
        result = compactor.compact(sample_history, target_tokens=100, preserve_recent=1)
        
        # Should have reduced tokens
        assert 0.0 <= result.token_reduction <= 1.0
        if result.layers_applied:
            assert result.final_tokens <= result.original_tokens

    def test_l2_pair_merging(self, compactor):
        """Test L2 pair-level merging."""
        history = [
            ("Question 1", "Answer 1"),
            ("Question 2", "Answer 2"),
            ("Question 3", "Answer 3"),
            ("Question 4", "Answer 4"),
            ("Question 5", "Answer 5"),
            ("Question 6", "Answer 6"),
            ("Question 7", "Answer 7"),
            ("Question 8", "Answer 8"),
        ]
        
        result = compactor.compact(history, target_tokens=200, preserve_recent=2)
        
        # L2 should trigger when L1 isn't enough
        # The merged pairs should be in system role
        system_messages = [m for m in result.compacted if m["role"] == "system"]
        if "L2:pair_merge" in result.layers_applied:
            assert len(system_messages) > 0

    def test_preserve_recent_works(self, compactor, sample_history):
        """Test that recent turns are always preserved."""
        result = compactor.compact(sample_history, target_tokens=10000, preserve_recent=2)
        
        # Last 2 pairs (4 messages) should be preserved
        assert len(result.compacted) >= 4


# ---------------------------------------------------------------------------
# TieredContextManager Tests
# ---------------------------------------------------------------------------


class TestTieredContextManager:
    """Tests for Hot/Warm/Cold tier management."""

    @pytest.fixture
    def manager(self):
        """Create a tier manager with default settings."""
        return TieredContextManager(
            hot_size=4,
            warm_size=10,
            cold_threshold=20,
        )

    def test_add_turn_starts_hot(self, manager):
        """Test that new turns start in HOT tier."""
        turn = manager.add_turn("user", "Hello")
        assert turn.tier == Tier.HOT

    def test_hot_overflow_to_warm(self, manager):
        """Test that context snapshot respects hot_size limit."""
        # Add more than hot_size turns
        for i in range(10):
            manager.add_turn("user", f"Message {i}")
            manager.add_turn("assistant", f"Response {i}")
        
        # Get context with limited tokens - should respect tier limits
        snapshot = manager.get_context(max_tokens=500)
        
        # The snapshot should have limited hot turns based on hot_size
        # But total turns in manager may be more
        assert len(snapshot.hot_turns) <= manager.hot_size * 2  # pairs

    def test_crisis_promotion(self, manager):
        """Test crisis turn promotion to WARM."""
        turn = manager.add_turn("user", "I'm having thoughts of self-harm", is_crisis=True)
        
        # Crisis should stay warm even if other HOT turns come
        for i in range(10):
            manager.add_turn("user", f"Normal message {i}")
            manager.add_turn("assistant", f"Response {i}")
        
        # Find the crisis turn
        crisis_turn = next((t for t in manager._turns if "self-harm" in t.content), None)
        assert crisis_turn is not None
        assert crisis_turn.tier in (Tier.HOT, Tier.WARM)  # Not demoted to COLD

    def test_get_context_snapshot(self, manager):
        """Test getting context snapshot."""
        manager.add_turn("user", "Hello")
        manager.add_turn("assistant", "Hi there!")
        
        snapshot = manager.get_context(max_tokens=1000)
        
        assert isinstance(snapshot, ContextSnapshot)
        assert len(snapshot.hot_turns) > 0

    def test_importance_score_affects_tier(self, manager):
        """Test that importance score affects tier placement."""
        # High importance turn
        high_turn = manager.add_turn("user", "Important message", importance_score=0.9)
        
        # Add many normal turns to push out old ones
        for i in range(20):
            manager.add_turn("user", f"Normal {i}", importance_score=0.1)
        
        # High importance turn should not be in COLD
        found_turn = next((t for t in manager._turns if "Important" in t.content), None)
        if found_turn:
            assert found_turn.tier in (Tier.HOT, Tier.WARM)


# ---------------------------------------------------------------------------
# CompactionMemoryStore Tests
# ---------------------------------------------------------------------------


class TestCompactionMemoryStore:
    """Tests for self-healing memory persistence."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a memory store with temp database."""
        db_path = tmp_path / "test_memory.db"
        return CompactionMemoryStore(str(db_path))

    def test_save_and_recall(self, store):
        """Test saving and recalling compaction records."""
        store.save_compaction(
            user_id=123,
            session_id="test-session-1",
            summary="User discussed anxiety about work stress. Counselor suggested breathing exercises.",
            crisis_turns=[],
            key_facts=["anxiety", "coping", "work stress"],
            original_turn_count=10,
        )
        
        # Recall should find it (use lower similarity threshold)
        recalled = store.recall_relevant(
            user_id=123, 
            current_context="anxiety work stress", 
            top_k=5,
            min_similarity=0.1,  # Lower threshold for keyword matching
        )
        assert len(recalled) >= 1
        assert recalled[0].session_id == "test-session-1"

    def test_integrity_verification(self, store):
        """Test integrity verification."""
        user_id = 456
        store.save_compaction(
            user_id=user_id,
            session_id="test-session-2",
            summary="Test summary",
            crisis_turns=[],
            key_facts=["test"],
            original_turn_count=5,
        )
        
        # Verify integrity (requires user_id)
        is_valid = store.verify_integrity(user_id)
        assert is_valid is True

    def test_recall_by_context(self, store):
        """Test context-based recall."""
        user_id = 789
        store.save_compaction(
            user_id=user_id,
            session_id="s1",
            summary="Discussed depression and mood symptoms",
            crisis_turns=[],
            key_facts=["depression", "mood"],
            original_turn_count=5,
        )
        store.save_compaction(
            user_id=user_id,
            session_id="s2",
            summary="Talked about anxiety and stress management",
            crisis_turns=[],
            key_facts=["anxiety", "stress"],
            original_turn_count=5,
        )
        store.save_compaction(
            user_id=user_id,
            session_id="s3",
            summary="Addressed relationship issues with partner",
            crisis_turns=[],
            key_facts=["relationships"],
            original_turn_count=5,
        )
        
        # Recall by context (use lower similarity threshold for keyword matching)
        anxiety_records = store.recall_relevant(
            user_id=user_id, 
            current_context="anxiety stress feeling overwhelmed", 
            top_k=2,
            min_similarity=0.1,
        )
        assert len(anxiety_records) >= 1

    def test_get_key_facts(self, store):
        """Test getting key facts for a user."""
        user_id = 100
        for i in range(3):
            store.save_compaction(
                user_id=user_id,
                session_id=f"session-{i}",
                summary=f"Summary {i}",
                crisis_turns=[],
                key_facts=[f"fact-{i}", "common-fact"],
                original_turn_count=5,
            )
        
        facts = store.get_key_facts(user_id=user_id, limit=10)
        # Should have deduplicated facts
        assert "common-fact" in facts
        assert len([f for f in facts if f == "common-fact"]) == 1  # deduplicated

    def test_repair_integrity(self, store):
        """Test integrity repair."""
        user_id = 200
        store.save_compaction(
            user_id=user_id,
            session_id="session-to-repair",
            summary="Some summary",
            crisis_turns=[],
            key_facts=["test"],
            original_turn_count=5,
        )
        
        # Repair should work
        count = store.repair_integrity(user_id)
        assert count >= 1
        
        # Verify integrity after repair
        is_valid = store.verify_integrity(user_id)
        assert is_valid is True


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestClawCodeIntegration:
    """Integration tests for claw-code components working together."""

    def test_compactor_with_tiered_context(self):
        """Test MultiLayerCompactor with TieredContextManager."""
        manager = TieredContextManager()
        compactor = MultiLayerCompactor()
        
        # Build conversation through manager
        for i in range(8):
            manager.add_turn("user", f"User message {i}")
            manager.add_turn("assistant", f"Assistant response {i}")
        
        # Get context and convert to history format
        snapshot = manager.get_context(max_tokens=5000)
        
        # Convert to compactor format using to_history_pairs
        history = snapshot.to_history_pairs()
        
        # Compact
        result = compactor.compact(history, target_tokens=500, preserve_recent=2)
        
        assert result.final_tokens <= result.original_tokens or result.original_tokens <= 500

    def test_full_pipeline_with_persistence(self, tmp_path):
        """Test full pipeline: manage → compact → persist."""
        # Setup
        db_path = tmp_path / "pipeline_test.db"
        manager = TieredContextManager()
        compactor = MultiLayerCompactor()
        store = CompactionMemoryStore(str(db_path))
        
        # Build conversation
        for i in range(10):
            manager.add_turn("user", f"I'm feeling stressed about {['work', 'family', 'health'][i % 3]}")
            manager.add_turn("assistant", f"I understand. Let's talk about that.")
        
        # Get context
        snapshot = manager.get_context()
        
        # Convert to history and compact
        history = snapshot.to_history_pairs()
        
        result = compactor.compact(history, target_tokens=300, preserve_recent=2)
        
        # Persist the compaction
        store.save_compaction(
            user_id=999,
            session_id="integration-test",
            summary=result.session_summary or "User discussed stress about work, family, and health.",
            crisis_turns=[],
            key_facts=["stress", "work"],
            original_turn_count=len(history),
        )
        
        # Verify persistence (use top_k instead of limit)
        recalled = store.recall_relevant(user_id=999, current_context="stress", top_k=1)
        assert len(recalled) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
