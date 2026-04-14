"""
Unit tests for the tools module.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mental_health_llm.tools.crisis_lookup import (
    CrisisResource,
    crisis_lookup,
    get_crisis_resources,
)
from mental_health_llm.tools.journal_export import (
    export_session,
    export_to_file,
    format_journal_entry,
)


class TestCrisisResource:
    """Tests for CrisisResource dataclass."""

    def test_format_with_phone(self):
        resource = CrisisResource(
            name="Test Helpline",
            phone="123-456",
            hours="24/7",
            languages=["English"],
        )
        formatted = resource.format()
        assert "Test Helpline" in formatted
        assert "123-456" in formatted

    def test_format_with_website(self):
        resource = CrisisResource(
            name="Online Resource",
            phone="",
            hours="24/7",
            languages=["English"],
            website="https://example.com",
        )
        formatted = resource.format()
        assert "https://example.com" in formatted

    def test_format_with_note(self):
        resource = CrisisResource(
            name="Special Line",
            phone="988",
            hours="24/7",
            languages=["English"],
            note="For youth",
        )
        formatted = resource.format()
        assert "For youth" in formatted


class TestGetCrisisResources:
    """Tests for get_crisis_resources function."""

    def test_hk_resources(self):
        resources = get_crisis_resources("HK")
        assert len(resources) >= 1
        assert any("Samaritans" in r.name for r in resources)

    def test_us_resources(self):
        resources = get_crisis_resources("US")
        assert len(resources) >= 1
        assert any("988" in r.phone for r in resources)

    def test_uk_resources(self):
        resources = get_crisis_resources("UK")
        assert len(resources) >= 1

    def test_lowercase_locale(self):
        resources = get_crisis_resources("hk")
        assert len(resources) >= 1

    def test_unknown_locale_falls_back_to_intl(self):
        resources = get_crisis_resources("XX")
        # Should return international resources
        assert len(resources) >= 1
        assert any("iasp" in r.name.lower() or "befrienders" in r.name.lower() for r in resources)


class TestCrisisLookup:
    """Tests for crisis_lookup function."""

    def test_returns_formatted_string(self):
        result = crisis_lookup("HK")
        assert "Crisis Support Resources" in result
        assert "Samaritans" in result

    def test_includes_encouragement(self):
        result = crisis_lookup("US")
        assert "don't have to face this alone" in result.lower()

    def test_unknown_locale_still_returns_help(self):
        result = crisis_lookup("ZZ")
        assert "Crisis Support Resources" in result


class TestFormatJournalEntry:
    """Tests for format_journal_entry function."""

    def test_basic_format(self):
        entry = format_journal_entry(
            user_message="I'm feeling anxious",
            assistant_response="I hear you. Let's explore that together.",
        )
        assert "I'm feeling anxious" in entry
        assert "I hear you" in entry
        assert "You:" in entry
        assert "Counselor:" in entry

    def test_with_skill(self):
        entry = format_journal_entry(
            user_message="Test",
            assistant_response="Response",
            skill="anxiety-support",
        )
        assert "Anxiety Support" in entry

    def test_with_timestamp(self):
        ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        entry = format_journal_entry(
            user_message="Test",
            assistant_response="Response",
            timestamp=ts,
        )
        assert "14:30" in entry


class TestExportSession:
    """Tests for export_session function."""

    def test_empty_history(self):
        result = export_session(user_id=123, history=[])
        assert "Session Journal" in result
        assert "No conversation history" in result

    def test_with_history(self):
        history = [
            ("Hello", "Hi, how can I help?"),
            ("I'm stressed", "I understand. Let's talk about it."),
        ]
        result = export_session(user_id=456, history=history)
        assert "Session Journal" in result
        assert "2 exchanges" in result
        assert "Hello" in result
        assert "I'm stressed" in result

    def test_includes_reflection(self):
        history = [("Test", "Response")]
        result = export_session(user_id=789, history=history, include_summary=True)
        assert "Reflection" in result
        assert "feelings" in result

    def test_excludes_reflection_when_disabled(self):
        history = [("Test", "Response")]
        result = export_session(user_id=789, history=history, include_summary=False)
        assert "Reflection" not in result

    def test_includes_footer(self):
        result = export_session(user_id=1, history=[])
        assert "personal reflection" in result.lower()


class TestExportToFile:
    """Tests for export_to_file function."""

    def test_writes_file(self, tmp_path):
        history = [("Hello", "Hi there!")]
        output_file = tmp_path / "journal.md"

        result = export_to_file(
            user_id=123,
            history=history,
            output_path=str(output_file),
        )

        assert Path(result).exists()
        content = output_file.read_text()
        assert "Session Journal" in content
        assert "Hello" in content
