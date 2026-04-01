"""
Structured session outcomes and analytics for the Mental Health LLM.

Tracks how each counseling session ends with a typed ``SessionOutcome`` enum,
logs outcomes with metadata to a JSONL file for longitudinal analysis, and
provides a ``/stats`` command for the Telegram bot.

Source pattern: claw-code runtime.py typed stop reasons (completed,
max_turns_reached, max_budget_reached).

Usage:

    from mental_health_llm.session_outcome import (
        SessionOutcome, OutcomeLogger, format_stats_report,
    )

    logger = OutcomeLogger("logs/session_outcomes.jsonl")
    logger.log(
        user_id=12345,
        outcome=SessionOutcome.RESOLVED,
        skill="cbt-therapy",
        turns=8,
    )

    # For /stats command
    report = format_stats_report(logger.load_all())
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outcome enum
# ---------------------------------------------------------------------------


class SessionOutcome(str, Enum):
    """How a counseling session ended."""

    RESOLVED = "resolved"
    """User's concern was addressed and conversation ended naturally."""

    CRISIS_ESCALATED = "crisis_escalated"
    """Crisis was detected and emergency resources were provided."""

    SESSION_TIMEOUT = "session_timeout"
    """Session expired due to inactivity."""

    SKILL_SWITCHED = "skill_switched"
    """User was re-routed to a different skill mid-session."""

    USER_ENDED = "user_ended"
    """User explicitly ended the session (e.g. /clear, /start)."""

    ERROR = "error"
    """Session ended due to a generation or system error."""


# ---------------------------------------------------------------------------
# Outcome record
# ---------------------------------------------------------------------------


def _make_record(
    user_id: int,
    outcome: SessionOutcome,
    *,
    skill: str = "",
    model_key: str = "",
    turns: int = 0,
    crisis_detected: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured outcome record."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "user_id": user_id,
        "outcome": outcome.value,
        "skill": skill,
        "model_key": model_key,
        "turns": turns,
        "crisis_detected": crisis_detected,
    }
    if extra:
        record["extra"] = extra
    return record


# ---------------------------------------------------------------------------
# JSONL Logger
# ---------------------------------------------------------------------------


class OutcomeLogger:
    """Append-only JSONL logger for session outcomes."""

    def __init__(self, log_path: str | Path = "logs/session_outcomes.jsonl"):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def log(
        self,
        user_id: int,
        outcome: SessionOutcome,
        *,
        skill: str = "",
        model_key: str = "",
        turns: int = 0,
        crisis_detected: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log a session outcome to the JSONL file.

        Returns the record that was written.
        """
        record = _make_record(
            user_id,
            outcome,
            skill=skill,
            model_key=model_key,
            turns=turns,
            crisis_detected=crisis_detected,
            extra=extra,
        )
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Failed to write outcome log to %s", self._path)

        return record

    def load_all(self) -> list[dict[str, Any]]:
        """Load all outcome records from the JSONL file."""
        if not self._path.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(self._path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed line %d in %s", line_no, self._path
                    )
        return records

    def load_for_user(self, user_id: int) -> list[dict[str, Any]]:
        """Load outcome records for a specific user."""
        return [r for r in self.load_all() if r.get("user_id") == user_id]


# ---------------------------------------------------------------------------
# Stats formatting (for /stats command)
# ---------------------------------------------------------------------------


def format_stats_report(
    records: list[dict[str, Any]],
    *,
    user_id: int | None = None,
) -> str:
    """Format outcome records into a human-readable stats report.

    Args:
        records: List of outcome record dicts.
        user_id: If set, filter to this user and label accordingly.

    Returns:
        Formatted Telegram-friendly text report.
    """
    if user_id is not None:
        records = [r for r in records if r.get("user_id") == user_id]

    if not records:
        scope = "your" if user_id else "all"
        return f"No session data recorded yet for {scope} sessions."

    # Outcome distribution
    outcome_counts: dict[str, int] = {}
    skill_counts: dict[str, int] = {}
    total_turns = 0
    crisis_count = 0

    for r in records:
        outcome = r.get("outcome", "unknown")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        skill = r.get("skill", "")
        if skill:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

        total_turns += r.get("turns", 0)

        if r.get("crisis_detected"):
            crisis_count += 1

    total_sessions = len(records)
    avg_turns = total_turns / total_sessions if total_sessions else 0

    # Build report
    scope_label = "Your" if user_id else "All"
    lines = [f"{scope_label} Session Statistics\n"]

    lines.append(f"Total sessions: {total_sessions}")
    lines.append(f"Average turns per session: {avg_turns:.1f}")

    if crisis_count:
        lines.append(f"Crisis detections: {crisis_count}")

    # Outcome breakdown
    lines.append("\nOutcome Distribution:")
    outcome_labels = {
        "resolved": "Resolved",
        "crisis_escalated": "Crisis Escalated",
        "session_timeout": "Timed Out",
        "skill_switched": "Skill Switched",
        "user_ended": "User Ended",
        "error": "Error",
    }
    for outcome_val, count in sorted(
        outcome_counts.items(), key=lambda x: x[1], reverse=True
    ):
        label = outcome_labels.get(outcome_val, outcome_val)
        pct = (count / total_sessions) * 100
        bar = _bar(pct)
        lines.append(f"  {label}: {count} ({pct:.0f}%) {bar}")

    # Skill usage
    if skill_counts:
        lines.append("\nSkill Usage:")
        for skill_name, count in sorted(
            skill_counts.items(), key=lambda x: x[1], reverse=True
        ):
            pct = (count / total_sessions) * 100
            lines.append(f"  {skill_name}: {count} ({pct:.0f}%)")

    return "\n".join(lines)


def _bar(pct: float, width: int = 10) -> str:
    """Render a simple text progress bar."""
    filled = round(pct / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"
