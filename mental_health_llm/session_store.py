"""
SQLite-backed session persistence for mental health counseling.

Stores conversation history, crisis flags, active skill, model key, and
timestamps per user.  Auto-saves after each turn and supports session
resumption on bot restart.  Sessions expire after a configurable window
(default 24 hours) with automatic cleanup.

Source pattern: claw-code session.rs (JSON-serialized sessions with
save_to_path / load_from_path).

Usage:

    from mental_health_llm.session_store import SQLiteSessionStore

    store = SQLiteSessionStore("data/sessions.db")

    # After each exchange
    store.save_turn(user_id=123, user_msg="I feel anxious",
                    assistant_msg="I hear you...", skill="cbt-therapy")

    # On bot restart — restore session
    session = store.load_session(user_id=123)
    if session:
        history = session["messages"]  # list of [user, assistant] pairs

    # Periodic maintenance
    removed = store.cleanup_expired()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SQLiteSessionStore:
    """SQLite-backed session storage with auto-save and expiry.

    Each user has at most one active session.  The session stores the full
    conversation history as a JSON array of ``[user_msg, assistant_msg]``
    pairs, plus metadata (crisis flags, active skill, model key).
    """

    def __init__(
        self,
        db_path: str | Path = "data/sessions.db",
        expiry_hours: float = 24.0,
    ) -> None:
        self._db_path = str(db_path)
        self._expiry_seconds = expiry_hours * 3600
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        removed = self.cleanup_expired()
        if removed:
            logger.info("Cleaned up %d expired sessions", removed)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id    INTEGER PRIMARY KEY,
                    messages   TEXT    NOT NULL DEFAULT '[]',
                    crisis_flags TEXT  NOT NULL DEFAULT '[]',
                    active_skill TEXT  NOT NULL DEFAULT '',
                    model_key  TEXT    NOT NULL DEFAULT '',
                    created_at REAL   NOT NULL,
                    updated_at REAL   NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_turn(
        self,
        user_id: int,
        user_msg: str,
        assistant_msg: str,
        *,
        skill: str = "",
        is_crisis: bool = False,
        model_key: str = "",
    ) -> None:
        """Append a conversation turn to the user's session.

        Creates the session row on first call; subsequent calls append to
        the existing message list.  Call this after every exchange.
        """
        now = time.time()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT messages, crisis_flags, created_at "
                "FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if row is not None:
                messages: list = json.loads(row[0])
                crisis_flags: list[int] = json.loads(row[1])
                created_at = row[2]
            else:
                messages = []
                crisis_flags = []
                created_at = now

            messages.append([user_msg, assistant_msg])
            if is_crisis:
                crisis_flags.append(len(messages) - 1)

            conn.execute(
                """
                INSERT INTO sessions
                    (user_id, messages, crisis_flags, active_skill,
                     model_key, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    messages     = excluded.messages,
                    crisis_flags = excluded.crisis_flags,
                    active_skill = excluded.active_skill,
                    model_key    = CASE
                        WHEN excluded.model_key != ''
                        THEN excluded.model_key
                        ELSE sessions.model_key
                    END,
                    updated_at   = excluded.updated_at
                """,
                (
                    user_id,
                    json.dumps(messages, ensure_ascii=False),
                    json.dumps(crisis_flags),
                    skill,
                    model_key,
                    created_at,
                    now,
                ),
            )

    def update_model_key(self, user_id: int, model_key: str) -> None:
        """Update the model key for an existing session."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET model_key = ?, updated_at = ? "
                "WHERE user_id = ?",
                (model_key, now, user_id),
            )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def load_session(self, user_id: int) -> Optional[dict[str, Any]]:
        """Load a session if it exists and hasn't expired.

        Returns:
            Dict with keys: user_id, messages, crisis_flags, active_skill,
            model_key, created_at, updated_at.  Or ``None`` if no valid
            session exists.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT messages, crisis_flags, active_skill, model_key, "
                "       created_at, updated_at "
                "FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        if row is None:
            return None

        # Check expiry
        if time.time() - row[5] > self._expiry_seconds:
            self.delete_session(user_id)
            return None

        messages = json.loads(row[0])
        crisis_flags = json.loads(row[1])

        return {
            "user_id": user_id,
            "messages": messages,
            "crisis_flags": crisis_flags,
            "active_skill": row[2],
            "model_key": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }

    def has_session(self, user_id: int) -> bool:
        """Check whether a non-expired session exists."""
        return self.load_session(user_id) is not None

    # ------------------------------------------------------------------
    # Delete / cleanup
    # ------------------------------------------------------------------

    def delete_session(self, user_id: int) -> None:
        """Delete a user's session."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE user_id = ?", (user_id,)
            )

    def cleanup_expired(self) -> int:
        """Remove all sessions older than the expiry window.

        Returns:
            Number of sessions removed.
        """
        cutoff = time.time() - self._expiry_seconds
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Helpers for Telegram bot integration
    # ------------------------------------------------------------------

    def restore_history(self, user_id: int) -> list[tuple[str, str]]:
        """Load session and return history as (user, assistant) tuples.

        Convenience method for drop-in replacement of the in-memory
        ``user_histories`` dict in telegram_bot.py.
        """
        session = self.load_session(user_id)
        if session is None:
            return []
        return [tuple(pair) for pair in session["messages"]]

    def restore_crisis_flags(self, user_id: int) -> set[int]:
        """Load crisis turn indices for a user."""
        session = self.load_session(user_id)
        if session is None:
            return set()
        return set(session["crisis_flags"])
