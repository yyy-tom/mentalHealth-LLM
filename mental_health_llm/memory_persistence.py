"""
Compaction memory persistence for cross-session recall.

Extends session_store.py by persisting compaction summaries, enabling
self-healing memory that survives restarts and supports semantic retrieval
of relevant past context.

Source pattern: claw-code .claude/memory/ persistence with reconciliation.

Usage:
    from mental_health_llm.memory_persistence import CompactionMemoryStore

    store = CompactionMemoryStore("data/memory.db")

    # After compaction
    store.save_compaction(
        user_id=123,
        session_id="abc",
        summary="User discussed exam anxiety...",
        crisis_turns=[...],
        key_facts=["final exams next week", "pre-med student"],
    )

    # On session restore — retrieve relevant past context
    relevant = store.recall_relevant(user_id=123, current_context="feeling anxious")
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompactionRecord:
    """A persisted compaction summary."""

    record_id: str
    user_id: int
    session_id: str
    summary: str
    crisis_turns: list[dict]
    key_facts: list[str]
    original_turn_count: int
    created_at: float
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "summary": self.summary,
            "crisis_turns": self.crisis_turns,
            "key_facts": self.key_facts,
            "original_turn_count": self.original_turn_count,
            "created_at": self.created_at,
        }


class CompactionMemoryStore:
    """SQLite-backed compaction memory with optional embedding search.

    Persists compaction summaries so that:
    1. Sessions can be fully restored after crashes (self-healing)
    2. Relevant past sessions can be recalled for context (cross-session memory)
    3. Key facts are preserved for long-term user understanding
    """

    def __init__(
        self,
        db_path: str | Path = "data/memory.db",
        max_records_per_user: int = 50,
        embedding_fn: Optional[callable] = None,
    ) -> None:
        """Initialize the compaction memory store.

        Args:
            db_path: Path to SQLite database file.
            max_records_per_user: Maximum compaction records to keep per user.
            embedding_fn: Optional function to compute embeddings for semantic search.
                          Signature: (text: str) -> list[float]
        """
        self._db_path = str(db_path)
        self._max_records = max_records_per_user
        self._embedding_fn = embedding_fn
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compaction_memory (
                    record_id   TEXT PRIMARY KEY,
                    user_id     INTEGER NOT NULL,
                    session_id  TEXT NOT NULL,
                    summary     TEXT NOT NULL,
                    crisis_turns TEXT NOT NULL DEFAULT '[]',
                    key_facts   TEXT NOT NULL DEFAULT '[]',
                    original_turn_count INTEGER NOT NULL DEFAULT 0,
                    embedding   BLOB,
                    created_at  REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_compaction_user
                ON compaction_memory(user_id, created_at DESC)
            """)
            # Integrity tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_integrity (
                    user_id     INTEGER PRIMARY KEY,
                    checksum    TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    last_verified REAL NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_compaction(
        self,
        user_id: int,
        session_id: str,
        summary: str,
        crisis_turns: list[dict],
        key_facts: list[str],
        original_turn_count: int = 0,
    ) -> str:
        """Save a compaction result for future retrieval.

        Args:
            user_id: User identifier.
            session_id: Unique session identifier.
            summary: The compacted summary text.
            crisis_turns: List of crisis turn dicts (preserved verbatim).
            key_facts: Extracted key facts for quick recall.
            original_turn_count: Number of turns before compaction.

        Returns:
            The generated record_id.
        """
        now = time.time()
        record_id = self._generate_record_id(user_id, session_id, now)

        # Compute embedding if function provided
        embedding_blob = None
        if self._embedding_fn is not None:
            try:
                embedding = self._embedding_fn(summary)
                embedding_blob = self._serialize_embedding(embedding)
            except Exception as e:
                logger.warning("Failed to compute embedding: %s", e)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO compaction_memory
                    (record_id, user_id, session_id, summary, crisis_turns,
                     key_facts, original_turn_count, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    summary = excluded.summary,
                    crisis_turns = excluded.crisis_turns,
                    key_facts = excluded.key_facts,
                    original_turn_count = excluded.original_turn_count,
                    embedding = excluded.embedding
                """,
                (
                    record_id,
                    user_id,
                    session_id,
                    summary,
                    json.dumps(crisis_turns, ensure_ascii=False),
                    json.dumps(key_facts, ensure_ascii=False),
                    original_turn_count,
                    embedding_blob,
                    now,
                ),
            )

            # Enforce per-user limit
            self._enforce_limit(conn, user_id)

            # Update integrity checksum
            self._update_integrity(conn, user_id)

        logger.debug("Saved compaction record %s for user %d", record_id, user_id)
        return record_id

    def _generate_record_id(self, user_id: int, session_id: str, timestamp: float) -> str:
        """Generate a unique record ID."""
        content = f"{user_id}:{session_id}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _enforce_limit(self, conn: sqlite3.Connection, user_id: int) -> None:
        """Delete oldest records if user exceeds max_records."""
        conn.execute(
            """
            DELETE FROM compaction_memory
            WHERE record_id IN (
                SELECT record_id FROM compaction_memory
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (user_id, self._max_records),
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def load_latest(self, user_id: int) -> Optional[CompactionRecord]:
        """Load the most recent compaction for a user."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT record_id, user_id, session_id, summary, crisis_turns,
                       key_facts, original_turn_count, created_at
                FROM compaction_memory
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

        if row is None:
            return None

        return CompactionRecord(
            record_id=row[0],
            user_id=row[1],
            session_id=row[2],
            summary=row[3],
            crisis_turns=json.loads(row[4]),
            key_facts=json.loads(row[5]),
            original_turn_count=row[6],
            created_at=row[7],
        )

    def load_by_session(self, session_id: str) -> Optional[CompactionRecord]:
        """Load compaction for a specific session."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT record_id, user_id, session_id, summary, crisis_turns,
                       key_facts, original_turn_count, created_at
                FROM compaction_memory
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        return CompactionRecord(
            record_id=row[0],
            user_id=row[1],
            session_id=row[2],
            summary=row[3],
            crisis_turns=json.loads(row[4]),
            key_facts=json.loads(row[5]),
            original_turn_count=row[6],
            created_at=row[7],
        )

    def recall_relevant(
        self,
        user_id: int,
        current_context: str,
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> list[CompactionRecord]:
        """Retrieve relevant past compaction summaries.

        If embedding_fn was provided, uses cosine similarity for semantic search.
        Otherwise, falls back to keyword overlap scoring.

        Args:
            user_id: User to search for.
            current_context: Current conversation context to match against.
            top_k: Maximum number of records to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of relevant CompactionRecords, most relevant first.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT record_id, user_id, session_id, summary, crisis_turns,
                       key_facts, original_turn_count, embedding, created_at
                FROM compaction_memory
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (user_id,),
            ).fetchall()

        if not rows:
            return []

        records_with_scores: list[tuple[float, CompactionRecord]] = []

        # Compute query embedding if available
        query_embedding = None
        if self._embedding_fn is not None:
            try:
                query_embedding = self._embedding_fn(current_context)
            except Exception as e:
                logger.warning("Failed to compute query embedding: %s", e)

        for row in rows:
            record = CompactionRecord(
                record_id=row[0],
                user_id=row[1],
                session_id=row[2],
                summary=row[3],
                crisis_turns=json.loads(row[4]),
                key_facts=json.loads(row[5]),
                original_turn_count=row[6],
                created_at=row[8],
            )

            # Compute similarity score
            if query_embedding is not None and row[7] is not None:
                doc_embedding = self._deserialize_embedding(row[7])
                score = self._cosine_similarity(query_embedding, doc_embedding)
            else:
                # Fallback to keyword overlap
                score = self._keyword_overlap(current_context, record.summary)

            if score >= min_similarity:
                records_with_scores.append((score, record))

        # Sort by score descending and return top_k
        records_with_scores.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in records_with_scores[:top_k]]

    def get_key_facts(self, user_id: int, limit: int = 20) -> list[str]:
        """Get all key facts for a user, deduplicated and sorted by recency."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key_facts FROM compaction_memory
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        seen: set[str] = set()
        facts: list[str] = []
        for row in rows:
            for fact in json.loads(row[0]):
                normalized = fact.strip().lower()
                if normalized not in seen:
                    seen.add(normalized)
                    facts.append(fact)

        return facts

    # ------------------------------------------------------------------
    # Integrity & Recovery (Self-Healing)
    # ------------------------------------------------------------------

    def verify_integrity(self, user_id: int) -> bool:
        """Verify data integrity for a user's memory records.

        Returns True if integrity check passes, False if corruption detected.
        """
        with self._connect() as conn:
            # Get stored integrity info
            integrity_row = conn.execute(
                "SELECT checksum, record_count FROM memory_integrity WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if integrity_row is None:
                # No integrity record — rebuild it
                self._update_integrity(conn, user_id)
                return True

            stored_checksum, stored_count = integrity_row

            # Compute current checksum
            current_checksum, current_count = self._compute_checksum(conn, user_id)

            if stored_checksum != current_checksum or stored_count != current_count:
                logger.warning(
                    "Integrity check failed for user %d: expected %s/%d, got %s/%d",
                    user_id,
                    stored_checksum,
                    stored_count,
                    current_checksum,
                    current_count,
                )
                return False

        return True

    def repair_integrity(self, user_id: int) -> int:
        """Attempt to repair integrity issues.

        Currently just rebuilds the integrity record. Future: could attempt
        to recover from backup or reconstruct from session store.

        Returns:
            Number of records after repair.
        """
        with self._connect() as conn:
            # Delete any malformed records
            conn.execute(
                """
                DELETE FROM compaction_memory
                WHERE user_id = ? AND (
                    summary IS NULL OR
                    json_valid(crisis_turns) = 0 OR
                    json_valid(key_facts) = 0
                )
                """,
                (user_id,),
            )

            # Rebuild integrity
            self._update_integrity(conn, user_id)

            # Return current count
            row = conn.execute(
                "SELECT COUNT(*) FROM compaction_memory WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        return row[0] if row else 0

    def _update_integrity(self, conn: sqlite3.Connection, user_id: int) -> None:
        """Update the integrity record for a user."""
        checksum, count = self._compute_checksum(conn, user_id)
        now = time.time()
        conn.execute(
            """
            INSERT INTO memory_integrity (user_id, checksum, record_count, last_verified)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                checksum = excluded.checksum,
                record_count = excluded.record_count,
                last_verified = excluded.last_verified
            """,
            (user_id, checksum, count, now),
        )

    def _compute_checksum(self, conn: sqlite3.Connection, user_id: int) -> tuple[str, int]:
        """Compute a checksum over all records for a user."""
        rows = conn.execute(
            """
            SELECT record_id, summary FROM compaction_memory
            WHERE user_id = ?
            ORDER BY created_at
            """,
            (user_id,),
        ).fetchall()

        if not rows:
            return "", 0

        content = "|".join(f"{r[0]}:{len(r[1])}" for r in rows)
        checksum = hashlib.md5(content.encode()).hexdigest()
        return checksum, len(rows)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_user_memory(self, user_id: int) -> int:
        """Delete all memory records for a user.

        Returns:
            Number of records deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM compaction_memory WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM memory_integrity WHERE user_id = ?",
                (user_id,),
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Embedding utilities
    # ------------------------------------------------------------------

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize embedding to bytes for storage."""
        import struct

        return struct.pack(f"{len(embedding)}f", *embedding)

    def _deserialize_embedding(self, blob: bytes) -> list[float]:
        """Deserialize embedding from bytes."""
        import struct

        n = len(blob) // 4  # 4 bytes per float
        return list(struct.unpack(f"{n}f", blob))

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _keyword_overlap(self, query: str, document: str) -> float:
        """Simple keyword overlap scoring as embedding fallback."""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        if not query_words or not doc_words:
            return 0.0

        overlap = len(query_words & doc_words)
        return overlap / max(len(query_words), 1)
