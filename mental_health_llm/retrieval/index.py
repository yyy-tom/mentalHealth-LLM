"""
Knowledge base indexing for psychoeducation content.

Uses sentence-transformers for embedding generation and stores
embeddings in a simple SQLite database for persistence.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Document:
    """A knowledge base document."""

    id: str
    title: str
    content: str
    source: str
    category: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_file(
        cls,
        path: Path,
        category: str = "general",
        chunk_size: int = 512,
    ) -> list["Document"]:
        """Load document(s) from a file.

        Args:
            path: Path to markdown or JSON file.
            category: Category tag (e.g., "anxiety", "crisis", "coping").
            chunk_size: Max characters per chunk (for splitting long docs).

        Returns:
            List of Document objects (one per chunk if file is large).
        """
        content = path.read_text(encoding="utf-8")
        source = path.name
        title = path.stem.replace("_", " ").title()

        # For JSON files, extract structured content
        if path.suffix == ".json":
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    # List of items (e.g., crisis resources)
                    docs = []
                    for i, item in enumerate(data):
                        item_content = json.dumps(item, indent=2)
                        item_id = f"{path.stem}_{i}"
                        docs.append(
                            cls(
                                id=item_id,
                                title=item.get("title", f"{title} #{i+1}"),
                                content=item_content,
                                source=source,
                                category=category,
                                metadata={"index": i},
                            )
                        )
                    return docs
                else:
                    content = json.dumps(data, indent=2)
            except json.JSONDecodeError:
                pass  # Treat as plain text

        # Split long markdown content into chunks
        if len(content) > chunk_size:
            chunks = _split_markdown(content, chunk_size)
            docs = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{path.stem}_{i}"
                docs.append(
                    cls(
                        id=chunk_id,
                        title=f"{title} (Part {i+1})" if len(chunks) > 1 else title,
                        content=chunk,
                        source=source,
                        category=category,
                        metadata={"chunk_index": i, "total_chunks": len(chunks)},
                    )
                )
            return docs

        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        return [
            cls(
                id=doc_id,
                title=title,
                content=content,
                source=source,
                category=category,
            )
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "category": self.category,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


def _split_markdown(content: str, chunk_size: int) -> list[str]:
    """Split markdown content into chunks, respecting headers.

    Splits at ## headers first, then paragraphs if still too long.
    """
    chunks = []
    current_chunk = ""

    # Split by headers first
    lines = content.split("\n")
    for line in lines:
        if line.startswith("## ") and current_chunk:
            # New section — save current chunk if substantial
            if len(current_chunk.strip()) > 50:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            if len(current_chunk) + len(line) > chunk_size:
                # Current chunk is full
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


class KnowledgeIndex:
    """Embedding index for knowledge base documents.

    Uses a lightweight sentence-transformer model for embeddings
    and stores everything in SQLite for persistence.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Small, fast, good quality
    EMBEDDING_DIM = 384

    def __init__(
        self,
        db_path: str | Path = "data/kb/index.db",
        model_name: str = DEFAULT_MODEL,
    ):
        """Initialize the knowledge index.

        Args:
            db_path: Path to SQLite database for storing embeddings.
            model_name: Sentence-transformer model name.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON documents(category)
            """)

    @property
    def model(self):
        """Lazy-load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for retrieval. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def add_document(self, doc: Document) -> None:
        """Add a document to the index.

        Computes embedding if not already present.
        """
        if doc.embedding is None:
            doc.embedding = self._embed(doc.content)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (id, title, content, source, category, embedding, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    doc.id,
                    doc.title,
                    doc.content,
                    doc.source,
                    doc.category,
                    doc.embedding.tobytes(),
                    json.dumps(doc.metadata),
                    doc.created_at,
                ),
            )

    def add_documents(self, docs: list[Document]) -> int:
        """Add multiple documents to the index.

        Returns:
            Number of documents added.
        """
        # Batch embed for efficiency
        texts = [d.content for d in docs]
        embeddings = self._embed_batch(texts)

        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb
            self.add_document(doc)

        return len(docs)

    def index_directory(
        self,
        directory: str | Path,
        category: str = "general",
    ) -> int:
        """Index all markdown and JSON files in a directory.

        Args:
            directory: Path to directory containing KB files.
            category: Category tag for all documents.

        Returns:
            Number of documents indexed.
        """
        directory = Path(directory)
        if not directory.exists():
            return 0

        all_docs = []
        for pattern in ["*.md", "*.json"]:
            for path in directory.glob(pattern):
                docs = Document.from_file(path, category=category)
                all_docs.extend(docs)

        # Also index subdirectories
        for subdir in directory.iterdir():
            if subdir.is_dir():
                sub_category = f"{category}/{subdir.name}"
                for pattern in ["*.md", "*.json"]:
                    for path in subdir.glob(pattern):
                        docs = Document.from_file(path, category=sub_category)
                        all_docs.extend(docs)

        if all_docs:
            return self.add_documents(all_docs)
        return 0

    def get_all_embeddings(self) -> tuple[list[Document], np.ndarray]:
        """Retrieve all documents and their embeddings.

        Returns:
            Tuple of (documents, embeddings array).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM documents").fetchall()

        docs = []
        embeddings = []
        for row in rows:
            doc = Document(
                id=row["id"],
                title=row["title"],
                content=row["content"],
                source=row["source"],
                category=row["category"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
            )
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            doc.embedding = emb
            docs.append(doc)
            embeddings.append(emb)

        if embeddings:
            return docs, np.stack(embeddings)
        return docs, np.array([])

    def count(self) -> int:
        """Return number of indexed documents."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def clear(self) -> None:
        """Clear all documents from the index."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents")

    def _embed(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Compute embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb for emb in embeddings]
