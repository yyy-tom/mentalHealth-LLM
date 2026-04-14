"""
Semantic search over the psychoeducation knowledge base.

Provides cosine similarity search with configurable top-k
and optional category filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .index import Document, KnowledgeIndex


@dataclass
class SearchResult:
    """A single search result with relevance score."""

    document: Document
    score: float  # Cosine similarity (0-1)
    rank: int

    @property
    def is_relevant(self) -> bool:
        """Check if result is relevant (score > 0.3)."""
        return self.score > 0.3

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "rank": self.rank,
        }


@dataclass
class RetrievalResult:
    """Collection of search results with metadata."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    total_candidates: int = 0
    search_time_ms: float = 0.0

    @property
    def has_results(self) -> bool:
        """Check if any results were found."""
        return len(self.results) > 0

    @property
    def top_result(self) -> Optional[SearchResult]:
        """Get the top-scoring result."""
        return self.results[0] if self.results else None

    def format_context(self, max_results: int = 3, max_chars: int = 1500) -> str:
        """Format results as context for model prompt.

        Args:
            max_results: Maximum number of results to include.
            max_chars: Maximum total characters.

        Returns:
            Formatted context string with citations.
        """
        if not self.results:
            return ""

        lines = ["[Relevant Context]", ""]
        char_count = 0

        for result in self.results[:max_results]:
            doc = result.document
            chunk = f"**{doc.title}** (Source: {doc.source})\n{doc.content[:500]}..."

            if char_count + len(chunk) > max_chars:
                break

            lines.append(chunk)
            lines.append("")
            char_count += len(chunk)

        lines.append("[End Context]")
        return "\n".join(lines)


class RetrievalEngine:
    """Semantic search engine for the knowledge base.

    Uses cosine similarity between query embedding and
    document embeddings for retrieval.
    """

    def __init__(
        self,
        index: KnowledgeIndex,
        default_top_k: int = 5,
        relevance_threshold: float = 0.3,
    ):
        """Initialize the retrieval engine.

        Args:
            index: KnowledgeIndex instance with indexed documents.
            default_top_k: Default number of results to return.
            relevance_threshold: Minimum score to consider relevant.
        """
        self.index = index
        self.default_top_k = default_top_k
        self.relevance_threshold = relevance_threshold
        self._docs: list[Document] = []
        self._embeddings: Optional[np.ndarray] = None

    def _load_index(self) -> None:
        """Load documents and embeddings from index."""
        if self._embeddings is None or len(self._docs) != self.index.count():
            self._docs, self._embeddings = self.index.get_all_embeddings()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        include_below_threshold: bool = False,
    ) -> RetrievalResult:
        """Search for relevant documents.

        Args:
            query: User query string.
            top_k: Number of results to return (default: default_top_k).
            category: Filter by category (optional).
            include_below_threshold: Include results below relevance threshold.

        Returns:
            RetrievalResult with ranked documents.
        """
        import time

        start_time = time.perf_counter()

        self._load_index()

        if len(self._docs) == 0:
            return RetrievalResult(query=query)

        k = top_k or self.default_top_k

        # Compute query embedding
        query_embedding = self.index._embed(query)

        # Filter by category if specified
        if category:
            mask = np.array(
                [d.category.startswith(category) for d in self._docs]
            )
            filtered_docs = [d for d, m in zip(self._docs, mask) if m]
            filtered_embeddings = self._embeddings[mask]
        else:
            filtered_docs = self._docs
            filtered_embeddings = self._embeddings

        if len(filtered_docs) == 0:
            return RetrievalResult(query=query)

        # Compute cosine similarity
        scores = self._cosine_similarity(query_embedding, filtered_embeddings)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if not include_below_threshold and score < self.relevance_threshold:
                continue
            results.append(
                SearchResult(
                    document=filtered_docs[idx],
                    score=score,
                    rank=rank,
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            query=query,
            results=results,
            total_candidates=len(filtered_docs),
            search_time_ms=elapsed_ms,
        )

    def search_for_skill(
        self,
        query: str,
        skill: str,
        top_k: int = 3,
    ) -> RetrievalResult:
        """Search with skill-aware category mapping.

        Maps skill names to KB categories for more relevant results.

        Args:
            query: User query string.
            skill: Detected skill name (e.g., "anxiety-support").
            top_k: Number of results.

        Returns:
            RetrievalResult with skill-relevant documents.
        """
        # Map skills to KB categories
        skill_to_category = {
            "anxiety-support": "psychoeducation/anxiety",
            "depression-support": "psychoeducation/depression",
            "sleep-support": "psychoeducation/sleep",
            "stress-management": "coping",
            "crisis-intervention": "crisis",
            "general-support": None,  # Search all
        }

        category = skill_to_category.get(skill)
        result = self.search(query, top_k=top_k, category=category)

        # If category search yields few results, search all
        if len(result.results) < 2 and category is not None:
            all_result = self.search(query, top_k=top_k, category=None)
            if len(all_result.results) > len(result.results):
                return all_result

        return result

    def _cosine_similarity(
        self, query: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and all embeddings."""
        # Normalize
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        emb_norms = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Dot product
        return np.dot(emb_norms, query_norm)
