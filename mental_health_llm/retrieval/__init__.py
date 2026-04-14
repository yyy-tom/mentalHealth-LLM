"""
Retrieval module for psychoeducation knowledge base.

Provides embedding-based search over curated mental health content
to ground model responses with evidence-based information.
"""

from .index import KnowledgeIndex, Document
from .search import RetrievalEngine, SearchResult
from .citation import CitationFormatter

__all__ = [
    "KnowledgeIndex",
    "Document",
    "RetrievalEngine",
    "SearchResult",
    "CitationFormatter",
]
