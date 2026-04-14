"""
Citation formatting for retrieved content.

Provides utilities for formatting knowledge base citations
in model responses for transparency and verifiability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .search import SearchResult


@dataclass
class Citation:
    """A citation reference."""

    id: str
    source: str
    title: str
    relevance: float

    def format_inline(self) -> str:
        """Format as inline citation [Source: filename]."""
        return f"[Source: {self.source}]"

    def format_footnote(self, index: int) -> str:
        """Format as footnote reference [^index]."""
        return f"[^{index}]"

    def format_full(self, index: int) -> str:
        """Format as full citation for footer."""
        return f"[^{index}]: {self.title} ({self.source})"


class CitationFormatter:
    """Utility for adding citations to model responses.

    Tracks which KB documents were used and formats appropriate
    citations in the response text.
    """

    def __init__(self, style: str = "inline"):
        """Initialize formatter.

        Args:
            style: Citation style - "inline", "footnote", or "none".
        """
        if style not in ("inline", "footnote", "none"):
            raise ValueError(f"Unknown citation style: {style}")
        self.style = style
        self._citations: list[Citation] = []

    def add_citation(self, result: SearchResult) -> Citation:
        """Add a citation from a search result."""
        citation = Citation(
            id=result.document.id,
            source=result.document.source,
            title=result.document.title,
            relevance=result.score,
        )
        self._citations.append(citation)
        return citation

    def add_citations(self, results: list[SearchResult]) -> list[Citation]:
        """Add multiple citations from search results."""
        return [self.add_citation(r) for r in results]

    @property
    def citations(self) -> list[Citation]:
        """Get all added citations."""
        return self._citations

    def format_response(
        self,
        response: str,
        max_citations: int = 3,
    ) -> str:
        """Add citation markers and footer to response.

        Args:
            response: Model response text.
            max_citations: Maximum citations to include.

        Returns:
            Response with citations added.
        """
        if self.style == "none" or not self._citations:
            return response

        # Take top citations by relevance
        top_citations = sorted(
            self._citations, key=lambda c: c.relevance, reverse=True
        )[:max_citations]

        if self.style == "inline":
            # Add inline citations at end
            sources = ", ".join(c.source for c in top_citations)
            return f"{response}\n\n[Sources: {sources}]"

        elif self.style == "footnote":
            # Add footnote markers and footer
            footer_lines = []
            for i, citation in enumerate(top_citations, start=1):
                footer_lines.append(citation.format_full(i))

            footer = "\n".join(footer_lines)
            return f"{response}\n\n---\n{footer}"

        return response

    def format_context_with_citations(
        self,
        results: list[SearchResult],
        max_results: int = 3,
    ) -> tuple[str, list[Citation]]:
        """Format search results as context with tracked citations.

        Args:
            results: Search results to format.
            max_results: Maximum results to include.

        Returns:
            Tuple of (formatted context, citations list).
        """
        if not results:
            return "", []

        citations = []
        lines = ["[Relevant Context]", ""]

        for i, result in enumerate(results[:max_results], start=1):
            doc = result.document
            citation = self.add_citation(result)
            citations.append(citation)

            # Truncate content to ~400 chars
            content = doc.content
            if len(content) > 400:
                content = content[:397] + "..."

            lines.append(f"**[{i}] {doc.title}**")
            lines.append(content)
            lines.append("")

        lines.append("[End Context]")

        return "\n".join(lines), citations

    def clear(self) -> None:
        """Clear all tracked citations."""
        self._citations = []


def extract_citations_from_response(response: str) -> list[str]:
    """Extract source citations from a model response.

    Looks for patterns like [Source: filename] or [Sources: a, b, c].

    Args:
        response: Model response text.

    Returns:
        List of extracted source names.
    """
    # Match [Source: ...] or [Sources: ...]
    pattern = r"\[Sources?:\s*([^\]]+)\]"
    matches = re.findall(pattern, response)

    sources = []
    for match in matches:
        # Split comma-separated sources
        for source in match.split(","):
            source = source.strip()
            if source:
                sources.append(source)

    return sources


def has_citation(response: str) -> bool:
    """Check if response contains any citation markers.

    Args:
        response: Model response text.

    Returns:
        True if citations are present.
    """
    patterns = [
        r"\[Sources?:",  # Inline citations
        r"\[\^\d+\]",  # Footnote markers
        r"---\n\[\^",  # Footnote footer
    ]
    return any(re.search(p, response) for p in patterns)
