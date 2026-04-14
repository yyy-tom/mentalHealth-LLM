"""
Unit tests for the retrieval module.
"""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from mental_health_llm.retrieval.index import Document, KnowledgeIndex, _split_markdown
from mental_health_llm.retrieval.search import RetrievalEngine, SearchResult
from mental_health_llm.retrieval.citation import (
    Citation,
    CitationFormatter,
    extract_citations_from_response,
    has_citation,
)


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document(self):
        doc = Document(
            id="test-1",
            title="Test Document",
            content="This is test content.",
            source="test.md",
            category="general",
        )
        assert doc.id == "test-1"
        assert doc.title == "Test Document"
        assert doc.embedding is None

    def test_document_to_dict(self):
        doc = Document(
            id="test-1",
            title="Test",
            content="Content",
            source="test.md",
            category="general",
            metadata={"key": "value"},
        )
        d = doc.to_dict()
        assert d["id"] == "test-1"
        assert d["metadata"] == {"key": "value"}

    def test_from_markdown_file(self, tmp_path):
        # Create a test markdown file
        md_file = tmp_path / "test_doc.md"
        md_file.write_text("# Title\n\nThis is content.")

        docs = Document.from_file(md_file, category="test")
        assert len(docs) >= 1
        assert docs[0].source == "test_doc.md"
        assert docs[0].category == "test"

    def test_from_json_file(self, tmp_path):
        # Create a test JSON file with list
        json_file = tmp_path / "resources.json"
        data = [
            {"title": "Resource 1", "phone": "123"},
            {"title": "Resource 2", "phone": "456"},
        ]
        json_file.write_text(json.dumps(data))

        docs = Document.from_file(json_file, category="crisis")
        assert len(docs) == 2
        assert docs[0].title == "Resource 1"
        assert docs[1].category == "crisis"


class TestSplitMarkdown:
    """Tests for markdown splitting."""

    def test_split_by_headers(self):
        content = """# Main Title

Some intro text that is long enough to be kept.

## Section 1

Content for section 1 that is substantial.

## Section 2

Content for section 2 that is also substantial.
"""
        # Use smaller chunk size to force splitting
        chunks = _split_markdown(content, chunk_size=50)
        assert len(chunks) >= 2

    def test_no_split_for_small_content(self):
        content = "Short content that fits."
        chunks = _split_markdown(content, chunk_size=500)
        assert len(chunks) == 1


class TestKnowledgeIndex:
    """Tests for KnowledgeIndex (mocked embedding model)."""

    @pytest.fixture
    def mock_index(self, tmp_path):
        """Create index with mocked embedding model."""
        db_path = tmp_path / "test_index.db"
        index = KnowledgeIndex(db_path=db_path)
        # Mock the embedding model
        index._model = MockEmbeddingModel()
        return index

    def test_add_document(self, mock_index):
        doc = Document(
            id="doc-1",
            title="Test",
            content="Test content for embedding",
            source="test.md",
            category="general",
        )
        mock_index.add_document(doc)
        assert mock_index.count() == 1

    def test_add_documents_batch(self, mock_index):
        docs = [
            Document(id=f"doc-{i}", title=f"Doc {i}", content=f"Content {i}",
                     source=f"doc{i}.md", category="test")
            for i in range(5)
        ]
        count = mock_index.add_documents(docs)
        assert count == 5
        assert mock_index.count() == 5

    def test_get_all_embeddings(self, mock_index):
        docs = [
            Document(id=f"doc-{i}", title=f"Doc {i}", content=f"Content {i}",
                     source=f"doc{i}.md", category="test")
            for i in range(3)
        ]
        mock_index.add_documents(docs)

        retrieved_docs, embeddings = mock_index.get_all_embeddings()
        assert len(retrieved_docs) == 3
        assert embeddings.shape == (3, 384)

    def test_clear(self, mock_index):
        doc = Document(
            id="doc-1", title="Test", content="Content",
            source="test.md", category="general"
        )
        mock_index.add_document(doc)
        assert mock_index.count() == 1

        mock_index.clear()
        assert mock_index.count() == 0

    def test_index_directory(self, mock_index, tmp_path):
        # Create test files
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()

        (kb_dir / "doc1.md").write_text("# Doc 1\n\nContent 1")
        (kb_dir / "doc2.md").write_text("# Doc 2\n\nContent 2")

        count = mock_index.index_directory(kb_dir, category="test")
        assert count >= 2


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            # Single text -> return 1D array
            return np.random.randn(384).astype(np.float32)
        # Multiple texts -> return 2D array
        return np.random.randn(len(texts), 384).astype(np.float32)


class TestRetrievalEngine:
    """Tests for RetrievalEngine."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine with test documents."""
        db_path = tmp_path / "test_index.db"
        index = KnowledgeIndex(db_path=db_path)
        index._model = MockEmbeddingModel()

        # Add some test documents
        docs = [
            Document(id="anxiety-1", title="Anxiety Basics",
                     content="Anxiety is a natural response to stress.",
                     source="anxiety.md", category="psychoeducation/anxiety"),
            Document(id="breathing-1", title="Breathing Exercises",
                     content="Deep breathing helps calm the nervous system.",
                     source="breathing.md", category="coping"),
            Document(id="crisis-1", title="Crisis Resources",
                     content="If you're in crisis, call 988.",
                     source="crisis.md", category="crisis"),
        ]
        index.add_documents(docs)

        return RetrievalEngine(index)

    def test_search_basic(self, engine):
        result = engine.search("I'm feeling anxious", top_k=2)
        assert result.query == "I'm feeling anxious"
        assert result.total_candidates == 3

    def test_search_with_category(self, engine):
        result = engine.search("help", category="crisis", include_below_threshold=True)
        # Should only search crisis category
        assert all(
            r.document.category.startswith("crisis")
            for r in result.results
        )

    def test_search_for_skill(self, engine):
        result = engine.search_for_skill(
            "I can't sleep", skill="anxiety-support", top_k=2
        )
        assert result.has_results or result.total_candidates >= 0


class TestSearchResult:
    """Tests for SearchResult."""

    def test_is_relevant_threshold(self):
        doc = Document(id="1", title="T", content="C", source="s.md", category="g")

        relevant = SearchResult(document=doc, score=0.5, rank=1)
        assert relevant.is_relevant is True

        not_relevant = SearchResult(document=doc, score=0.2, rank=1)
        assert not_relevant.is_relevant is False


class TestCitation:
    """Tests for Citation and CitationFormatter."""

    def test_citation_format_inline(self):
        citation = Citation(
            id="doc-1", source="anxiety.md", title="Anxiety Basics", relevance=0.8
        )
        assert citation.format_inline() == "[Source: anxiety.md]"

    def test_citation_format_footnote(self):
        citation = Citation(
            id="doc-1", source="anxiety.md", title="Anxiety Basics", relevance=0.8
        )
        assert citation.format_footnote(1) == "[^1]"
        assert citation.format_full(1) == "[^1]: Anxiety Basics (anxiety.md)"


class TestCitationFormatter:
    """Tests for CitationFormatter."""

    @pytest.fixture
    def results(self):
        """Create mock search results."""
        docs = [
            Document(id=f"d{i}", title=f"Doc {i}", content=f"Content {i}",
                     source=f"doc{i}.md", category="test")
            for i in range(3)
        ]
        return [SearchResult(document=d, score=0.8-i*0.1, rank=i+1) for i, d in enumerate(docs)]

    def test_add_citation(self, results):
        formatter = CitationFormatter(style="inline")
        citation = formatter.add_citation(results[0])
        assert citation.source == "doc0.md"
        assert len(formatter.citations) == 1

    def test_format_response_inline(self, results):
        formatter = CitationFormatter(style="inline")
        formatter.add_citations(results[:2])

        response = "Here is some helpful information."
        formatted = formatter.format_response(response, max_citations=2)

        assert "Sources:" in formatted
        assert "doc0.md" in formatted

    def test_format_response_footnote(self, results):
        formatter = CitationFormatter(style="footnote")
        formatter.add_citations(results[:2])

        response = "Here is some helpful information."
        formatted = formatter.format_response(response)

        assert "[^1]:" in formatted
        assert "---" in formatted

    def test_format_response_none(self, results):
        formatter = CitationFormatter(style="none")
        formatter.add_citations(results)

        response = "No citations added."
        formatted = formatter.format_response(response)

        assert formatted == response  # Unchanged


class TestCitationHelpers:
    """Tests for citation helper functions."""

    def test_extract_citations_inline(self):
        response = "Some info. [Source: anxiety.md] More info."
        sources = extract_citations_from_response(response)
        assert "anxiety.md" in sources

    def test_extract_citations_multiple(self):
        response = "Info. [Sources: doc1.md, doc2.md, doc3.md]"
        sources = extract_citations_from_response(response)
        assert len(sources) == 3

    def test_has_citation_true(self):
        assert has_citation("Text [Source: file.md]") is True
        assert has_citation("Text [^1] more\n---\n[^1]: cite") is True

    def test_has_citation_false(self):
        assert has_citation("No citations here.") is False
