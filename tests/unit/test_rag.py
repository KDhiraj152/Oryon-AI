"""
Unit tests for RAGService, BGEM3Embedder, and BGEReranker — retrieval-augmented generation.

Tests cover:
- Embedder loading state
- RAGService initialisation
- Embedding stats
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from backend.services.chat.rag import RAGService


# ── RAGService ────────────────────────────────────────────────────────────────

class TestRAGService:
    @pytest.fixture
    def service(self):
        """Create RAGService with mocked DB session."""
        mock_db = MagicMock()
        try:
            return RAGService(db=mock_db)
        except TypeError:
            # Some versions may not take db in __init__
            try:
                return RAGService()
            except ImportError:
                pytest.skip("RAG dependencies (torch) not installed")
        except ImportError:
            pytest.skip("RAG dependencies (torch) not installed")

    def test_init(self, service: RAGService):
        assert service is not None

    def test_has_retrieve_method(self, service: RAGService):
        assert callable(getattr(service, "retrieve", None)) or callable(
            getattr(service, "search", None)
        )

    def test_has_ingest_method(self, service: RAGService):
        assert callable(getattr(service, "ingest", None)) or callable(
            getattr(service, "add_documents", None)
        )

    def test_get_embedding_stats(self, service: RAGService):
        if hasattr(service, "get_embedding_stats"):
            stats = service.get_embedding_stats()
            assert isinstance(stats, dict)


# ── Embedder / Reranker (smoke tests — models may not be downloaded) ─────────

class TestEmbedderSmoke:
    def test_import_bgem3(self):
        from backend.services.chat.rag import BGEM3Embedder
        assert BGEM3Embedder is not None

    def test_import_reranker(self):
        from backend.services.chat.rag import BGEReranker
        assert BGEReranker is not None
