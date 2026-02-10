"""
RAG Tasks (Celery)
==================
Tasks for RAG (Retrieval-Augmented Generation) operations.
"""

import logging
from typing import Any, Dict, List, Optional

from .celery_config import celery_app

logger = logging.getLogger(__name__)

# Lazy-loaded services
_rag_engine = None
_reranker = None


def get_rag_engine():
    """Get or initialize RAG engine (lazy loading)."""
    global _rag_engine
    if _rag_engine is None:
        from backend.services.rag import RAGEngine

        _rag_engine = RAGEngine()
    return _rag_engine


def get_reranker():
    """Get shared reranker singleton (lazy loading)."""
    global _reranker
    if _reranker is None:
        from backend.services.rag import get_reranker as _get_singleton

        _reranker = _get_singleton()  # Use singleton instead of creating new instance
    return _reranker


@celery_app.task(
    name="rag.query",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    soft_time_limit=60,
    time_limit=90,
)
def query(
    self,
    question: str,
    context_sources: list[str] | None = None,
    top_k: int = 5,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Answer a question using RAG.

    Args:
        question: User question
        context_sources: Optional list of source document IDs to search
        top_k: Number of context chunks to retrieve
        options: Additional options

    Returns:
        Dict with answer and sources
    """
    try:
        import asyncio

        rag = get_rag_engine()
        options = options or {}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                rag.query(
                    question=question, top_k=top_k, sources=context_sources, **options
                )
            )
        finally:
            loop.close()

        return {
            "success": True,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "metadata": result.get("metadata", {}),
        }

    except Exception as e:
        logger.error(f"RAG query failed: {e}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        return {
            "success": False,
            "error": str(e),
            "question": question,
        }


@celery_app.task(
    name="rag.retrieve",
    bind=True,
    soft_time_limit=30,
    time_limit=45,
)
def retrieve(
    self,
    query: str,
    top_k: int = 10,
    use_reranker: bool = True,
) -> dict[str, Any]:
    """
    Retrieve relevant documents without generation.

    Args:
        query: Search query
        top_k: Number of results
        use_reranker: Whether to rerank results

    Returns:
        Dict with retrieved documents
    """
    try:
        import asyncio

        rag = get_rag_engine()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Retrieve
            results = loop.run_until_complete(
                rag.retrieve(query, top_k=top_k * 2 if use_reranker else top_k)
            )

            # Rerank if requested
            if use_reranker and results:
                reranker = get_reranker()
                results = loop.run_until_complete(
                    reranker.rerank(query, results, top_k=top_k)
                )
        finally:
            loop.close()

        return {
            "success": True,
            "results": results,
            "count": len(results),
            "reranked": use_reranker,
        }

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@celery_app.task(
    name="rag.rerank",
    bind=True,
    soft_time_limit=30,
    time_limit=45,
)
def rerank(
    self,
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Rerank documents by relevance to query.

    Args:
        query: Search query
        documents: List of documents to rerank
        top_k: Number of top results to return

    Returns:
        Dict with reranked documents
    """
    try:
        import asyncio

        reranker = get_reranker()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(
                reranker.rerank(query, documents, top_k=top_k)
            )
        finally:
            loop.close()

        return {
            "success": True,
            "results": results,
            "original_count": len(documents),
            "returned_count": len(results),
        }

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@celery_app.task(
    name="rag.ingest_document",
    bind=True,
    soft_time_limit=300,
    time_limit=330,
)
def ingest_document(
    self,
    document_id: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> dict[str, Any]:
    """
    Ingest document into RAG system.

    Chunks document and creates embeddings.

    Args:
        document_id: Document identifier
        content: Document content
        metadata: Document metadata
        chunk_size: Size of chunks in tokens
        chunk_overlap: Overlap between chunks

    Returns:
        Dict with ingestion results
    """
    try:
        import asyncio

        rag = get_rag_engine()
        metadata = metadata or {}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                rag.ingest(
                    document_id=document_id,
                    content=content,
                    metadata=metadata,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
        finally:
            loop.close()

        return {
            "success": True,
            "document_id": document_id,
            "chunks_created": result.get("chunks_created", 0),
            "embeddings_created": result.get("embeddings_created", 0),
        }

    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": document_id,
        }


@celery_app.task(
    name="rag.delete_document",
    bind=True,
    soft_time_limit=30,
    time_limit=45,
)
def delete_document(
    self,
    document_id: str,
) -> dict[str, Any]:
    """
    Delete document from RAG system.

    Args:
        document_id: Document identifier

    Returns:
        Dict with deletion result
    """
    try:
        import asyncio

        rag = get_rag_engine()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(rag.delete(document_id))
        finally:
            loop.close()

        return {
            "success": True,
            "document_id": document_id,
            "chunks_deleted": result.get("chunks_deleted", 0),
        }

    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": document_id,
        }


@celery_app.task(
    name="rag.curriculum_qa",
    bind=True,
    soft_time_limit=90,
    time_limit=120,
)
def curriculum_qa(
    self,
    question: str,
    grade: int,
    subject: str,
    language: str = "en",
) -> dict[str, Any]:
    """
    Answer curriculum-related questions.

    Specialized RAG for NCERT/educational content.

    Args:
        question: Student question
        grade: Grade level
        subject: Subject (math, science, etc.)
        language: Response language

    Returns:
        Dict with educational answer
    """
    try:
        import asyncio

        rag = get_rag_engine()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Build curriculum-aware query
            enhanced_query = f"Grade {grade} {subject}: {question}"

            result = loop.run_until_complete(
                rag.query(
                    question=enhanced_query,
                    top_k=5,
                    filter_metadata={
                        "grade": grade,
                        "subject": subject,
                    },
                )
            )
        finally:
            loop.close()

        return {
            "success": True,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "grade": grade,
            "subject": subject,
            "language": language,
        }

    except Exception as e:
        logger.error(f"Curriculum QA failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "question": question,
        }
