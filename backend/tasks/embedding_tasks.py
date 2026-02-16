"""
Embedding Tasks (Celery)
=========================
Tasks for text embedding using BGE-M3.
"""

import logging
from typing import Any

from .celery_config import celery_app

logger = logging.getLogger(__name__)

# Lazy-loaded model
_embedder = None

def get_embedder():
    """Get shared embedder singleton (lazy loading)."""
    global _embedder
    if _embedder is None:
        from backend.services.chat.rag import get_embedder as _get_singleton

        _embedder = _get_singleton()  # Use singleton instead of creating new instance
    return _embedder

@celery_app.task(
    name="embedding.embed_text",
    bind=True,
    max_retries=2,
    default_retry_delay=3,
    soft_time_limit=60,
    time_limit=90,
)
def embed_text(
    self, text: str, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Generate embedding for text.

    Args:
        text: Input text
        options: Additional options

    Returns:
        Dict with embedding vector
    """
    try:
        import asyncio

        embedder = get_embedder()
        options = options or {}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            embedding = loop.run_until_complete(embedder.embed(text))
        finally:
            loop.close()

        return {
            "success": True,
            "embedding": embedding.tolist()
            if hasattr(embedding, "tolist")
            else embedding,
            "dimensions": len(embedding),
            "text_length": len(text),
        }

    except Exception as e:  # task handler — intentionally broad
        logger.error("Embedding failed: %s", e)

        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        return {
            "success": False,
            "error": str(e),
        }

@celery_app.task(
    name="embedding.embed_batch",
    bind=True,
    max_retries=1,
    soft_time_limit=300,
    time_limit=330,
)
def embed_batch(
    self, texts: list[str], options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Batch generate embeddings for multiple texts.

    More efficient than individual embeddings.

    Args:
        texts: List of texts
        options: Additional options

    Returns:
        Dict with embeddings
    """
    try:
        import asyncio

        embedder = get_embedder()
        options = options or {}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            embeddings = loop.run_until_complete(embedder.embed_batch(texts))
        finally:
            loop.close()

        # Convert to lists if numpy arrays
        embeddings_list = [
            e.tolist() if hasattr(e, "tolist") else e for e in embeddings
        ]

        return {
            "success": True,
            "embeddings": embeddings_list,
            "count": len(texts),
            "dimensions": len(embeddings_list[0]) if embeddings_list else 0,
        }

    except Exception as e:  # task handler — intentionally broad
        logger.error("Batch embedding failed: %s", e)
        return {
            "success": False,
            "error": str(e),
        }

@celery_app.task(
    name="embedding.similarity",
    bind=True,
    soft_time_limit=30,
    time_limit=45,
)
def compute_similarity(
    self,
    text1: str,
    text2: str,
) -> dict[str, Any]:
    """
    Compute similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Dict with similarity score
    """
    try:
        import asyncio

        import numpy as np

        embedder = get_embedder()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            embeddings = loop.run_until_complete(embedder.embed_batch([text1, text2]))
        finally:
            loop.close()

        # SIMD-optimized cosine similarity
        e1, e2 = embeddings[0], embeddings[1]
        try:
            from backend.core.optimized.simd_ops import cosine_similarity_single

            similarity = cosine_similarity_single(e1, e2)
        except ImportError:
            similarity = float(
                np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            )

        return {
            "success": True,
            "similarity": similarity,
            "text1_length": len(text1),
            "text2_length": len(text2),
        }

    except Exception as e:  # task handler — intentionally broad
        logger.error("Similarity computation failed: %s", e)
        return {
            "success": False,
            "error": str(e),
        }

@celery_app.task(
    name="embedding.index_documents",
    bind=True,
    soft_time_limit=600,
    time_limit=660,
)
def index_documents(
    self,
    documents: list[dict[str, Any]],
    index_name: str,
) -> dict[str, Any]:
    """
    Index documents for vector search.

    Args:
        documents: List of documents with 'id', 'text', 'metadata'
        index_name: Name of the index

    Returns:
        Dict with indexing results
    """
    try:
        import asyncio

        embedder = get_embedder()

        # Extract texts
        texts = [doc["text"] for doc in documents]
        # ids are used when storing to vector DB (future implementation)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            embeddings = loop.run_until_complete(embedder.embed_batch(texts))
        finally:
            loop.close()

        # Store in vector database
        # This would connect to pgvector, Qdrant, etc.
        # For now, return success

        return {
            "success": True,
            "indexed_count": len(documents),
            "index_name": index_name,
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }

    except Exception as e:  # task handler — intentionally broad
        logger.error("Document indexing failed: %s", e)
        return {
            "success": False,
            "error": str(e),
        }

@celery_app.task(
    name="embedding.search",
    bind=True,
    soft_time_limit=30,
    time_limit=45,
)
def search(
    self,
    query: str,
    index_name: str,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Search for similar documents.

    Args:
        query: Search query
        index_name: Name of the index
        top_k: Number of results

    Returns:
        Dict with search results
    """
    try:
        import asyncio

        embedder = get_embedder()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            _query_embedding = loop.run_until_complete(embedder.embed(query))
        finally:
            loop.close()

        # Search in vector database (would use _query_embedding)
        # This would query pgvector, Qdrant, etc.
        # For now, return placeholder

        return {
            "success": True,
            "query": query,
            "index_name": index_name,
            "results": [],  # Would contain matched documents
            "top_k": top_k,
        }

    except Exception as e:  # task handler — intentionally broad
        logger.error("Search failed: %s", e)
        return {
            "success": False,
            "error": str(e),
        }
