"""
Simplification Tasks (Celery)
==============================
Tasks for text simplification using Qwen2.5-3B-Instruct.
"""

import logging
from typing import Any, Dict, Optional

from .celery_config import celery_app

logger = logging.getLogger(__name__)

# Lazy-loaded model
_simplifier = None


def get_simplifier():
    """Get or initialize simplifier (lazy loading)."""
    global _simplifier
    if _simplifier is None:
        from backend.services.simplifier import TextSimplifier

        _simplifier = TextSimplifier()
    return _simplifier


@celery_app.task(
    name="simplify.text",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    soft_time_limit=300,
    time_limit=330,
)
def simplify_text(
    self,
    text: str,
    grade_level: int = 5,
    language: str = "en",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Simplify text for a given grade level.

    Args:
        text: Input text to simplify
        grade_level: Target grade level (1-12)
        language: Language code
        options: Additional options

    Returns:
        Dict with simplified text and metadata
    """
    try:
        import asyncio

        simplifier = get_simplifier()
        options = options or {}

        # Run async simplification
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                simplifier.simplify(text, grade_level, language)
            )
        finally:
            loop.close()

        return {
            "success": True,
            "simplified_text": result.get("simplified_text", ""),
            "grade_level": grade_level,
            "original_length": len(text),
            "simplified_length": len(result.get("simplified_text", "")),
            "metadata": result.get("metadata", {}),
        }

    except Exception as e:
        logger.error(f"Simplification failed: {e}")

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        return {
            "success": False,
            "error": str(e),
            "original_text": text,
        }


@celery_app.task(
    name="simplify.batch",
    bind=True,
    max_retries=1,
    soft_time_limit=600,
    time_limit=660,
)
def simplify_batch(
    self,
    texts: list,
    grade_level: int = 5,
    language: str = "en",
) -> dict[str, Any]:
    """
    Batch simplify multiple texts.

    Args:
        texts: List of texts to simplify
        grade_level: Target grade level
        language: Language code

    Returns:
        Dict with results for each text
    """
    try:
        import asyncio

        simplifier = get_simplifier()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                simplifier.simplify_batch(texts, grade_level, language)
            )
        finally:
            loop.close()

        return {
            "success": True,
            "results": result,
            "count": len(texts),
        }

    except Exception as e:
        logger.error(f"Batch simplification failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@celery_app.task(
    name="simplify.document",
    bind=True,
    max_retries=1,
    soft_time_limit=900,
    time_limit=960,
)
def simplify_document(
    self,
    document_id: str,
    grade_level: int = 5,
    language: str = "en",
) -> dict[str, Any]:
    """
    Simplify an entire document.

    Uses sentence-level processing for efficiency.

    Args:
        document_id: Document ID in storage
        grade_level: Target grade level
        language: Language code

    Returns:
        Dict with simplified document
    """
    try:
        import asyncio

        from backend.services.pipeline import SentenceLevelProcessor

        # Load document
        # document = load_document(document_id)

        # Process with sentence pipeline
        # TODO: Implement document processing with SentenceLevelProcessor
        _ = SentenceLevelProcessor  # Reference to avoid import issues

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # result = loop.run_until_complete(
            #     processor.process_document(document, "simplify", grade_level=grade_level)
            # )
            pass
        finally:
            loop.close()

        return {
            "success": True,
            "document_id": document_id,
            # "simplified_document": result,
        }

    except Exception as e:
        logger.error(f"Document simplification failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": document_id,
        }
