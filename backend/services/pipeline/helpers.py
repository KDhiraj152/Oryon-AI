"""
Collaboration Helpers
=====================

Utility functions for model collaboration:
- Embedding operations
- Similarity calculations
- Logging utilities
- Fallback result generation
"""

import logging
from typing import Any, Dict, List, Optional

from .types import (
    CollaborationPattern,
    CollaborationResult,
    ModelMessage,
)

logger = logging.getLogger(__name__)


class CollaborationHelpersMixin:
    """
    Mixin providing helper methods for collaboration patterns.

    Requires ModelAccessorsMixin to be mixed in as well.
    """

    def __init__(self) -> None:
        """Initialize messages list and metrics."""
        self._messages: list[ModelMessage] = []
        self._metrics: dict[str, int] = {
            "chain_used": 0,
            "verify_used": 0,
            "back_translate_used": 0,
            "ensemble_used": 0,
            "debate_used": 0,
            "iterative_used": 0,
            "semantic_check_used": 0,
            "audio_verify_used": 0,
            "document_chain_used": 0,
            "reranking_used": 0,
        }

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text using BGE-M3."""
        embedder = self._get_embedder()  # type: ignore
        if not embedder:
            return None

        try:
            # Handle different embedder interfaces
            if hasattr(embedder, "embed_async"):
                return await embedder.embed_async(text)
            elif hasattr(embedder, "embed"):
                return embedder.embed(text)
            elif hasattr(embedder, "encode"):
                return embedder.encode(text).tolist()
            else:
                return None
        except Exception as e:
            logger.warning(f"[Collaborator] Embedding failed: {e}")
            return None

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Hardware optimized: Uses SIMD operations when available.
        """
        import numpy as np

        a_arr = np.array(a)
        b_arr = np.array(b)

        try:
            from backend.core.optimized.simd_ops import cosine_similarity_single

            return cosine_similarity_single(a_arr, b_arr)
        except ImportError:
            pass

        dot_product = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _log_message(
        self,
        from_model: str,
        to_model: str,
        content: str,
        context: dict[str, Any],
    ) -> None:
        """Log a message between models."""
        msg = ModelMessage(
            from_model=from_model,
            to_model=to_model,
            content=content[:500],  # Truncate for logging
            context=context,
        )
        self._messages.append(msg)
        logger.debug(f"[{from_model}→{to_model}] {content[:100]}...")

    def _fallback_result(
        self,
        pattern: CollaborationPattern,
        text: str,
    ) -> CollaborationResult:
        """Return a fallback result when collaboration fails."""
        return CollaborationResult(
            pattern=pattern,
            final_output=text,
            confidence=0.5,
            consensus=False,
            iterations=0,
            participating_models=[],
            messages=[],
            scores={},
            processing_time_ms=0,
            metadata={"fallback": True},
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get collaboration metrics."""
        return self._metrics.copy()

    def clear_messages(self) -> None:
        """Clear the message history."""
        self._messages.clear()
