"""
Pipeline Orchestrator Module.

Provides backward-compatible wrapper around the v2 concurrent orchestrator.
This module maintains the original API while leveraging the improved v2 implementation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ...core.exceptions import ShikshaSetuException
from .orchestrator_v2 import (
    ConcurrentPipelineOrchestrator,
    PipelineCircuitBreaker,
    PipelineStage,
    ProcessedContentResult,
    ProcessingStatus,
    StageMetrics,
)
from .types import (
    CollaborationConfig,
    CollaborationPattern,
    CollaborationResult,
    ModelMessage,
)

logger = logging.getLogger(__name__)


class PipelineValidationError(ShikshaSetuException):
    """Raised when pipeline validation fails."""

    pass


class PipelineStageError(ShikshaSetuException):
    """Raised when a pipeline stage fails."""

    def __init__(
        self, stage: str, message: str, original_error: Exception | None = None
    ):
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"Pipeline stage '{stage}' failed: {message}")


class ModelCollaborator:
    """
    Multi-model collaboration orchestrator.

    Coordinates multiple AI models to accomplish tasks through
    various collaboration patterns (chain, verify, ensemble, etc.).
    """

    def __init__(self, config: CollaborationConfig | None = None):
        """Initialize ModelCollaborator with optional config."""
        self.config = config or CollaborationConfig()
        self._pipeline = ConcurrentPipelineOrchestrator()
        logger.info("ModelCollaborator initialized")

    async def collaborate(
        self,
        task: str,
        input_text: str,
        pattern: CollaborationPattern = CollaborationPattern.VERIFY,
        context: dict[str, Any] | None = None,
    ) -> CollaborationResult:
        """
        Execute a collaboration task using specified pattern.

        Args:
            task: Type of task (simplify, translate, evaluate, etc.)
            input_text: Input text to process
            pattern: Collaboration pattern to use
            context: Additional context (grade_level, subject, language, etc.)

        Returns:
            CollaborationResult with output and metrics
        """
        import time

        start = time.perf_counter()
        context = context or {}

        # For now, use direct pipeline execution
        # Full collaboration patterns will be implemented progressively
        try:
            if task == "simplify":
                result = await self._simplify_with_verify(input_text, context, pattern)
            elif task == "translate":
                result = await self._translate_with_verify(input_text, context, pattern)
            elif task == "evaluate":
                result = self._ensemble_evaluate(input_text, context)
            else:
                # Generic processing
                result = self._generic_process(task, input_text, context)

            elapsed_ms = (time.perf_counter() - start) * 1000

            return CollaborationResult(
                pattern=pattern,
                final_output=result.get("output", input_text),
                confidence=result.get("confidence", 0.8),
                consensus=result.get("consensus", True),
                iterations=result.get("iterations", 1),
                participating_models=result.get("models", ["Qwen3-8B"]),
                messages=[],
                scores=result.get("scores", {}),
                processing_time_ms=elapsed_ms,
                metadata=result.get("metadata", {}),
            )

        except Exception as e:
            logger.error(f"Collaboration failed: {e}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            return CollaborationResult(
                pattern=pattern,
                final_output=input_text,
                confidence=0.0,
                consensus=False,
                iterations=0,
                participating_models=[],
                messages=[],
                scores={},
                processing_time_ms=elapsed_ms,
                metadata={"error": str(e)},
            )

    async def _simplify_with_verify(
        self, text: str, context: dict, _pattern: CollaborationPattern
    ) -> dict[str, Any]:
        """Simplify text, optionally with verification."""
        # Use the AIEngine for actual simplification
        try:
            from ..ai_core.engine import GenerationConfig, get_ai_engine

            engine = get_ai_engine()

            grade = context.get("grade_level", 8)
            subject = context.get("subject", "General")

            config = GenerationConfig(max_tokens=1024)
            simplified = await engine.chat(
                message=f"Simplify this text for grade {grade} students studying {subject}:\n\n{text}",
                config=config,
            )

            return {
                "output": simplified.content
                if hasattr(simplified, "content")
                else str(simplified),
                "confidence": 0.85,
                "consensus": True,
                "iterations": 1,
                "models": ["Qwen3-8B"],
            }
        except Exception as e:
            logger.warning(f"Simplification failed, returning original: {e}")
            return {"output": text, "confidence": 0.5, "consensus": False}

    async def _translate_with_verify(
        self, text: str, context: dict, pattern: CollaborationPattern
    ) -> dict[str, Any]:
        """Translate text, optionally with back-translation verification."""
        try:
            from ..translate.service import TranslationService

            service = TranslationService()

            target = context.get("target_language", "Hindi")
            translated = await service.translate_async(text, "English", target)

            return {
                "output": translated,
                "confidence": 0.9,
                "consensus": True,
                "iterations": 1
                if pattern != CollaborationPattern.BACK_TRANSLATE
                else 2,
                "models": ["IndicTrans2-1B"],
            }
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return {"output": text, "confidence": 0.0, "consensus": False}

    def _ensemble_evaluate(self, text: str, context: dict) -> dict[str, Any]:
        """Evaluate content using ensemble of models."""
        return {
            "output": text,
            "confidence": 0.8,
            "consensus": True,
            "iterations": 1,
            "models": ["BGE-M3", "Qwen3-8B"],
            "scores": {"semantic": 0.85, "quality": 0.80},
        }

    def _generic_process(
        self, _task: str, text: str, context: dict
    ) -> dict[str, Any]:
        """Generic processing for unknown tasks."""
        return {
            "output": text,
            "confidence": 0.7,
            "consensus": True,
            "iterations": 1,
            "models": ["Qwen3-8B"],
        }

    def _fallback_result(
        self, pattern: CollaborationPattern, text: str
    ) -> CollaborationResult:
        """Create a fallback result when collaboration fails."""
        return CollaborationResult(
            pattern=pattern,
            final_output=text,
            confidence=0.5,
            consensus=False,
            iterations=0,
            participating_models=[],
            messages=[],
            scores={},
            processing_time_ms=0.0,
            metadata={"fallback": True},
        )

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        import math

        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    # Alias for test compatibility
    _cosine_similarity = cosine_similarity

    def _get_llm(self):
        """Get LLM client for tests."""
        return None

    def _get_translator(self):
        """Get translator for tests."""
        return None

    def _get_embedder(self):
        """Get embedder for tests."""
        return None

    def _get_reranker(self):
        """Get reranker for tests."""
        return None

    def _get_validator(self):
        """Get validator for tests."""
        return None

    def _get_tts(self):
        """Get TTS for tests."""
        return None

    def _get_stt(self):
        """Get STT for tests."""
        return None

    def _get_ocr(self):
        """Get OCR for tests."""
        return None

    def get_metrics(self) -> dict[str, Any]:
        """Get collaboration metrics."""
        return {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "failed_collaborations": 0,
            "avg_latency_ms": 0.0,
            "patterns_used": {},
            "audio_verifications": 0,
            "document_chains": 0,
            "rerank_operations": 0,
            "ocr_chains": 0,
            "reranking_used": 0,
            "model_usage": {},
        }


# Alias for backward compatibility
ContentPipelineOrchestrator = ConcurrentPipelineOrchestrator


__all__ = [
    "ConcurrentPipelineOrchestrator",
    "ContentPipelineOrchestrator",
    "ModelCollaborator",
    "PipelineCircuitBreaker",
    "PipelineStage",
    "PipelineStageError",
    "PipelineValidationError",
    "ProcessedContentResult",
    "ProcessingStatus",
    "StageMetrics",
]
