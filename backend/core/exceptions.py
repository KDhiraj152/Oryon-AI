"""Custom exception classes for ShikshaSetu.

Includes:
- Base exceptions for API errors
- Pipeline-specific exceptions with retry support
- Circuit breaker and retry decorators
"""

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


class ShikshaSetuException(Exception):
    """Base exception for all ShikshaSetu errors."""

    def __init__(
        self, detail: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR"
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        self.timestamp = datetime.now(UTC).isoformat()
        super().__init__(detail)

    def to_dict(self):
        """Convert exception to dictionary for API response."""
        return {
            "error": self.error_code,
            "detail": self.detail,
            "status_code": self.status_code,
            "timestamp": self.timestamp,
        }


class ContentNotFoundError(ShikshaSetuException):
    """Raised when content is not found."""

    def __init__(self, content_id: str):
        super().__init__(
            detail=f"Content with ID {content_id} not found",
            status_code=404,
            error_code="CONTENT_NOT_FOUND",
        )


class DocumentNotFoundError(ShikshaSetuException):
    """Raised when document is not found."""

    def __init__(self, document_id: str):
        super().__init__(
            detail=f"Document with ID {document_id} not found",
            status_code=404,
            error_code="DOCUMENT_NOT_FOUND",
        )


class InvalidFileError(ShikshaSetuException):
    """Raised when uploaded file is invalid."""

    def __init__(self, reason: str):
        super().__init__(
            detail=f"Invalid file: {reason}", status_code=400, error_code="INVALID_FILE"
        )


class TaskNotFoundError(ShikshaSetuException):
    """Raised when task is not found."""

    def __init__(self, task_id: str):
        super().__init__(
            detail=f"Task with ID {task_id} not found",
            status_code=404,
            error_code="TASK_NOT_FOUND",
        )


class AuthenticationError(ShikshaSetuException):
    """Raised when authentication fails."""

    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            detail=detail, status_code=401, error_code="AUTHENTICATION_FAILED"
        )


class AuthorizationError(ShikshaSetuException):
    """Raised when user is not authorized."""

    def __init__(self, detail: str = "Not authorized to access this resource"):
        super().__init__(
            detail=detail, status_code=403, error_code="AUTHORIZATION_FAILED"
        )


class ValidationError(ShikshaSetuException):
    """Raised when input validation fails."""

    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=422, error_code="VALIDATION_ERROR")


class RateLimitError(ShikshaSetuException):
    """Raised when rate limit is exceeded."""

    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            detail=detail, status_code=429, error_code="RATE_LIMIT_EXCEEDED"
        )


class ProcessingError(ShikshaSetuException):
    """Raised when content processing fails."""

    def __init__(self, detail: str):
        super().__init__(
            detail=f"Processing failed: {detail}",
            status_code=500,
            error_code="PROCESSING_ERROR",
        )


class DatabaseError(ShikshaSetuException):
    """Raised when database operation fails."""

    def __init__(self, detail: str):
        super().__init__(
            detail=f"Database error: {detail}",
            status_code=500,
            error_code="DATABASE_ERROR",
        )


# =============================================================================
# PIPELINE EXCEPTIONS (with retry support)
# =============================================================================


class PipelineError(ShikshaSetuException):
    """Base exception for all pipeline errors with retry metadata."""

    def __init__(
        self,
        detail: str,
        stage: str = "unknown",
        original_error: Exception | None = None,
        context: dict[str, Any] | None = None,
        retryable: bool = True,
    ):
        super().__init__(
            detail=detail, status_code=500, error_code=f"PIPELINE_{stage.upper()}_ERROR"
        )
        self.stage = stage
        self.original_error = original_error
        self.context = context or {}
        self.retryable = retryable

    def to_dict(self):
        base = super().to_dict()
        base.update(
            {
                "stage": self.stage,
                "retryable": self.retryable,
                "context": self.context,
            }
        )
        return base


class SimplificationError(PipelineError):
    """Error during text simplification."""

    def __init__(
        self,
        detail: str,
        original_error: Exception | None = None,
        complexity_level: int | None = None,
    ):
        super().__init__(
            detail=detail,
            stage="simplification",
            original_error=original_error,
            context={"complexity_level": complexity_level} if complexity_level else {},
        )


class TranslationError(PipelineError):
    """Error during translation."""

    def __init__(
        self,
        detail: str,
        original_error: Exception | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ):
        super().__init__(
            detail=detail,
            stage="translation",
            original_error=original_error,
            context={"source_lang": source_lang, "target_lang": target_lang},
        )


class AudioGenerationError(PipelineError):
    """Error during TTS audio generation."""

    def __init__(
        self,
        detail: str,
        original_error: Exception | None = None,
        language: str | None = None,
    ):
        super().__init__(
            detail=detail,
            stage="audio_generation",
            original_error=original_error,
            context={"language": language} if language else {},
        )


class TranscriptionError(PipelineError):
    """Error during STT transcription."""

    def __init__(self, detail: str, original_error: Exception | None = None):
        super().__init__(
            detail=detail, stage="transcription", original_error=original_error
        )


class OCRError(PipelineError):
    """Error during OCR text extraction."""

    def __init__(
        self,
        detail: str,
        original_error: Exception | None = None,
        file_path: str | None = None,
    ):
        super().__init__(
            detail=detail,
            stage="ocr",
            original_error=original_error,
            context={"file_path": file_path} if file_path else {},
        )


class EmbeddingError(PipelineError):
    """Error during text embedding."""

    def __init__(self, detail: str, original_error: Exception | None = None):
        super().__init__(
            detail=detail, stage="embedding", original_error=original_error
        )


class ModelLoadError(PipelineError):
    """Error loading a model."""

    def __init__(
        self, detail: str, model_id: str, original_error: Exception | None = None
    ):
        super().__init__(
            detail=detail,
            stage="model_loading",
            original_error=original_error,
            context={"model_id": model_id},
            retryable=False,
        )


class ModelTimeoutError(PipelineError):
    """Model inference timed out."""

    def __init__(
        self,
        detail: str,
        model_id: str,
        timeout_seconds: float,
        original_error: Exception | None = None,
    ):
        super().__init__(
            detail=detail,
            stage="inference",
            original_error=original_error,
            context={"model_id": model_id, "timeout_seconds": timeout_seconds},
            retryable=True,
        )


class CollaborationError(PipelineError):
    """Error during multi-model collaboration."""

    def __init__(
        self,
        detail: str,
        pattern: str,
        participating_models: list[str],
        original_error: Exception | None = None,
    ):
        super().__init__(
            detail=detail,
            stage="collaboration",
            original_error=original_error,
            context={"pattern": pattern, "models": participating_models},
        )


# =============================================================================
# RETRY CONFIGURATION & DECORATOR
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.25
    retryable_exceptions: tuple = field(
        default_factory=lambda: (
            TimeoutError,
            ConnectionError,
            OSError,
            RuntimeError,
        )
    )
    non_retryable_exceptions: tuple = field(
        default_factory=lambda: (
            ValueError,
            TypeError,
            KeyError,
            ModelLoadError,
        )
    )


DEFAULT_RETRY_CONFIG = RetryConfig()
TRANSLATION_RETRY_CONFIG = RetryConfig(
    max_attempts=3, initial_delay=1.0, max_delay=10.0
)
SIMPLIFICATION_RETRY_CONFIG = RetryConfig(
    max_attempts=2, initial_delay=0.5, max_delay=5.0
)
TTS_RETRY_CONFIG = RetryConfig(max_attempts=2, initial_delay=1.0, max_delay=10.0)
EMBEDDING_RETRY_CONFIG = RetryConfig(max_attempts=3, initial_delay=0.5, max_delay=5.0)


def _compute_retry_delay(cfg: RetryConfig, attempt: int) -> float:
    """Compute delay for a retry attempt with optional jitter."""
    delay = min(
        cfg.initial_delay * (cfg.exponential_base ** (attempt - 1)),
        cfg.max_delay,
    )
    if cfg.jitter:
        delay += delay * cfg.jitter_factor * random.random()
    return delay


def _raise_if_wrapped(
    last_exc: Exception | None,
    wrapper: type[PipelineError] | None,
    max_attempts: int,
) -> None:
    """Raise the final exception, optionally wrapped."""
    if wrapper and last_exc:
        raise wrapper(
            detail=f"Failed after {max_attempts} attempts: {last_exc}",
            original_error=last_exc,
        ) from last_exc
    if last_exc:
        raise last_exc


def _handle_non_retryable(
    exc: Exception,
    func_name: str,
    wrapper: type[PipelineError] | None,
) -> None:
    """Handle a non-retryable exception by wrapping and re-raising."""
    logger.warning(f"[Retry] Non-retryable in {func_name}: {exc}")
    if wrapper:
        raise wrapper(detail=str(exc), original_error=exc) from exc
    raise


def with_retry(
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    exception_wrapper: type[PipelineError] | None = None,
) -> Callable[[F], F]:
    """
    Retry decorator with exponential backoff.

    Example:
        @with_retry(config=RetryConfig(max_attempts=3))
        async def translate_text(text: str) -> str:
            ...
    """
    cfg = config or DEFAULT_RETRY_CONFIG

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await _run_async_with_retry(func, args, kwargs, cfg, on_retry, exception_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_sync_with_retry(func, args, kwargs, cfg, exception_wrapper)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


async def _handle_retryable_async(
    exc: Exception,
    func_name: str,
    attempt: int,
    cfg: RetryConfig,
    on_retry: Callable[[int, Exception, float], None] | None,
) -> None:
    """Handle a retryable exception: log, notify callback, and sleep."""
    delay = _compute_retry_delay(cfg, attempt)
    logger.warning(
        f"[Retry] {func_name} attempt {attempt}/{cfg.max_attempts}. "
        f"Retrying in {delay:.2f}s..."
    )
    if on_retry:
        on_retry(attempt, exc, delay)
    await asyncio.sleep(delay)


async def _run_async_with_retry(
    func: Callable,
    args: tuple,
    kwargs: dict,
    cfg: RetryConfig,
    on_retry: Callable[[int, Exception, float], None] | None,
    wrapper: type[PipelineError] | None,
) -> Any:
    """Run an async function with retry logic."""
    last_exception: Exception | None = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except cfg.non_retryable_exceptions as exc:
            _handle_non_retryable(exc, func.__name__, wrapper)
        except cfg.retryable_exceptions as exc:
            last_exception = exc
            if attempt < cfg.max_attempts:
                await _handle_retryable_async(exc, func.__name__, attempt, cfg, on_retry)
        except Exception as exc:
            _check_non_retryable_pipeline_error(exc)
            last_exception = exc
            if attempt < cfg.max_attempts:
                await asyncio.sleep(cfg.initial_delay)
    _raise_if_wrapped(last_exception, wrapper, cfg.max_attempts)


def _check_non_retryable_pipeline_error(exc: Exception) -> None:
    """Re-raise if exception is a non-retryable PipelineError."""
    if isinstance(exc, PipelineError) and not exc.retryable:
        raise exc


def _run_sync_with_retry(
    func: Callable,
    args: tuple,
    kwargs: dict,
    cfg: RetryConfig,
    wrapper: type[PipelineError] | None,
) -> Any:
    """Run a sync function with retry logic."""
    last_exception: Exception | None = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except cfg.non_retryable_exceptions as exc:
            _handle_non_retryable(exc, func.__name__, wrapper)
        except Exception as exc:
            last_exception = exc
            if attempt >= cfg.max_attempts:
                break
            delay = cfg.initial_delay * (cfg.exponential_base ** (attempt - 1))
            time.sleep(delay)
    _raise_if_wrapped(last_exception, wrapper, cfg.max_attempts)


# =============================================================================
# CIRCUIT BREAKER — Canonical implementation in core/circuit_breaker.py
# Re-exported here for backward compatibility.
# =============================================================================
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
