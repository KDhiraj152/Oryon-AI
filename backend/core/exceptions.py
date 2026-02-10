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
        grade_level: int | None = None,
    ):
        super().__init__(
            detail=detail,
            stage="simplification",
            original_error=original_error,
            context={"grade_level": grade_level} if grade_level else {},
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


def with_retry(
    config: RetryConfig = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    exception_wrapper: type[PipelineError] | None = None,
):
    """
    Retry decorator with exponential backoff.

    Example:
        @with_retry(config=RetryConfig(max_attempts=3))
        async def translate_text(text: str) -> str:
            ...
    """
    config = config or DEFAULT_RETRY_CONFIG

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.non_retryable_exceptions as e:
                    logger.warning(f"[Retry] Non-retryable in {func.__name__}: {e}")
                    if exception_wrapper:
                        raise exception_wrapper(detail=str(e), original_error=e) from e
                    raise
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt >= config.max_attempts:
                        break
                    delay = min(
                        config.initial_delay
                        * (config.exponential_base ** (attempt - 1)),
                        config.max_delay,
                    )
                    if config.jitter:
                        delay += delay * config.jitter_factor * random.random()
                    logger.warning(
                        f"[Retry] {func.__name__} attempt {attempt}/{config.max_attempts}. Retrying in {delay:.2f}s..."
                    )
                    if on_retry:
                        on_retry(attempt, e, delay)
                    await asyncio.sleep(delay)
                except Exception as e:
                    if isinstance(e, PipelineError) and not e.retryable:
                        raise
                    last_exception = e
                    if attempt >= config.max_attempts:
                        break
                    await asyncio.sleep(config.initial_delay)

            if exception_wrapper and last_exception:
                raise exception_wrapper(
                    detail=f"Failed after {config.max_attempts} attempts: {last_exception}",
                    original_error=last_exception,
                ) from last_exception
            if last_exception:
                raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.non_retryable_exceptions as e:
                    if exception_wrapper:
                        raise exception_wrapper(detail=str(e), original_error=e) from e
                    raise
                except (config.retryable_exceptions, Exception) as e:
                    last_exception = e
                    if attempt >= config.max_attempts:
                        break
                    delay = config.initial_delay * (
                        config.exponential_base ** (attempt - 1)
                    )
                    time.sleep(delay)
            if last_exception:
                raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0


class CircuitBreaker:
    """Circuit breaker for graceful degradation."""

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        fallback: Callable | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def _should_allow(self) -> bool:
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.config.timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    return True
                return False
            return True

    async def _record_success(self):
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _record_failure(self):
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"[CircuitBreaker:{self.name}] OPEN after {self._failure_count} failures"
                )

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not await self._should_allow():
                if self.fallback:
                    return (
                        await self.fallback(*args, **kwargs)
                        if asyncio.iscoroutinefunction(self.fallback)
                        else self.fallback(*args, **kwargs)
                    )
                raise PipelineError(
                    detail=f"Circuit breaker '{self.name}' is OPEN",
                    stage=self.name,
                    retryable=False,
                )
            try:
                result = await func(*args, **kwargs)
                await self._record_success()
                return result
            except Exception:
                await self._record_failure()
                raise

        return wrapper

    def reset(self):
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
