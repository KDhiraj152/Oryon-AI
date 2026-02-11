"""
Production-Grade Concurrent Pipeline Orchestrator v2.

Replaces sequential execution with concurrent asyncio patterns,
proper backpressure handling, and failure recovery.

Key improvements:
- asyncio.gather() for concurrent stage execution
- Semaphore-based backpressure (prevents memory spike)
- Automatic stage priority with timeout escalation
- Circuit breaker pattern for cascading failures
- Comprehensive observability with spans
- Proper async database session management
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from sqlalchemy import select

from ...cache import get_redis
from ...core.exceptions import ShikshaSetuException
from ...database import get_async_db_session
from ...models import ProcessedContent
from ..error_tracking import PerformanceMonitor, add_breadcrumb, capture_exception
from .model_clients_async import (
    BERTAsyncClient,
    IndicTrans2AsyncClient,
    MMSTTSAsyncClient,
    QwenAsyncClient,
)


# Lazy import to avoid circular dependency with curriculum_validation
def _get_validate_in_pipeline():
    from ..curriculum_validation import validate_in_pipeline

    return validate_in_pipeline


logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages."""

    SIMPLIFICATION = "simplification"
    TRANSLATION = "translation"
    VALIDATION = "validation"
    SPEECH = "speech"


class ProcessingStatus(Enum):
    """Processing status values."""

    SUCCESS = "success"
    FAILED = "failed"
    DEGRADED = "degraded"
    TIMEOUT = "timeout"


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    stage: str
    processing_time_ms: int
    success: bool
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    error_message: str | None = None
    retry_count: int = 0
    fallback_used: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ProcessedContentResult:
    """Result of content processing through the pipeline."""

    id: str
    original_text: str
    simplified_text: str
    translated_text: str
    language: str
    grade_level: int
    subject: str
    audio_file_path: str | None = None
    ncert_alignment_score: float = 0.0
    audio_accuracy_score: float | None = None
    validation_status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: list[StageMetrics] = field(default_factory=list)


class PipelineCircuitBreaker:
    """Synchronous circuit breaker for pipeline cascading failure prevention.

    Simplified version of core.circuit_breaker.CircuitBreaker for
    non-async pipeline stages.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._is_open = False

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._is_open = True
            logger.warning(
                f"Pipeline circuit breaker OPEN after {self._failure_count} failures"
            )

    def record_success(self) -> None:
        """Record a success and recover toward closed state."""
        if self._failure_count > 0:
            self._failure_count -= 1
        if self._failure_count == 0 and self._is_open:
            self._is_open = False
            logger.info("Pipeline circuit breaker CLOSED — recovery successful")

    def check(self) -> bool:
        """Return True if requests should be allowed."""
        if not self._is_open:
            return True
        if time.time() - self._last_failure_time > self.recovery_timeout:
            logger.info("Pipeline circuit breaker entering HALF-OPEN state")
            return True
        return False


class ConcurrentPipelineOrchestrator:
    """
    Production-grade concurrent pipeline orchestrator.

    Executes pipeline stages concurrently with proper:
    - Backpressure handling (semaphores)
    - Timeout management per stage
    - Circuit breaker for cascading failures
    - Cache-aware execution
    - Comprehensive error recovery

    NOTE: In UNIVERSAL_MODE, subject restrictions are bypassed.
    """

    # Configuration
    SUPPORTED_LANGUAGES = [
        "Hindi",
        "Tamil",
        "Telugu",
        "Bengali",
        "Marathi",
        "Gujarati",
        "Kannada",
    ]
    # NOTE: Subject list kept for backward compatibility but not enforced in UNIVERSAL_MODE
    SUPPORTED_SUBJECTS = [
        "Mathematics",
        "Science",
        "Social Studies",
        "English",
        "History",
        "Geography",
        "General",
    ]
    SUPPORTED_FORMATS = ["text", "audio", "both"]
    MIN_GRADE = 1  # Extended range for universal access
    MAX_GRADE = 16  # Includes undergraduate levels

    # Stage-specific timeouts (seconds)
    STAGE_TIMEOUTS = {
        PipelineStage.SIMPLIFICATION: 30,
        PipelineStage.TRANSLATION: 25,
        PipelineStage.VALIDATION: 20,
        PipelineStage.SPEECH: 40,
    }

    # Backpressure limits (concurrent tasks per stage)
    STAGE_CONCURRENCY = {
        PipelineStage.SIMPLIFICATION: 5,
        PipelineStage.TRANSLATION: 5,
        PipelineStage.VALIDATION: 10,
        PipelineStage.SPEECH: 3,  # Most resource-intensive
    }

    def __init__(self, api_key: str | None = None):
        """Initialize orchestrator with async model clients."""
        self.api_key = api_key
        self.qwen_client = QwenAsyncClient(api_key or "")
        self.indictrans2_client = IndicTrans2AsyncClient(api_key or "")
        self.bert_client = BERTAsyncClient(api_key or "")
        self.tts_client = MMSTTSAsyncClient(api_key or "")

        # Initialize circuit breakers per stage
        self.circuit_breakers = {
            stage: PipelineCircuitBreaker() for stage in PipelineStage
        }

        # Semaphores for backpressure control
        self.semaphores = {
            stage: asyncio.Semaphore(self.STAGE_CONCURRENCY[stage])
            for stage in PipelineStage
        }

        # Metrics accumulator
        self.metrics: list[StageMetrics] = []

        # Performance monitor (lazy initialization)
        self._perf_monitor = None

        logger.info("ConcurrentPipelineOrchestrator v2 initialized")

    @property
    def perf_monitor(self):
        """Lazy initialization of performance monitor."""
        if self._perf_monitor is None:
            self._perf_monitor = PerformanceMonitor("pipeline_orchestrator", "pipeline")
        return self._perf_monitor

    async def process_content(
        self,
        input_data: str | bytes,
        target_language: str,
        grade_level: int,
        subject: str,
        output_format: str = "both",
        user_id: str | None = None,
    ) -> ProcessedContentResult:
        """
        Process content through concurrent pipeline.

        This is the main entry point. It orchestrates all stages concurrently
        while respecting dependencies and handling failures gracefully.

        Args:
            input_data: Raw text or document content
            target_language: Target Indian language
            grade_level: Grade level (5-12)
            subject: Subject area
            output_format: Output format ('text', 'audio', 'both')
            user_id: Optional user ID for tracking

        Returns:
            ProcessedContentResult with all outputs and metrics

        Raises:
            ShikshaSetuException: If validation fails
        """
        # Validate inputs
        self._validate_parameters(
            input_data, target_language, grade_level, subject, output_format
        )

        # Reset metrics for this run
        self.metrics = []

        # Normalize input
        if isinstance(input_data, bytes):
            input_data = input_data.decode("utf-8", errors="replace")

        original_text = input_data
        cache_key = self._compute_cache_key(original_text, target_language, grade_level)

        # Check cache first (async - offloaded to executor to avoid blocking event loop)
        cached_result = await self._check_cache_async(cache_key)
        if cached_result:
            logger.info(f"Cache hit for {cache_key}")
            add_breadcrumb("Cache hit", category="pipeline", level="debug")
            return cached_result

        logger.info(
            f"Pipeline start: lang={target_language}, grade={grade_level}, "
            f"subject={subject}, format={output_format}"
        )

        try:
            # Execute stages with proper dependency management
            with self.perf_monitor.span("pipeline_execution"):
                # Stage 1: Simplification (can run immediately)
                simplify_task = asyncio.create_task(
                    self._execute_stage_with_backpressure(
                        PipelineStage.SIMPLIFICATION,
                        self._simplify_text,
                        original_text,
                        grade_level,
                        subject,
                    )
                )

                # Wait for simplification (required for translation)
                simplified_text, simplify_metrics = await simplify_task
                self.metrics.append(simplify_metrics)

                if not simplified_text:
                    raise ShikshaSetuException(
                        "Simplification failed - empty result", status_code=500
                    )

                # Stages 2-4: Translation, Validation, Speech (concurrent)
                # These can run in parallel since Translation doesn't depend on Validation
                translate_task = asyncio.create_task(
                    self._execute_stage_with_backpressure(
                        PipelineStage.TRANSLATION,
                        self._translate_text,
                        simplified_text,
                        target_language,
                    )
                )

                validation_task = asyncio.create_task(
                    self._execute_stage_with_backpressure(
                        PipelineStage.VALIDATION,
                        self._validate_content,
                        original_text,
                        simplified_text,
                        grade_level,
                        subject,
                    )
                )

                # Speech depends on translation, so wait for translation first
                translated_text, translate_metrics = await translate_task
                self.metrics.append(translate_metrics)

                if not translated_text:
                    raise ShikshaSetuException(
                        "Translation failed - empty result", status_code=500
                    )

                # Validation can complete independently
                ncert_score, validation_metrics = await validation_task
                self.metrics.append(validation_metrics)

                # Speech generation (if requested)
                audio_file_path = None
                audio_accuracy_score = None

                if output_format in ["audio", "both"]:
                    speech_task = asyncio.create_task(
                        self._execute_stage_with_backpressure(
                            PipelineStage.SPEECH,
                            self._generate_speech,
                            translated_text,
                            target_language,
                            subject,
                        )
                    )

                    try:
                        speech_result, speech_metrics = await speech_task
                        self.metrics.append(speech_metrics)

                        if isinstance(speech_result, tuple):
                            audio_file_path, audio_accuracy_score = speech_result
                    except TimeoutError:
                        logger.warning(
                            "Speech generation timeout - continuing without audio"
                        )
                        self.metrics.append(
                            StageMetrics(
                                stage=PipelineStage.SPEECH.value,
                                processing_time_ms=0,
                                success=False,
                                status=ProcessingStatus.TIMEOUT,
                                error_message="Speech generation timeout",
                            )
                        )

                # Store processed content in database
                content_id = await self._store_content_async(
                    original_text=original_text,
                    simplified_text=simplified_text,
                    translated_text=translated_text,
                    language=target_language,
                    grade_level=grade_level,
                    subject=subject,
                    audio_file_path=audio_file_path,
                    ncert_alignment_score=ncert_score,
                    audio_accuracy_score=audio_accuracy_score,
                    user_id=user_id,
                )

                # Run comprehensive curriculum validation using async session
                async with get_async_db_session() as db:
                    db_result = await db.execute(
                        select(ProcessedContent).where(
                            ProcessedContent.id == content_id
                        )
                    )
                    content = db_result.scalar_one_or_none()

                    if content:
                        validate_in_pipeline = _get_validate_in_pipeline()
                        (
                            validation_passed,
                            validation_summary,
                        ) = await validate_in_pipeline(
                            db=db, content=content, text=simplified_text
                        )
                    else:
                        validation_passed, validation_summary = True, {}

                # Create result
                result = ProcessedContentResult(
                    id=str(content_id),
                    original_text=original_text,
                    simplified_text=simplified_text,
                    translated_text=translated_text,
                    language=target_language,
                    grade_level=grade_level,
                    subject=subject,
                    audio_file_path=audio_file_path,
                    ncert_alignment_score=ncert_score,
                    audio_accuracy_score=audio_accuracy_score,
                    validation_status="passed" if validation_passed else "warning",
                    metadata=validation_summary,
                    metrics=self.metrics,
                )

                # Cache result (non-blocking via executor)
                await self._cache_result_async(cache_key, result)

                logger.info(f"Pipeline completed successfully: {content_id}")
                final_result: ProcessedContentResult = result
                return final_result

        except TimeoutError as e:
            logger.error(f"Pipeline timeout: {e}")
            add_breadcrumb(f"Pipeline timeout: {e}", category="error", level="error")
            raise ShikshaSetuException("Pipeline processing timeout", status_code=504)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            add_breadcrumb(f"Pipeline error: {e}", category="error", level="error")
            capture_exception(e)
            raise ShikshaSetuException(
                f"Pipeline processing failed: {e!s}", status_code=500
            )

    async def _execute_stage_with_backpressure(
        self, stage: PipelineStage, handler, *args, **kwargs
    ) -> tuple[Any, StageMetrics]:
        """
        Execute pipeline stage with backpressure control and timeout.

        Uses semaphore to limit concurrent executions and timeout to
        prevent hanging requests.
        """
        # Check circuit breaker
        if not self.circuit_breakers[stage].check():
            raise ShikshaSetuException(
                f"Pipeline stage {stage.value} is unavailable (circuit breaker open)",
                status_code=503,
            )

        start_time = time.time()

        async with self.semaphores[stage]:
            try:
                # Execute with stage-specific timeout
                timeout = self.STAGE_TIMEOUTS[stage]
                result = await asyncio.wait_for(
                    handler(*args, **kwargs), timeout=timeout
                )

                # Record success
                elapsed_ms = int((time.time() - start_time) * 1000)
                metrics = StageMetrics(
                    stage=stage.value,
                    processing_time_ms=elapsed_ms,
                    success=True,
                    status=ProcessingStatus.SUCCESS,
                )

                self.circuit_breakers[stage].record_success()
                logger.debug(f"Stage {stage.value} completed in {elapsed_ms}ms")

                return result, metrics

            except TimeoutError:
                elapsed_ms = int((time.time() - start_time) * 1000)
                metrics = StageMetrics(
                    stage=stage.value,
                    processing_time_ms=elapsed_ms,
                    success=False,
                    status=ProcessingStatus.TIMEOUT,
                    error_message=f"Stage timeout after {self.STAGE_TIMEOUTS[stage]}s",
                )

                self.circuit_breakers[stage].record_failure()
                logger.error(f"Stage {stage.value} timeout")

                raise ShikshaSetuException(
                    f"Stage {stage.value} timeout", status_code=504
                )

            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                metrics = StageMetrics(
                    stage=stage.value,
                    processing_time_ms=elapsed_ms,
                    success=False,
                    status=ProcessingStatus.FAILED,
                    error_message=str(e),
                )

                self.circuit_breakers[stage].record_failure()
                logger.error(f"Stage {stage.value} failed: {e}")

                raise

    async def _simplify_text(self, text: str, grade_level: int, subject: str) -> str:
        """Simplify text for target grade level."""
        return await self.qwen_client.process(
            text=text, grade_level=grade_level, subject=subject
        )

    async def _translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language."""
        return await self.indictrans2_client.process(
            text=text, target_language=target_language
        )

    async def _validate_content(
        self, original_text: str, simplified_text: str, grade_level: int, subject: str
    ) -> float:
        """Validate content for curriculum alignment."""
        result = await self.bert_client.process(
            text=simplified_text, grade_level=grade_level, subject=subject
        )
        return float(result) if not isinstance(result, float) else result  # type: ignore[arg-type]

    async def _generate_speech(
        self,
        text: str,
        language: str,
        subject: str
        | None = None,  # Reserved for future subject-specific voice selection
    ) -> tuple[str, float]:
        """Generate speech for translated content.

        Args:
            text: Text to convert to speech
            language: Target language for TTS
            subject: Reserved for future subject-specific voice customization
        """
        # Subject parameter reserved for future voice style customization
        _ = subject
        return await self.tts_client.process(text=text, language=language)  # type: ignore[return-value]

    async def _store_content_async(
        self,
        original_text: str,
        simplified_text: str,
        translated_text: str,
        language: str,
        grade_level: int,
        subject: str,
        audio_file_path: str | None,
        ncert_alignment_score: float,
        audio_accuracy_score: float | None,
        user_id: str | None = None,
    ) -> str:
        """Store processed content in database using async session."""
        async with get_async_db_session() as db:
            content = ProcessedContent(
                original_text=original_text,
                simplified_text=simplified_text,
                translated_text=translated_text,
                language=language,
                grade_level=grade_level,
                subject=subject,
                audio_file_path=audio_file_path,
                ncert_alignment_score=ncert_alignment_score,
                audio_accuracy_score=audio_accuracy_score,
                user_id=user_id,
            )

            db.add(content)
            await db.flush()
            content_id = content.id
            await db.commit()

            return str(content_id)

    def _compute_cache_key(self, text: str, language: str, grade_level: int) -> str:
        """Compute cache key for processed content."""
        import hashlib

        key_str = f"{text}|{language}|{grade_level}"
        return f"pipeline:{hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()}"

    async def _check_cache_async(self, cache_key: str) -> ProcessedContentResult | None:
        """Check if result exists in cache (non-blocking, runs in executor)."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._check_cache, cache_key)

    def _check_cache(self, cache_key: str) -> ProcessedContentResult | None:
        """Check if result exists in cache (sync helper for executor)."""
        redis = get_redis()
        if not redis:
            return None

        try:
            cached_json = redis.get(cache_key)
            if cached_json:
                import json

                data = json.loads(cached_json)
                # Reconstruct result from JSON
                return ProcessedContentResult(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

        return None

    async def _cache_result_async(self, cache_key: str, result: ProcessedContentResult) -> None:
        """Cache processing result (non-blocking, runs in executor)."""
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._cache_result, cache_key, result)

    def _cache_result(self, cache_key: str, result: ProcessedContentResult) -> None:
        """Cache processing result (sync helper for executor)."""
        redis = get_redis()
        if not redis:
            return

        try:
            import json

            result_json = json.dumps(result.__dict__, default=str)
            # 24-hour TTL for cache
            redis.setex(cache_key, 86400, result_json)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def _validate_parameters(
        self,
        input_data: str | bytes,
        target_language: str,
        grade_level: int,
        subject: str,  # Reserved for future subject-specific validation
        output_format: str,
    ) -> None:
        """Validate pipeline parameters.

        Args:
            input_data: Raw text or document content
            target_language: Target Indian language
            grade_level: Grade level for content adaptation
            subject: Subject area (reserved for future validation)
            output_format: Output format ('text', 'audio', 'both')
        """
        # Subject parameter reserved for future subject-specific validation
        _ = subject

        if not input_data:
            raise ShikshaSetuException("Input data is empty", status_code=400)

        if target_language not in self.SUPPORTED_LANGUAGES:
            raise ShikshaSetuException(
                f"Unsupported language: {target_language}", status_code=400
            )

        if not (self.MIN_GRADE <= grade_level <= self.MAX_GRADE):
            raise ShikshaSetuException(
                f"Grade level must be between {self.MIN_GRADE} and {self.MAX_GRADE}",
                status_code=400,
            )

        # NOTE: Subject validation skipped in UNIVERSAL_MODE
        # Any subject/topic is allowed - the system handles all knowledge domains

        if output_format not in self.SUPPORTED_FORMATS:
            raise ShikshaSetuException(
                f"Unsupported output format: {output_format}", status_code=400
            )

    async def close(self):
        """Close all HTTP clients and cleanup resources."""
        clients = [
            self.qwen_client,
            self.indictrans2_client,
            self.bert_client,
            self.tts_client,
        ]
        for client in clients:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
        logger.info("ConcurrentPipelineOrchestrator closed all clients")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures cleanup."""
        await self.close()
        return False
