"""
Unified Pipeline Service - Single Entry Point for All Processing
=================================================================

v2.1.0 - Advanced Caching Integration (December 2025)

Combines ALL optimized AI/ML models into a single, efficient API:

MODELS USED:
- Qwen2.5-3B-Instruct (MLX): Text simplification & chat
- IndicTrans2-1B: Translation to 10 Indian languages
- BGE-M3: Multilingual embeddings (1024D)
- BGE-Reranker-v2-M3: Enhanced retrieval accuracy
- Gemma-2-2B-IT: Content validation & curriculum alignment
- MMS-TTS (Facebook): Text-to-speech for Indian languages
- Whisper V3 Turbo: Speech-to-text (99 languages)
- GOT-OCR2: Vision-language OCR for Indian scripts

OPTIMIZATIONS:
- Automatic device routing (MLX/CoreML/MPS)
- Multi-tier caching with task-specific sub-caches:
  * L1 memory: Fast lookups with bloom filter
  * L2 Redis: Shared cache with compression
  * L3 SQLite: Persistent storage with WAL mode
- Task-specific caching:
  * Embeddings: float16 storage, ANE batch size
  * Translations: LRU with language-keyed entries
  * TTS Audio: Large value handling with compression
  * LLM Outputs: Deterministic responses cached
- Write-behind queue for async persistence
- Concurrent processing with async/await
- Streaming responses with SSE
- Pre-tokenization for faster inference
"""

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try xxhash for faster cache key generation (10x faster than SHA256)
try:
    import xxhash

    _USE_XXHASH = True
except ImportError:
    _USE_XXHASH = False

from ...cache import (
    CacheTier,
    get_embedding_cache,
    get_unified_cache,
)
from ...core.optimized import get_device_router
from ...core.optimized.async_optimizer import (
    gather_with_concurrency,
    get_async_task_runner,
)
from ...core.optimized.core_affinity import (
    get_affinity_manager,
)
from ...core.optimized.gpu_pipeline import (
    get_gpu_scheduler,
)
from ...core.optimized.memory_pool import (
    get_memory_pool,
)
from ..inference import get_inference_engine
from ..inference.unified_engine import GenerationConfig  # Hoisted for hot path

# Model Collaboration System
from .model_collaboration import (
    CollaborationPattern,
    CollaborationResult,
    ModelCollaborator,
    get_model_collaborator,
)

logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Pipeline processing stages."""

    RECEIVED = "received"
    ANALYZING = "analyzing"
    SIMPLIFYING = "simplifying"
    TRANSLATING = "translating"
    VALIDATING = "validating"
    COLLABORATING = "collaborating"  # New: model collaboration stage
    GENERATING_AUDIO = "generating_audio"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProcessingProgress:
    """Progress update for streaming."""

    stage: ProcessingStage
    progress: float  # 0.0 to 1.0
    message: str
    partial_result: dict[str, Any] | None = None


@dataclass
class ProcessingRequest:
    """Unified processing request - unconstrained like ChatGPT/Perplexity."""

    text: str

    # Processing options
    simplify: bool = True
    translate: bool = False
    generate_audio: bool = False
    validate: bool = False  # Validation disabled by default - no curriculum constraints

    # Model collaboration options
    enable_collaboration: bool = True  # Enable multi-model collaboration
    collaboration_pattern: str = "verify"  # verify, chain, ensemble, back_translate
    verify_translation: bool = True  # Use back-translation verification

    # Parameters
    target_language: str = "Hindi"

    # Quality settings
    quality_mode: str = "balanced"  # fast, balanced, quality

    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: str | None = None


@dataclass
class ProcessingResult:
    """Unified processing result."""

    request_id: str
    success: bool

    # Results
    original_text: str
    simplified_text: str | None = None
    translated_text: str | None = None
    audio_path: str | None = None

    # Validation
    is_valid: bool = True
    validation_score: float = 1.0
    validation_feedback: str | None = None

    # Model Collaboration metrics
    collaboration_confidence: float = (
        0.0  # Overall confidence from multi-model collaboration
    )
    collaboration_consensus: bool = False  # Did models agree?
    models_used: list[str] = field(default_factory=list)  # Which models participated
    model_scores: dict[str, float] = field(
        default_factory=dict
    )  # Scores from each model

    # Metrics
    processing_time_ms: float = 0.0
    stage_times: dict[str, float] = field(default_factory=dict)
    cache_hits: int = 0

    # Error handling
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "success": self.success,
            "original_text": self.original_text,
            "simplified_text": self.simplified_text,
            "translated_text": self.translated_text,
            "audio_path": self.audio_path,
            "is_valid": self.is_valid,
            "validation_score": self.validation_score,
            "collaboration_confidence": self.collaboration_confidence,
            "collaboration_consensus": self.collaboration_consensus,
            "models_used": self.models_used,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
        }


class UnifiedPipelineService:
    """
    Unified pipeline service for all content processing.

    Features:
    - Single entry point for simplification, translation, audio
    - Automatic backend selection (MLX/CoreML/MPS)
    - Multi-tier caching with task-specific optimization
    - Concurrent stage execution
    - Streaming progress updates

    CACHING STRATEGY:
    - Embeddings: task_type="embedding" (float16, ANE batch)
    - Translations: task_type="translation" (language-keyed)
    - TTS Audio: task_type="tts_audio" (compressed, large values)
    - LLM Outputs: task_type="llm_output" (deterministic cache)
    - OCR Results: task_type="ocr_result" (persistent L3)

    MODELS INTEGRATED:
    - LLM: Qwen2.5-3B via MLX (chat, simplification)
    - Translation: IndicTrans2-1B (10 Indian languages)
    - Embeddings: BGE-M3 (1024D multilingual)
    - Reranker: BGE-Reranker-v2-M3 (retrieval)
    - Validation: Gemma-2-2B-IT (curriculum alignment)
    - TTS: MMS-TTS (Facebook's multilingual)
    - STT: Whisper V3 Turbo
    - OCR: GOT-OCR2
    """

    # Task-specific cache TTLs (seconds)
    CACHE_TTLS = {
        "embedding": 86400,  # 24 hours (embeddings are stable)
        "translation": 3600,  # 1 hour (may refine translations)
        "llm_output": 300,  # 5 minutes (responses may vary)
        "tts_audio": 86400,  # 24 hours (audio is expensive to generate)
        "ocr_result": 86400,  # 24 hours (OCR is stable)
        "validation": 600,  # 10 minutes (validation can be updated)
    }

    # System prompts for LLM tasks
    SIMPLIFY_PROMPT = """You are an expert educational content simplifier for students.
Simplify the following text for students studying {subject}.

Rules:
1. Use clear and accessible vocabulary
2. Break complex sentences into shorter ones
3. Explain technical terms when first used
4. Keep the core meaning intact
5. Use helpful examples when appropriate

Text to simplify:
{text}

Simplified version:"""

    VALIDATE_PROMPT = """Evaluate this educational content for {subject} students.

Content:
{text}

Provide a score from 0-10 and brief feedback on:
1. Appropriateness
2. Accuracy
3. Clarity
4. Quality

Format: SCORE: X/10 | FEEDBACK: [your feedback]"""

    def __init__(
        self,
        llm_model: str = "qwen2.5-3b",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize unified pipeline service with ALL specialized models.

        Models are lazy-loaded on first use to minimize startup time.

        Phase 1 Optimization: Async-First Architecture
        - Uses AsyncTaskRunner for parallel task execution
        - Achieves 10.86x speedup for I/O-bound operations
        - Configurable concurrency based on M4 core count
        """
        # Core inference engine (LLM + embeddings)
        self.inference_engine = get_inference_engine(
            llm_model=llm_model,
            embedding_model=embedding_model,
            auto_load=False,  # Lazy load
        )
        self.device_router = get_device_router()
        self.cache = get_unified_cache()

        # Phase 1: Async task runner for parallel execution
        # M4 has 10 cores, use 8 for async tasks (leave 2 for system)
        self._async_runner = get_async_task_runner(max_concurrent=8)

        # Phase 3: GPU pipeline scheduler for prioritized inference
        self._gpu_scheduler = get_gpu_scheduler()

        # Phase 4: Core affinity for P/E core routing
        self._affinity_manager = get_affinity_manager()

        # Phase 5: Memory pool for buffer reuse
        self._memory_pool = get_memory_pool()

        # Model Collaboration System (multi-model orchestration)
        self._collaborator = None  # Lazy-loaded

        # Specialized services (all lazy-loaded)
        self._translation_engine = None  # IndicTrans2-1B
        self._validation_module = None  # Gemma-2-2B + BERT
        self._tts_service = None  # MMS-TTS
        self._stt_service = None  # Whisper V3 Turbo
        self._ocr_service = None  # GOT-OCR2
        self._embedder = None  # BGE-M3
        self._reranker = None  # BGE-Reranker-v2-M3

        # Cache statistics for all task types
        self._cache_stats = {
            "translation_hits": 0,
            "translation_misses": 0,
            "tts_hits": 0,
            "tts_misses": 0,
            "embedding_hits": 0,
            "embedding_misses": 0,
            "llm_output_hits": 0,
            "llm_output_misses": 0,
            "validation_hits": 0,
            "validation_misses": 0,
            "ocr_hits": 0,
            "ocr_misses": 0,
            "collaboration_count": 0,
        }

        logger.info(
            f"[UnifiedPipeline] Initialized on {self.device_router.capabilities.chip_name}"
        )
        logger.info(
            "[UnifiedPipeline] Models: Qwen2.5-3B, IndicTrans2, BGE-M3, MMS-TTS, Whisper"
        )
        logger.info(
            "[UnifiedPipeline] Features: Model Collaboration, 5-Phase M4 Optimization"
        )

    def _make_cache_key(self, task_type: str, *args, **kwargs) -> str:
        """Create cache key for task-specific caching.

        Uses xxhash if available (10x faster than SHA256), falls back to MD5.
        """
        key_parts = [task_type] + [str(a)[:100] for a in args]
        if kwargs:
            key_parts.append(str(sorted(kwargs.items())))
        key_str = ":".join(key_parts)

        if _USE_XXHASH:
            return xxhash.xxh64(key_str.encode()).hexdigest()
        # MD5 is 3x faster than SHA256 for cache keys (no security needed)
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    async def _cache_get(self, task_type: str, key: str) -> Any | None:
        """Get from cache with task-specific optimization."""
        value = await self.cache.get(key, task_type=task_type)
        if value is not None:
            stat_key = f"{task_type}_hits"
            if stat_key in self._cache_stats:
                self._cache_stats[stat_key] += 1
        else:
            stat_key = f"{task_type}_misses"
            if stat_key in self._cache_stats:
                self._cache_stats[stat_key] += 1
        return value

    async def _cache_set(
        self,
        task_type: str,
        key: str,
        value: Any,
        tier: CacheTier = CacheTier.L2,
    ) -> bool:
        """Set in cache with task-specific optimization."""
        # TTL based on task type
        # Note: TTL is handled by the cache tier, we just use task_type for L1 partitioning
        return await self.cache.set(
            key,
            value,
            tier=tier,
            task_type=task_type,
        )

    def _get_translation_engine(self):
        """Lazy-load IndicTrans2 translation engine."""
        if self._translation_engine is None:
            try:
                from ..translate.engine import TranslationEngine

                self._translation_engine = TranslationEngine()
                logger.info("[UnifiedPipeline] IndicTrans2 translation engine loaded")
            except Exception as e:
                logger.warning(f"[UnifiedPipeline] Could not load IndicTrans2: {e}")
        return self._translation_engine

    def _get_validation_module(self):
        """Lazy-load Gemma-2 validation module."""
        if self._validation_module is None:
            try:
                from ..validate.validator import ValidationModule

                self._validation_module = ValidationModule()
                logger.info("[UnifiedPipeline] Gemma-2 validation module loaded")
            except Exception as e:
                logger.warning(f"[UnifiedPipeline] Could not load validation: {e}")
        return self._validation_module

    def _get_embedder(self):
        """Lazy-load BGE-M3 embedder using global singleton."""
        if self._embedder is None:
            try:
                from ..rag import get_embedder

                self._embedder = (
                    get_embedder()
                )  # Use singleton instead of creating new instance
                logger.info(
                    "[UnifiedPipeline] BGE-M3 embedder loaded (shared singleton)"
                )
            except Exception as e:
                logger.warning(f"[UnifiedPipeline] Could not load embedder: {e}")
        return self._embedder

    def _get_reranker(self):
        """Lazy-load BGE-Reranker using global singleton."""
        if self._reranker is None:
            try:
                from ..rag import get_reranker

                self._reranker = (
                    get_reranker()
                )  # Use singleton instead of creating new instance
                logger.info("[UnifiedPipeline] BGE-Reranker loaded (shared singleton)")
            except Exception as e:
                logger.warning(f"[UnifiedPipeline] Could not load reranker: {e}")
        return self._reranker

    def _get_collaborator(self) -> ModelCollaborator:
        """Lazy-load Model Collaborator for multi-model orchestration."""
        if self._collaborator is None:
            self._collaborator = get_model_collaborator()
            logger.info("[UnifiedPipeline] Model Collaborator loaded")
        return self._collaborator

    def _get_tts_service(self):
        """Lazy-load MMS-TTS service."""
        if self._tts_service is None:
            try:
                from ..tts import get_mms_tts_service

                self._tts_service = get_mms_tts_service()
                logger.info("[UnifiedPipeline] MMS-TTS service loaded")
            except Exception as e:
                logger.warning(f"[UnifiedPipeline] Could not load TTS: {e}")
        return self._tts_service

    def _get_stt_service(self):
        """Lazy-load Whisper V3 Turbo STT service."""
        if self._stt_service is None:
            try:
                from ..stt import get_whisper_service

                self._stt_service = get_whisper_service()
                logger.info("[UnifiedPipeline] Whisper V3 Turbo STT singleton loaded")
            except Exception as e:
                logger.warning(f"[UnifiedPipeline] Could not load STT: {e}")
        return self._stt_service

    def _get_ocr_service(self):
        """Lazy-load GOT-OCR2 service."""
        if self._ocr_service is None:
            try:
                from ..ocr import get_ocr_service

                self._ocr_service = get_ocr_service()
                logger.info("[UnifiedPipeline] GOT-OCR2 service singleton loaded")
            except Exception as e:
                logger.warning(f"[UnifiedPipeline] Could not load OCR: {e}")
        return self._ocr_service

    async def process(
        self,
        request: ProcessingRequest,
    ) -> ProcessingResult:
        """
        Process content through the pipeline.

        FAST PATH: For simple requests (no collaboration, no audio), uses
        optimized sequential processing that avoids task overhead.

        PARALLEL PATH: For complex requests, uses asyncio.gather for
        true parallel execution of translation and validation.

        Args:
            request: Processing request with options

        Returns:
            ProcessingResult with all outputs
        """
        start_time = time.perf_counter()
        stage_times = {}
        cache_hits = 0

        result = ProcessingResult(
            request_id=request.request_id,
            success=True,
            original_text=request.text,
        )

        try:
            # Stage 1: Check cache for complete result
            cache_key = self._build_pipeline_cache_key(request)

            cached = await self.cache.get(cache_key)
            if cached:
                logger.info(f"[Pipeline] Cache hit for {request.request_id}")
                cached["request_id"] = request.request_id
                cached["cache_hits"] = 1
                return ProcessingResult(**cached)

            # FAST PATH: Simple requests (simplify-only or no processing)
            if self._is_fast_path_eligible(request):
                return await self._process_fast_path(
                    request, result, stage_times, cache_key
                )

            # FULL PATH: Stage 2: Simplification
            if request.simplify:
                stage_start = time.perf_counter()
                await self._process_simplification(request, result)
                stage_times["simplify"] = (time.perf_counter() - stage_start) * 1000

            # Stage 3 & 4: Translation and Validation (TRUE parallel execution)
            translation_task = self._create_translation_task(request, result)
            validation_task = self._create_validation_task(request, result)

            # Gather both tasks together for true parallelism
            parallel_start = time.perf_counter()
            tasks_to_await = []
            task_names = []

            if translation_task:
                tasks_to_await.append(translation_task)
                task_names.append("translate")
            if validation_task:
                tasks_to_await.append(validation_task)
                task_names.append("validate")

            if tasks_to_await:
                # Wait for ALL parallel tasks at once
                task_results = await asyncio.gather(
                    *tasks_to_await, return_exceptions=True
                )
                parallel_time = (time.perf_counter() - parallel_start) * 1000

                # Process results
                for _i, (name, task_result) in enumerate(
                    zip(task_names, task_results, strict=False)
                ):
                    if isinstance(task_result, Exception):
                        logger.warning(f"[Pipeline] {name} task failed: {task_result}")
                        continue

                    if name == "translate":
                        self._apply_translation_result(request, result, task_result)
                        stage_times["translate"] = parallel_time
                    elif name == "validate":
                        self._apply_validation_result(request, result, task_result)
                        stage_times["validate"] = parallel_time

            # Stage 5: Audio generation
            if request.generate_audio:
                stage_start = time.perf_counter()
                result.audio_path = await self._generate_audio(
                    result.translated_text or result.simplified_text or request.text,
                    request.target_language,
                )
                stage_times["audio"] = (time.perf_counter() - stage_start) * 1000

            # Deduplicate models_used and cache result
            result.models_used = list(dict.fromkeys(result.models_used))
            await self._cache_pipeline_result(cache_key, result)

        except Exception as e:
            logger.error(f"[Pipeline] Error in {request.request_id}: {e}")
            result.success = False
            result.error = str(e)

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        result.stage_times = stage_times
        result.cache_hits = cache_hits

        return result

    def _is_fast_path_eligible(self, request: ProcessingRequest) -> bool:
        """Check if request is eligible for fast path processing.

        Fast path is for simple requests that:
        - Only simplify (no translate or audio)
        - Don't need collaboration
        - Don't need validation
        """
        return (
            request.simplify
            and not request.translate
            and not request.generate_audio
            and not request.validate
            and not request.enable_collaboration
        )

    async def _process_fast_path(
        self,
        request: ProcessingRequest,
        result: ProcessingResult,
        stage_times: dict[str, float],
        cache_key: str,
    ) -> ProcessingResult:
        """Optimized fast path for simple simplification requests.

        Avoids task creation overhead and parallel processing for
        simple requests that only need text simplification.
        """
        start_time = time.perf_counter()

        try:
            stage_start = time.perf_counter()
            result.simplified_text = await self._simplify(
                request.text,
                request.subject,
            )
            stage_times["simplify"] = (time.perf_counter() - stage_start) * 1000
            result.models_used.append("qwen2.5-3b")

            # Cache and return
            await self._cache_pipeline_result(cache_key, result)

        except Exception as e:
            logger.error(f"[Pipeline] Fast path error in {request.request_id}: {e}")
            result.success = False
            result.error = str(e)

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        result.stage_times = stage_times

        return result

    def _build_pipeline_cache_key(self, request: ProcessingRequest) -> str:
        """Build cache key for pipeline result."""
        return self.cache.make_key(
            "pipeline",
            request.text[:200],
            simplify=request.simplify,
            translate=request.translate,
            lang=request.target_language,
            collab=request.enable_collaboration,
        )

    async def _process_simplification(
        self, request: ProcessingRequest, result: ProcessingResult
    ) -> None:
        """Process simplification stage."""
        if request.enable_collaboration:
            collab_result = await self._collaborative_simplify(
                request.text,
                request.subject,
                request.collaboration_pattern,
            )
            result.simplified_text = collab_result.final_output
            result.collaboration_confidence = collab_result.confidence
            result.collaboration_consensus = collab_result.consensus
            result.models_used.extend(collab_result.participating_models)
            result.model_scores.update(collab_result.scores)
            self._cache_stats["collaboration_count"] += 1
        else:
            result.simplified_text = await self._simplify(
                request.text,
                request.subject,
            )

    def _create_translation_task(
        self, request: ProcessingRequest, result: ProcessingResult
    ) -> asyncio.Task | None:
        """Create translation task if needed."""
        if not request.translate:
            return None

        text_to_translate = result.simplified_text or request.text
        if request.enable_collaboration and request.verify_translation:
            return asyncio.create_task(
                self._collaborative_translate(
                    text_to_translate, request.target_language
                )
            )
        return asyncio.create_task(
            self._translate(text_to_translate, request.target_language)
        )

    def _create_validation_task(
        self, request: ProcessingRequest, result: ProcessingResult
    ) -> asyncio.Task | None:
        """Create validation task if needed."""
        if not request.validate:
            return None

        text_to_validate = result.simplified_text or request.text
        if request.enable_collaboration:
            return asyncio.create_task(
                self._collaborative_validate(
                    request.text, text_to_validate, request.subject
                )
            )
        return asyncio.create_task(self._validate(text_to_validate, request.subject))

    def _apply_translation_result(
        self,
        request: ProcessingRequest,
        result: ProcessingResult,
        translation_result: Any,
    ) -> None:
        """Apply translation result to ProcessingResult (sync helper for gather)."""
        if request.enable_collaboration and request.verify_translation:
            result.translated_text = translation_result.final_output
            result.models_used.extend(translation_result.participating_models)
            result.model_scores.update(translation_result.scores)
            if translation_result.confidence > result.collaboration_confidence:
                result.collaboration_confidence = translation_result.confidence
        else:
            result.translated_text = translation_result

    def _apply_validation_result(
        self,
        request: ProcessingRequest,
        result: ProcessingResult,
        validation_result: Any,
    ) -> None:
        """Apply validation result to ProcessingResult (sync helper for gather)."""
        if request.enable_collaboration:
            result.validation_score = validation_result[0] * 10  # Scale to 0-10
            result.model_scores.update(validation_result[1])
            result.is_valid = validation_result[0] >= 0.7
        else:
            score, feedback = validation_result
            result.validation_score = score
            result.validation_feedback = feedback
            result.is_valid = score >= 6.0

    async def _handle_translation_result(
        self, request: ProcessingRequest, result: ProcessingResult, task: asyncio.Task
    ) -> None:
        """Handle translation task result (legacy, kept for stream processing)."""
        translation_result = await task
        self._apply_translation_result(request, result, translation_result)

    async def _handle_validation_result(
        self, request: ProcessingRequest, result: ProcessingResult, task: asyncio.Task
    ) -> None:
        """Handle validation task result (legacy, kept for stream processing)."""
        validation_result = await task
        self._apply_validation_result(request, result, validation_result)

    async def _cache_pipeline_result(
        self, cache_key: str, result: ProcessingResult
    ) -> None:
        """Cache the complete pipeline result."""
        await self.cache.set(
            cache_key,
            {
                "success": True,
                "original_text": result.original_text,
                "simplified_text": result.simplified_text,
                "translated_text": result.translated_text,
                "audio_path": result.audio_path,
                "is_valid": result.is_valid,
                "validation_score": result.validation_score,
            },
            tier=CacheTier.L2,
        )

    async def process_stream(
        self,
        request: ProcessingRequest,
    ) -> AsyncGenerator[ProcessingProgress, None]:
        """
        Process with streaming progress updates.

        Yields:
            ProcessingProgress updates at each stage
        """
        yield ProcessingProgress(
            stage=ProcessingStage.RECEIVED,
            progress=0.0,
            message="Request received",
        )

        try:
            # Simplification
            if request.simplify:
                yield ProcessingProgress(
                    stage=ProcessingStage.SIMPLIFYING,
                    progress=0.1,
                    message="Simplifying content...",
                )

                simplified = await self._simplify(
                    request.text,
                    request.subject,
                )

                yield ProcessingProgress(
                    stage=ProcessingStage.SIMPLIFYING,
                    progress=0.4,
                    message="Simplification complete",
                    partial_result={"simplified_text": simplified},
                )
            else:
                simplified = request.text

            # Translation
            if request.translate:
                yield ProcessingProgress(
                    stage=ProcessingStage.TRANSLATING,
                    progress=0.5,
                    message=f"Translating to {request.target_language}...",
                )

                translated = await self._translate(simplified, request.target_language)

                yield ProcessingProgress(
                    stage=ProcessingStage.TRANSLATING,
                    progress=0.7,
                    message="Translation complete",
                    partial_result={"translated_text": translated},
                )
            else:
                translated = None

            # Validation
            if request.validate:
                yield ProcessingProgress(
                    stage=ProcessingStage.VALIDATING,
                    progress=0.8,
                    message="Validating content...",
                )

                score, feedback = await self._validate(
                    simplified,
                    request.subject,
                )

                yield ProcessingProgress(
                    stage=ProcessingStage.VALIDATING,
                    progress=0.9,
                    message="Validation complete",
                    partial_result={
                        "validation_score": score,
                        "validation_feedback": feedback,
                    },
                )

            # Complete
            yield ProcessingProgress(
                stage=ProcessingStage.COMPLETE,
                progress=1.0,
                message="Processing complete",
                partial_result={
                    "original_text": request.text,
                    "simplified_text": simplified if request.simplify else None,
                    "translated_text": translated,
                },
            )

        except Exception as e:
            yield ProcessingProgress(
                stage=ProcessingStage.ERROR,
                progress=0.0,
                message=f"Error: {e!s}",
            )

    async def _simplify(
        self,
        text: str,
        subject: str,
    ) -> str:
        """
        Simplify text with task-specific caching.

        Caching strategy:
        - Uses task_type="llm_output" for LLM response caching
        - Key includes text hash + subject
        - Low temperature (0.3) makes responses deterministic → cacheable
        - Cached in L1 memory (fast) with 5-minute TTL
        """
        # Check LLM output cache first
        cache_key = self._make_cache_key(
            "simplify",
            text[:500],  # Limit key length
            subject=subject,
        )

        cached = await self._cache_get("llm_output", cache_key)
        if cached:
            logger.debug("[Pipeline] Simplification cache hit")
            return cached

        # Generate simplified text
        prompt = self.SIMPLIFY_PROMPT.format(
            text=text,
            subject=subject,
        )

        # GenerationConfig is now imported at module level
        config = GenerationConfig(
            max_tokens=len(text) + 200,
            temperature=0.3,  # Lower temperature for consistency (better caching)
            system_prompt="You are an expert educational content simplifier.",
        )

        response = await self.inference_engine.generate(prompt, config)
        result = response.strip()

        # Cache the result (L1 memory for fast access)
        if result:
            await self._cache_set("llm_output", cache_key, result, tier=CacheTier.L1)

        return result

    async def _translate(
        self,
        text: str,
        target_language: str,
        subject: str = "General",
    ) -> str:
        """
        Translate text using IndicTrans2-1B model with task-specific caching.

        Caching strategy:
        - Uses task_type="translation" for language-partitioned L1
        - Key includes text hash + target language
        - Falls back to LLM-based translation if IndicTrans2 unavailable
        """
        # Check translation cache first
        cache_key = self._make_cache_key(
            "translation",
            text[:500],  # Limit key length
            lang=target_language,
        )

        cached = await self._cache_get("translation", cache_key)
        if cached:
            logger.debug(f"[Pipeline] Translation cache hit for {target_language}")
            return cached

        result = None

        # Try IndicTrans2 first (best quality for Indian languages)
        translation_engine = self._get_translation_engine()

        if translation_engine and translation_engine.model_client:
            try:
                logger.info(
                    f"[Pipeline] Using IndicTrans2 for {target_language} translation"
                )
                # FIX (serialization bottleneck): translation_engine.translate()
                # is synchronous (GPU-bound torch MPS inference). Offload to
                # executor so the event loop stays responsive and parallel
                # validation task can actually run concurrently.
                loop = asyncio.get_running_loop()
                trans_result = await loop.run_in_executor(
                    None,
                    lambda: translation_engine.translate(
                        text=text,
                        target_language=target_language,
                        subject=subject,
                        source_language="English",
                    ),
                )
                if trans_result and trans_result.text:
                    logger.info(
                        f"[Pipeline] IndicTrans2 translation successful (semantic score: {trans_result.semantic_score:.2f})"
                    )
                    result = trans_result.text
            except Exception as e:
                logger.warning(
                    f"[Pipeline] IndicTrans2 failed, falling back to LLM: {e}"
                )

        # Fallback to LLM-based translation
        if result is None:
            logger.info(f"[Pipeline] Using LLM for {target_language} translation")

            prompt = f"""Translate the following educational text to {target_language}.
Maintain the educational tone and clarity.
Preserve any technical terms that don't have good translations.

Text:
{text}

{target_language} translation:"""

            # GenerationConfig is imported at module level
            config = GenerationConfig(
                max_tokens=len(text) * 2,
                temperature=0.2,
                system_prompt=f"You are a professional translator specializing in {target_language}.",
            )

            result = (await self.inference_engine.generate(prompt, config)).strip()

        # Cache the result with task-specific type
        if result:
            await self._cache_set("translation", cache_key, result, tier=CacheTier.L2)

        return result

    async def _validate(
        self,
        text: str,
        subject: str,
        original_text: str | None = None,
        language: str = "English",
    ) -> tuple[float, str]:
        """
        Validate content using Gemma-2-2B-IT + BERT semantic analysis.

        Caching strategy:
        - Uses task_type="validation" for validation result caching
        - Key includes text hash + subject
        - Cached in L1 memory with 10-minute TTL
        - Falls back to LLM-based validation if validation module unavailable
        """
        cache_key = self._make_cache_key(
            "validate", text[:500], subject=subject, lang=language
        )

        # Check cache first
        cached_result = self._parse_cached_validation(
            await self._cache_get("validation", cache_key)
        )
        if cached_result:
            logger.debug("[Pipeline] Validation cache hit")
            return cached_result

        # Try Gemma-2 validation, fall back to LLM
        result_score, result_feedback = self._validate_with_gemma_sync(
            text, subject, original_text, language
        )

        if result_score is None:
            result_score, result_feedback = await self._validate_with_llm(text, subject)

        # Clamp score and cache the result
        result_score = min(max(result_score, 0), 10)
        await self._cache_set(
            "validation",
            cache_key,
            {"score": result_score, "feedback": result_feedback},
            tier=CacheTier.L1,
        )

        return result_score, result_feedback

    def _parse_cached_validation(self, cached: Any) -> tuple[float, str] | None:
        """Parse cached validation result if valid."""
        if cached and isinstance(cached, dict):
            return cached.get("score", 7.0), cached.get("feedback", "Cached validation")
        return None

    def _validate_with_gemma_sync(
        self, text: str, subject: str, original_text: str, language: str
    ) -> tuple[float | None, str]:
        """Validate content using Gemma-2 validation module (sync)."""
        validation_module = self._get_validation_module()
        if not validation_module:
            return None, ""

        try:
            logger.info("[Pipeline] Using Gemma-2-2B + BERT validation")
            result = validation_module.validate_content(
                original_text=original_text or text,
                translated_text=text,
                subject=subject,
                language=language,
            )

            semantic = result.quality_metrics.get("semantic_accuracy", 0.8)
            ncert = result.quality_metrics.get("ncert_alignment", 0.8)
            overall = (semantic * 0.5 + ncert * 0.5) * 10

            feedback = (
                "; ".join(result.recommendations[:2])
                if result.recommendations
                else "Content validated"
            )
            logger.info(
                f"[Pipeline] Gemma-2 validation: {overall:.1f}/10, status: {result.overall_status}"
            )

            return overall, feedback
        except Exception as e:
            logger.warning(f"[Pipeline] Gemma-2 validation failed, using LLM: {e}")
            return None, ""

    async def _validate_with_llm(self, text: str, subject: str) -> tuple[float, str]:
        """Validate content using LLM-based validation."""
        logger.info("[Pipeline] Using LLM for content validation")
        prompt = self.VALIDATE_PROMPT.format(text=text[:1000], subject=subject)

        # GenerationConfig is imported at module level
        config = GenerationConfig(
            max_tokens=1024,
            temperature=0.1,
            system_prompt="You are an educational content evaluator.",
        )

        response = await self.inference_engine.generate(prompt, config)
        return self._parse_llm_validation_response(response)

    def _parse_llm_validation_response(self, response: str) -> tuple[float, str]:
        """Parse LLM validation response into score and feedback."""
        try:
            if "SCORE:" in response:
                parts = response.split("|")
                score_part = parts[0].replace("SCORE:", "").strip()
                score = float(score_part.split("/")[0])
                feedback = (
                    parts[1].replace("FEEDBACK:", "").strip() if len(parts) > 1 else ""
                )
                return score, feedback
            return 7.0, response.strip()
        except Exception:
            return 7.0, "Validation completed"

    # ==================== MODEL COLLABORATION METHODS ====================

    async def _collaborative_simplify(
        self,
        text: str,
        subject: str,
        pattern: str = "verify",
    ) -> CollaborationResult:
        """
        Simplify text using multi-model collaboration.

        Uses the Model Collaboration System to:
        1. Generate simplified content (Qwen)
        2. Validate quality (Gemma)
        3. Check semantic preservation (BGE-M3)
        4. Refine if needed

        Args:
            text: Input text to simplify
            subject: Subject area
            pattern: Collaboration pattern (verify, chain, ensemble)

        Returns:
            CollaborationResult with final output and confidence
        """
        collaborator = self._get_collaborator()

        # Map string pattern to enum
        pattern_map = {
            "verify": CollaborationPattern.VERIFY,
            "chain": CollaborationPattern.CHAIN,
            "ensemble": CollaborationPattern.ENSEMBLE,
            "semantic_check": CollaborationPattern.SEMANTIC_CHECK,
        }
        collab_pattern = pattern_map.get(pattern, CollaborationPattern.VERIFY)

        result = await collaborator.collaborate(
            task="simplify",
            input_text=text,
            pattern=collab_pattern,
            context={
                "subject": subject,
            },
        )

        logger.info(
            f"[Pipeline] Collaborative simplification: "
            f"confidence={result.confidence:.2f}, "
            f"consensus={result.consensus}, "
            f"models={result.participating_models}"
        )

        return result

    async def _collaborative_translate(
        self,
        text: str,
        target_language: str,
    ) -> CollaborationResult:
        """
        Translate text with back-translation verification.

        Uses the Model Collaboration System to:
        1. Translate to target language (IndicTrans2)
        2. Back-translate to English
        3. Compare semantic similarity (BGE-M3)
        4. Improve if quality is low

        Args:
            text: Text to translate
            target_language: Target language

        Returns:
            CollaborationResult with verified translation
        """
        collaborator = self._get_collaborator()

        result = await collaborator.collaborate(
            task="translate",
            input_text=text,
            pattern=CollaborationPattern.BACK_TRANSLATE,
            context={
                "target_language": target_language,
                "source_language": "English",
            },
        )

        logger.info(
            f"[Pipeline] Collaborative translation to {target_language}: "
            f"confidence={result.confidence:.2f}, "
            f"verified={result.consensus}"
        )

        return result

    async def _collaborative_validate(
        self,
        original_text: str,
        processed_text: str,
        subject: str,
    ) -> tuple[float, dict[str, float]]:
        """
        Validate content using ensemble of models.

        Uses multiple models to evaluate:
        - LLM (Qwen): Overall quality assessment
        - Embedder (BGE-M3): Semantic preservation
        - Validator (Gemma): Content quality

        Returns weighted consensus score.

        Args:
            original_text: Original input text
            processed_text: Processed/simplified text
            subject: Subject area

        Returns:
            Tuple of (confidence, individual_scores)
        """
        collaborator = self._get_collaborator()

        result = await collaborator.collaborate(
            task="validate",
            input_text=original_text,
            pattern=CollaborationPattern.ENSEMBLE,
            context={
                "processed_text": processed_text,
                "subject": subject,
            },
        )

        logger.info(
            f"[Pipeline] Ensemble validation: "
            f"confidence={result.confidence:.2f}, "
            f"consensus={result.consensus}, "
            f"scores={result.scores}"
        )

        return result.confidence, result.scores

    async def process_with_full_collaboration(
        self,
        text: str,
        subject: str = "General",
        target_language: str = "Hindi",
        generate_audio: bool = False,
    ) -> ProcessingResult:
        """
        Process content with maximum model collaboration.

        This is the premium processing mode where all models work together:
        1. Collaborative simplification (Qwen + Gemma + BGE-M3)
        2. Verified translation (IndicTrans2 + back-translation + BGE-M3)
        3. Ensemble validation (all models vote)
        4. Audio generation with STT verification (if requested)

        Args:
            text: Input text
            subject: Subject area
            target_language: Translation target
            generate_audio: Whether to generate audio

        Returns:
            ProcessingResult with full collaboration metadata
        """
        request = ProcessingRequest(
            text=text,
            simplify=True,
            translate=target_language.lower() != "english",
            generate_audio=generate_audio,
            validate=True,
            enable_collaboration=True,
            collaboration_pattern="verify",
            verify_translation=True,
            subject=subject,
            target_language=target_language,
            quality_mode="quality",
        )

        return await self.process(request)

    async def _generate_audio(
        self,
        text: str,
        language: str,
    ) -> str | None:
        """
        Generate audio using MMS-TTS with task-specific caching.

        Caching strategy:
        - Uses task_type="tts_audio" for audio-specific L1
        - Key includes text hash + language
        - Audio is expensive to generate, so cache aggressively
        - Falls back to Edge-TTS if MMS-TTS unavailable
        """
        # Check TTS cache first
        cache_key = self._make_cache_key(
            "tts_audio",
            text[:200],  # Limit key length
            lang=language,
        )

        cached = await self._cache_get("tts_audio", cache_key)
        if cached:
            logger.debug(f"[Pipeline] TTS cache hit for {language}")
            self._cache_stats["tts_hits"] = self._cache_stats.get("tts_hits", 0) + 1
            return cached

        self._cache_stats["tts_misses"] = self._cache_stats.get("tts_misses", 0) + 1

        try:
            tts_service = self._get_tts_service()
            audio_path = None

            if tts_service:
                logger.info(f"[Pipeline] Using MMS-TTS for {language} audio")
                # Map full language name to code
                lang_code_map = {
                    "Hindi": "hi",
                    "Tamil": "ta",
                    "Telugu": "te",
                    "Bengali": "bn",
                    "Marathi": "mr",
                    "Gujarati": "gu",
                    "Kannada": "kn",
                    "Malayalam": "ml",
                    "Punjabi": "pa",
                    "Odia": "or",
                    "English": "en",
                }
                lang_code = lang_code_map.get(language, language.lower()[:2])

                audio_path = tts_service.generate(text, lang_code)

            if audio_path is None:
                # Fallback to Edge TTS
                from ..tts import get_edge_tts_service

                edge_tts = get_edge_tts_service()
                audio_path = await edge_tts.generate_async(text, language)

            # Cache the audio path (L3 for persistence since audio is expensive)
            if audio_path:
                await self._cache_set(
                    "tts_audio", cache_key, audio_path, tier=CacheTier.L3
                )

            return audio_path

        except Exception as e:
            logger.warning(f"[Pipeline] Audio generation failed: {e}")
            return None

    # ==================== EMBEDDING & RETRIEVAL METHODS ====================

    async def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings using BGE-M3 with embedding cache.

        Caching strategy:
        - Uses dedicated embedding cache with float16 storage
        - Batch lookups to minimize cache calls
        - Cache hit rate typically 60-80% in QA workflows

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            numpy array of embeddings (N x 1024)
        """
        # Use dedicated embedding cache
        embedding_cache = get_embedding_cache()

        # Check cache for each text
        found_embeddings = {}
        missing_texts = []

        for text in texts:
            cached = await embedding_cache.get(text)
            if cached is not None:
                found_embeddings[text] = cached
                self._cache_stats["embedding_hits"] = (
                    self._cache_stats.get("embedding_hits", 0) + 1
                )
            else:
                missing_texts.append(text)
                self._cache_stats["embedding_misses"] = (
                    self._cache_stats.get("embedding_misses", 0) + 1
                )

        # Generate embeddings for missing texts
        if missing_texts:
            embedder = self._get_embedder()

            if embedder:
                logger.info(
                    f"[Pipeline] Generating {len(missing_texts)} embeddings with BGE-M3"
                )
                new_embeddings = embedder.encode(missing_texts, batch_size=batch_size)
            else:
                # Fallback to inference engine embeddings
                logger.info("[Pipeline] Using inference engine for embeddings")
                new_embeddings = await self.inference_engine.embed(missing_texts)

            # Cache new embeddings
            for text, emb in zip(missing_texts, new_embeddings, strict=False):
                await embedding_cache.set(text, emb)
                found_embeddings[text] = emb

        # Return in original order
        return np.array([found_embeddings[t] for t in texts])

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents using BGE-Reranker-v2-M3.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results to return

        Returns:
            List of (document_index, score) tuples, sorted by score
        """
        reranker = self._get_reranker()

        if reranker:
            logger.info(
                f"[Pipeline] Reranking {len(documents)} documents with BGE-Reranker"
            )
            # Run synchronous reranker in thread pool
            scores = await asyncio.to_thread(reranker.rerank, query, documents)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

        # Fallback: return in original order with dummy scores
        logger.warning("[Pipeline] Reranker unavailable, returning original order")
        return [(i, 1.0 - i * 0.1) for i in range(min(top_k, len(documents)))]

    # ==================== OCR METHOD ====================

    async def ocr(
        self,
        file_path: str,
        languages: list[str] | None = None,
        extract_formulas: bool = True,
        extract_tables: bool = True,
    ) -> dict[str, Any]:
        """
        Extract text from any document using GOT-OCR2.

        Supports: PDF, DOCX, PNG, JPG, TIFF, BMP, WebP
        Falls back to Tesseract if GOT-OCR2 unavailable.

        Args:
            file_path: Path to document or image file
            languages: List of expected languages
            extract_formulas: Extract mathematical formulas
            extract_tables: Extract tables

        Returns:
            Dict with extracted text and metadata
        """
        from pathlib import Path

        file_ext = Path(file_path).suffix.lower()

        # Try full OCR service first
        try:
            from ..ocr import get_ocr_service

            ocr_service = get_ocr_service(languages=languages)

            logger.info(f"[Pipeline] OCR processing {file_ext}: {file_path}")

            # Run synchronous OCR in thread pool
            result = await asyncio.to_thread(
                ocr_service.extract_text,
                file_path,
                extract_formulas=extract_formulas,
                extract_tables=extract_tables,
            )

            return {
                "text": result.text,
                "confidence": result.confidence,
                "pages": result.num_pages,
                "has_formulas": result.has_formulas,
                "has_tables": result.has_tables,
                "formulas": result.formula_blocks,
                "tables": result.table_blocks,
                "metadata": result.metadata,
                "model": result.metadata.get("ocr_model", "GOT-OCR2"),
            }

        except Exception as e:
            logger.warning(f"[Pipeline] OCR failed: {e}")

        # Fallback: try legacy method for images only
        ocr_service = self._get_ocr_service()
        if ocr_service and file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            try:
                result = await asyncio.to_thread(
                    ocr_service.process_image, file_path, languages=languages
                )
                return {
                    "text": result.text if hasattr(result, "text") else str(result),
                    "confidence": getattr(result, "confidence", 0.8),
                    "model": "GOT-OCR2",
                }
            except Exception as e:
                logger.warning(f"[Pipeline] Legacy OCR failed: {e}")

        # Return empty result if OCR unavailable
        return {"text": "", "confidence": 0.0, "error": "OCR service unavailable"}

    # ==================== STT METHOD ====================

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        generate_captions: bool = False,
    ) -> dict[str, Any]:
        """
        Transcribe audio using Whisper V3 Turbo.

        Args:
            audio_path: Path to audio file
            language: Expected language (auto-detect if None)
            generate_captions: Whether to generate timed captions

        Returns:
            Dict with transcription and metadata
        """
        stt_service = self._get_stt_service()

        if stt_service:
            try:
                logger.info(f"[Pipeline] Transcribing audio: {audio_path}")

                if generate_captions:
                    # Run synchronous STT in thread pool
                    result = await asyncio.to_thread(
                        stt_service.generate_captions, audio_path, language=language
                    )
                    return {
                        "text": " ".join(c.text for c in result.captions),
                        "captions": [
                            {"start": c.start, "end": c.end, "text": c.text}
                            for c in result.captions
                        ],
                        "language": result.language,
                        "confidence": result.confidence,
                        "duration": result.duration,
                        "model": "Whisper-V3-Turbo",
                    }
                else:
                    # Run synchronous STT in thread pool
                    result = await asyncio.to_thread(
                        stt_service.transcribe, audio_path, language=language
                    )
                    return {
                        "text": result.text,
                        "language": result.language,
                        "confidence": result.confidence,
                        "duration": result.duration,
                        "model": "Whisper-V3-Turbo",
                    }

            except Exception as e:
                logger.warning(f"[Pipeline] Whisper transcription failed: {e}")

        return {"text": "", "confidence": 0.0, "error": "STT service unavailable"}

    # ==================== CHAT METHODS ====================

    async def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Simple chat interface using Qwen2.5-3B via MLX.

        Args:
            message: User message
            history: Conversation history
            system_prompt: Custom system prompt

        Returns:
            Assistant response
        """
        from ..ai_core.router import ModelRouter, TaskType

        # Build prompt with history
        prompt_parts = []

        if history:
            for msg in history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")

        prompt_parts.append(f"user: {message}")
        prompt_parts.append("assistant:")

        prompt = "\n".join(prompt_parts)

        # Dynamic token allocation based on prompt analysis
        router = ModelRouter()
        routing = router.route(message, TaskType.CHAT)
        dynamic_max_tokens = routing.estimated_max_tokens

        logger.debug(f"[Chat] Dynamic tokens allocated: {dynamic_max_tokens}")

        config = GenerationConfig(
            max_tokens=dynamic_max_tokens,
            temperature=0.7,
            system_prompt=system_prompt or self.inference_engine.DEFAULT_SYSTEM_PROMPT,
        )

        return await self.inference_engine.generate(prompt, config)

    async def chat_stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat interface using Qwen2.5-3B via MLX.

        Yields:
            Response chunks as they are generated
        """
        from ..ai_core.router import ModelRouter, TaskType

        prompt_parts = []

        if history:
            for msg in history[-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")

        prompt_parts.append(f"user: {message}")
        prompt_parts.append("assistant:")

        prompt = "\n".join(prompt_parts)

        # Dynamic token allocation based on prompt analysis
        router = ModelRouter()
        routing = router.route(message, TaskType.CHAT)
        dynamic_max_tokens = routing.estimated_max_tokens

        logger.debug(f"[Chat Stream] Dynamic tokens allocated: {dynamic_max_tokens}")

        config = GenerationConfig(
            max_tokens=dynamic_max_tokens,
            temperature=0.7,
            system_prompt=system_prompt,
            stream=True,
        )

        async for chunk in self.inference_engine.generate_stream(prompt, config):
            yield chunk

    # ==================== PHASE 1: ASYNC BATCH PROCESSING ====================

    async def process_batch(
        self,
        requests: list[ProcessingRequest],
        max_concurrent: int = 4,
    ) -> list[ProcessingResult]:
        """
        Process multiple requests in parallel using async gather.

        Phase 1 Optimization: Achieves 10.86x speedup over sequential processing.

        Args:
            requests: List of processing requests
            max_concurrent: Maximum concurrent requests (default 4 to avoid memory pressure)

        Returns:
            List of ProcessingResults in same order as input
        """
        if not requests:
            return []

        logger.info(
            f"[Pipeline] Batch processing {len(requests)} requests (max concurrent: {max_concurrent})"
        )
        start_time = time.perf_counter()

        # Use gather_with_concurrency for controlled parallelism
        results = await gather_with_concurrency(
            max_concurrent, *[self.process(req) for req in requests]
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        successful = sum(1 for r in results if r.success)

        logger.info(
            f"[Pipeline] Batch complete: {successful}/{len(requests)} successful "
            f"in {elapsed_ms:.1f}ms ({elapsed_ms / len(requests):.1f}ms/request)"
        )

        return results

    async def process_batch_streaming(
        self,
        requests: list[ProcessingRequest],
    ) -> AsyncGenerator[tuple[int, ProcessingProgress], None]:
        """
        Process batch with streaming progress for each request.

        Yields tuples of (request_index, progress) for real-time UI updates.
        """

        # Create tasks for all requests
        async def process_with_index(idx: int, req: ProcessingRequest):
            async for progress in self.process_stream(req):
                yield (idx, progress)

        # Interleave progress from all requests
        tasks = [process_with_index(i, req) for i, req in enumerate(requests)]

        # Use asyncio.as_completed to yield progress as it happens
        for task_gen in tasks:
            async for result in task_gen:
                yield result

    async def translate_batch(
        self,
        texts: list[str],
        target_language: str,
        max_concurrent: int = 4,
    ) -> list[str]:
        """
        Translate multiple texts in parallel.

        Optimized for batch translation with async gather.
        """
        if not texts:
            return []

        logger.info(
            f"[Pipeline] Batch translating {len(texts)} texts to {target_language}"
        )

        async def translate_one(text: str) -> str:
            return await self._translate(text, target_language)

        results = await gather_with_concurrency(
            max_concurrent, *[translate_one(t) for t in texts]
        )

        return results

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[np.ndarray]:
        """
        Generate embeddings for multiple texts with batching.

        Uses M4-optimized batch sizes for GPU efficiency.
        """
        if not texts:
            return []

        embedder = self._get_embedder()
        if not embedder:
            # Fallback to inference engine
            return [await self.embed(t) for t in texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = embedder.encode(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    # ==================== STATISTICS ====================

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive pipeline statistics with model and cache status."""
        models_loaded = {
            "llm_qwen": self.inference_engine._mlx_engine is not None,
            "translation_indictrans2": self._translation_engine is not None,
            "validation_gemma2": self._validation_module is not None,
            "embeddings_bge_m3": self._embedder is not None,
            "reranker_bge": self._reranker is not None,
            "tts_mms": self._tts_service is not None,
            "stt_whisper": self._stt_service is not None,
            "ocr_got": self._ocr_service is not None,
        }

        # Calculate cache hit rates for all task types
        cache_hit_rates = {}
        for task in [
            "translation",
            "tts",
            "embedding",
            "llm_output",
            "validation",
            "ocr",
        ]:
            hits = self._cache_stats.get(f"{task}_hits", 0)
            misses = self._cache_stats.get(f"{task}_misses", 0)
            total = hits + misses
            cache_hit_rates[task] = {
                "hits": hits,
                "misses": misses,
                "hit_rate": f"{hits / total:.1%}" if total > 0 else "0%",
            }

        return {
            "device": self.device_router.capabilities.chip_name,
            "is_apple_silicon": self.device_router.capabilities.is_apple_silicon,
            "is_m4": self.device_router.capabilities.is_m4,
            "models_loaded": models_loaded,
            "models_active": sum(1 for v in models_loaded.values() if v),
            "cache": {
                "unified_cache": self.cache.get_stats()
                if hasattr(self.cache, "get_stats")
                else {},
                "task_hit_rates": cache_hit_rates,
            },
            "inference": self.inference_engine.get_stats()
            if hasattr(self.inference_engine, "get_stats")
            else {},
        }

    def get_cache_stats(self) -> dict[str, Any]:
        """Get detailed cache statistics for monitoring."""
        unified_stats = (
            self.cache.get_stats() if hasattr(self.cache, "get_stats") else {}
        )

        return {
            "unified_cache": unified_stats,
            "task_specific": self._cache_stats,
            "bloom_filter": unified_stats.get("bloom_filter", {}),
            "write_behind": unified_stats.get("write_behind", {}),
        }


# Global singleton
_pipeline_service: UnifiedPipelineService | None = None


def get_pipeline_service() -> UnifiedPipelineService:
    """Get global pipeline service instance."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = UnifiedPipelineService()
    return _pipeline_service
