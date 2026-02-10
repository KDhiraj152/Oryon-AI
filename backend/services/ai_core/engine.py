"""
AI Engine - Main Orchestration Layer
=====================================

Central AI engine that coordinates:
- Model loading with M4 hardware optimization
- Context-aware conversation handling
- RAG pipeline integration
- Response formatting and streaming
- Safety checks and validation

NOTE: Safety and filtering behavior is now controlled by the PolicyEngine.
Set ALLOW_UNRESTRICTED_MODE=true to bypass curriculum/educational filters.
See backend/policy/policy_module.py for configuration.
"""

import asyncio
import contextlib
import json
import logging
import threading
import time
import uuid as uuid_module
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Lazy-loaded tiktoken encoder for accurate token counting
_tiktoken_enc: Any = None


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base). Falls back to len//4."""
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_enc = None  # sentinel: don't retry (no tiktoken available)
    if _tiktoken_enc is not None:
        return len(_tiktoken_enc.encode(text))
    return len(text) // 4

from .context import (  # New optimizers
    AdaptiveContextAllocator,
    ContextManager,
    ContextPool,
    ContextRole,
)
from .formatter import FormattedResponse, Intent, ResponseFormatter, SourceReference
from .router import ModelRouter, TaskType
from .compute_budget import ComputeBudget, ComputeClass, classify_compute, get_budget

# Import policy module for centralized configuration
try:
    from ...policy import PolicyMode, get_policy_engine

    _POLICY_AVAILABLE = True
except ImportError:
    _POLICY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stream: bool = True

    # Safety settings - these can be overridden by PolicyEngine
    block_harmful: bool = True
    redact_secrets: bool = True

    # Performance settings
    use_cache: bool = True
    timeout_seconds: float = 30.0

    # RAG settings: None = auto-detect, True = force, False = skip
    use_rag: bool | None = None


@dataclass
class GenerationResult:
    """Result from text generation."""

    content: str
    tokens_prompt: int
    tokens_completion: int
    latency_ms: float
    model_id: str
    cached: bool = False
    sources: list[SourceReference] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class AIEngine:
    """
    Production-grade AI Engine for local deployment.

    Features:
    - Multi-turn conversation with context management
    - RAG integration for document Q&A
    - M4 hardware-optimized inference
    - Streaming token output
    - Response validation and formatting
    - Caching and graceful degradation

    Policy Integration:
    - Uses PolicyEngine for content filtering decisions
    - When ALLOW_UNRESTRICTED_MODE=true, bypasses educational filters
    - System-level safety can still be enabled independently
    """

    def __init__(
        self,
        context_manager: ContextManager | None = None,
        model_router: ModelRouter | None = None,
        formatter: ResponseFormatter | None = None,
        redis_client: Any | None = None,
    ):
        """Initialize the AI Engine."""
        self.context_manager = context_manager or ContextManager(redis_client)
        self.router = model_router or ModelRouter()
        self.formatter = formatter or ResponseFormatter()

        # Lazy-loaded components
        self._llm_client = None
        self._rag_service = None
        self._safety_guard = None
        self._policy_engine = None  # PolicyEngine for centralized filtering
        self._profile_service = None
        self._review_queue = None
        self._collaborator = None  # Multi-model validation
        self._sandbox = None  # Code execution sandbox
        self._self_optimizer = None  # Self-optimization loop
        self._safety_pipeline = None  # 3-pass safety verification

        # New optimization components
        self._retrieval_loop = None  # Self-optimizing retrieval
        self._feedback_learner = None  # User feedback learning
        self._resource_scheduler = None  # Predictive ANE/GPU scheduler
        self._speculative_decoder = None  # Metal speculative decoding
        self._context_allocator = None  # Adaptive context allocator
        self._context_pool = None  # Context pooling for reuse

        # Thread safety
        self._lock = threading.Lock()
        self._initialized = False

        # Metrics
        self._request_count = 0
        self._total_latency_ms = 0.0
        self._cache_hits = 0

        logger.info("AIEngine initialized")

    def _ensure_initialized(self):
        """Lazy initialization of components - simplified, no personalization."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            start = time.perf_counter()
            components_loaded = []

            # 0. Policy Engine - centralized filtering (load first)
            if _POLICY_AVAILABLE:
                try:
                    self._policy_engine = get_policy_engine()
                    components_loaded.append(
                        f"Policy({self._policy_engine.mode.value})"
                    )
                except Exception as e:
                    logger.warning(f"PolicyEngine initialization failed: {e}")

            # 1. RAG Service - use singleton, optional don't block on failure
            try:
                from ..rag import get_rag_service

                self._rag_service = get_rag_service()
                components_loaded.append("RAG")
            except Exception as e:
                logger.warning(f"RAG initialization failed (continuing without): {e}")

            # 2. Safety Guard - lightweight (now integrates with PolicyEngine)
            try:
                from .safety import SafetyGuard

                self._safety_guard = SafetyGuard()
                components_loaded.append("Safety")
            except Exception as e:
                logger.warning(f"SafetyGuard initialization skipped: {e}")

            # 3. Self-Optimizer - dynamic parameter tuning
            try:
                from ...core.optimized.self_optimizer import get_self_optimizer

                self._self_optimizer = get_self_optimizer()
                components_loaded.append("SelfOptimizer")
            except Exception as e:
                logger.warning(f"SelfOptimizer initialization skipped: {e}")

            # 4. Safety Pipeline - 3-pass verification
            try:
                from ..safety import get_safety_pipeline

                self._safety_pipeline = get_safety_pipeline(mode="balanced")
                components_loaded.append("SafetyPipeline")
            except Exception as e:
                logger.warning(f"SafetyPipeline initialization skipped: {e}")

            # 5. Self-Optimizing Retrieval Loop - iterative RAG refinement
            try:
                from ...core.optimized.self_optimizer import (
                    get_feedback_learner,
                    get_retrieval_loop,
                )

                self._retrieval_loop = get_retrieval_loop()
                self._feedback_learner = get_feedback_learner()
                components_loaded.append("RetrievalLoop")
            except Exception as e:
                logger.warning(f"RetrievalLoop initialization skipped: {e}")

            # 6. Predictive Resource Scheduler - ANE/GPU-aware scheduling
            try:
                from ...core.optimized.gpu_pipeline import get_predictive_scheduler

                self._resource_scheduler = get_predictive_scheduler()
                components_loaded.append("ResourceScheduler")
            except Exception as e:
                logger.warning(f"ResourceScheduler initialization skipped: {e}")

            # 7. Speculative Decoding - Metal/ANE acceleration
            try:
                from ...core.optimized.performance import get_speculative_decoder

                self._speculative_decoder = get_speculative_decoder()
                components_loaded.append("SpeculativeDecoder")
            except Exception as e:
                logger.warning(f"SpeculativeDecoder initialization skipped: {e}")

            # 8. Adaptive Context Allocator - dynamic context management
            try:
                self._context_allocator = AdaptiveContextAllocator()
                self._context_pool = ContextPool(max_size=50)
                components_loaded.append("ContextAllocator")
            except Exception as e:
                logger.warning(f"ContextAllocator initialization skipped: {e}")

            # NOTE: Personalization removed - no StudentProfile, no grade_level/subject tracking
            # This speeds up initialization significantly

            self._initialized = True
            elapsed = (time.perf_counter() - start) * 1000

            logger.info(
                f"AIEngine initialized in {elapsed:.0f}ms: {', '.join(components_loaded) if components_loaded else 'core only'}"
            )

    def _get_policy_engine(self):
        """Get the policy engine instance."""
        if self._policy_engine is None and _POLICY_AVAILABLE:
            with contextlib.suppress(Exception):
                self._policy_engine = get_policy_engine()
        return self._policy_engine

    def _should_block_harmful(self, config: "GenerationConfig") -> bool:
        """Determine if harmful content blocking should be applied.

        Checks PolicyEngine mode first, then falls back to config.
        """
        policy = self._get_policy_engine()
        if policy and policy.mode == PolicyMode.UNRESTRICTED:
            # In unrestricted mode, don't block based on educational/curriculum filters
            # But still respect system-level safety if configured
            return bool(policy.config.block_harmful_content)
        return config.block_harmful

    def _get_llm_client(self):
        """Get LLM client - uses shared MLX engine."""
        if self._llm_client is not None:
            return self._llm_client

        with self._lock:
            if self._llm_client is not None:
                return self._llm_client

            # Use shared inference engine (singleton with pre-loaded MLX)
            from ..inference import get_inference_engine

            self._llm_client = get_inference_engine()
            logger.info("Using UnifiedInferenceEngine (MLX) for LLM")

            return self._llm_client

    def _get_collaborator(self):
        """Get or create Model Collaborator for multi-model validation."""
        if self._collaborator is not None:
            return self._collaborator

        with self._lock:
            if self._collaborator is not None:
                return self._collaborator

            try:
                from ..pipeline.model_collaboration import get_model_collaborator

                self._collaborator = get_model_collaborator()
                logger.info(
                    "AIEngine: Model Collaborator loaded for multi-model validation"
                )
            except Exception as e:
                logger.warning(f"Could not load Model Collaborator: {e}")

            return self._collaborator

    def _get_sandbox(self):
        """Get or create code execution sandbox."""
        if self._sandbox is not None:
            return self._sandbox

        with self._lock:
            if self._sandbox is not None:
                return self._sandbox

            try:
                from .sandbox import get_sandbox

                self._sandbox = get_sandbox()
                logger.info("AIEngine: Code sandbox loaded for safe execution")
            except Exception as e:
                logger.warning(f"Could not load sandbox: {e}")

            return self._sandbox

    def _get_self_optimizer(self):
        """Get the self-optimizer for dynamic parameter tuning."""
        if self._self_optimizer is not None:
            return self._self_optimizer

        with self._lock:
            if self._self_optimizer is not None:
                return self._self_optimizer

            try:
                from ...core.optimized.self_optimizer import get_self_optimizer

                self._self_optimizer = get_self_optimizer()
            except Exception as e:
                logger.warning(f"Could not get self-optimizer: {e}")

            return self._self_optimizer

    def _get_safety_pipeline(self):
        """Get the 3-pass safety pipeline."""
        if self._safety_pipeline is not None:
            return self._safety_pipeline

        with self._lock:
            if self._safety_pipeline is not None:
                return self._safety_pipeline

            try:
                from ..safety import get_safety_pipeline

                self._safety_pipeline = get_safety_pipeline(mode="balanced")
            except Exception as e:
                logger.warning(f"Could not get safety pipeline: {e}")

            return self._safety_pipeline

    def _get_retrieval_loop(self):
        """Get the self-optimizing retrieval loop for iterative RAG refinement."""
        if self._retrieval_loop is not None:
            return self._retrieval_loop

        with self._lock:
            if self._retrieval_loop is not None:
                return self._retrieval_loop

            try:
                from ...core.optimized.self_optimizer import get_retrieval_loop

                self._retrieval_loop = get_retrieval_loop()
            except Exception as e:
                logger.warning(f"Could not get retrieval loop: {e}")

            return self._retrieval_loop

    def _get_feedback_learner(self):
        """Get the feedback learner for continuous improvement."""
        if self._feedback_learner is not None:
            return self._feedback_learner

        with self._lock:
            if self._feedback_learner is not None:
                return self._feedback_learner

            try:
                from ...core.optimized.self_optimizer import get_feedback_learner

                self._feedback_learner = get_feedback_learner()
            except Exception as e:
                logger.warning(f"Could not get feedback learner: {e}")

            return self._feedback_learner

    def _get_resource_scheduler(self):
        """Get the predictive resource scheduler for ANE/GPU-aware routing."""
        if self._resource_scheduler is not None:
            return self._resource_scheduler

        with self._lock:
            if self._resource_scheduler is not None:
                return self._resource_scheduler

            try:
                from ...core.optimized.gpu_pipeline import get_predictive_scheduler

                self._resource_scheduler = get_predictive_scheduler()
            except Exception as e:
                logger.warning(f"Could not get resource scheduler: {e}")

            return self._resource_scheduler

    def _get_speculative_decoder(self):
        """Get the Metal speculative decoder for accelerated generation."""
        if self._speculative_decoder is not None:
            return self._speculative_decoder

        with self._lock:
            if self._speculative_decoder is not None:
                return self._speculative_decoder

            try:
                from ...core.optimized.performance import get_speculative_decoder

                self._speculative_decoder = get_speculative_decoder()
            except Exception as e:
                logger.warning(f"Could not get speculative decoder: {e}")

            return self._speculative_decoder

    def _get_context_allocator(self):
        """Get the adaptive context allocator for dynamic context management."""
        if self._context_allocator is not None:
            return self._context_allocator

        with self._lock:
            if self._context_allocator is not None:
                return self._context_allocator

            try:
                self._context_allocator = AdaptiveContextAllocator()
            except Exception as e:
                logger.warning(f"Could not create context allocator: {e}")

            return self._context_allocator

    def _apply_optimized_params(self, config: "GenerationConfig") -> "GenerationConfig":
        """Apply self-optimized parameters to generation config."""
        optimizer = self._get_self_optimizer()
        if not optimizer:
            return config

        params = optimizer.get_current_params()

        # Default values for comparison (use int comparison for precision)
        DEFAULT_TEMPERATURE = 0.7
        DEFAULT_MAX_TOKENS = 512
        DEFAULT_TOP_P = 0.9

        # Apply optimized parameters (only if using defaults)
        if params.temperature and abs(config.temperature - DEFAULT_TEMPERATURE) < 0.001:
            config.temperature = params.temperature
        if params.max_tokens and config.max_tokens == DEFAULT_MAX_TOKENS:
            config.max_tokens = params.max_tokens
        if params.top_p and abs(config.top_p - DEFAULT_TOP_P) < 0.001:
            config.top_p = params.top_p

        return config

    async def _verify_response_safety(
        self,
        query: str,
        response: str,
        context_chunks: list,
    ) -> tuple:
        """Verify response through 3-pass safety pipeline.

        Returns:
            (verified_response, is_safe, safety_result)
        """
        pipeline = self._get_safety_pipeline()
        if not pipeline:
            return response, True, None

        try:
            result = await pipeline.verify(
                query=query,
                response=response,
                context=[
                    c.text if hasattr(c, "text") else str(c) for c in context_chunks
                ],
            )

            if result.overall_safe:
                return response, True, result
            elif result.filtered_response:
                # Use filtered version
                return result.filtered_response, True, result
            else:
                # Response blocked
                logger.warning(
                    f"Response blocked by safety pipeline: {result.rejection_reason}"
                )
                return (
                    "I apologize, but I cannot provide that response. "
                    "Please try rephrasing your question.",
                    False,
                    result,
                )
        except Exception as e:
            logger.warning(f"Safety pipeline check failed: {e}")
            return response, True, None

    def _record_optimization_metrics(
        self,
        result: "GenerationResult",
        sources: list,
        cache_hit: bool = False,
    ) -> None:
        """Record metrics for self-optimization."""
        optimizer = self._get_self_optimizer()
        if not optimizer:
            return

        try:
            from ...core.optimized.self_optimizer import OptimizationMetrics

            metrics = OptimizationMetrics(
                retrieval_precision=len(sources) / 5.0 if sources else 0.0,
                rerank_improvement=0.1,  # Placeholder
                generation_quality=result.confidence,
                semantic_coherence=result.confidence,
                generation_latency=result.latency_ms,
                total_latency=result.latency_ms,
                input_tokens=result.tokens_prompt,
                output_tokens=result.tokens_completion,
                chunks_retrieved=len(sources),
                cache_hit_rate=1.0 if cache_hit else 0.0,
            )

            optimizer.record_metrics(metrics)
        except Exception as e:
            logger.debug(f"Failed to record optimization metrics: {e}")

    def _get_evaluator(self):
        """Get or create Semantic Accuracy Evaluator for validation."""
        if not hasattr(self, "_evaluator"):
            self._evaluator = None

        if self._evaluator is not None:
            return self._evaluator

        with self._lock:
            if hasattr(self, "_evaluator") and self._evaluator is not None:
                return self._evaluator

            try:
                from ..evaluation import get_semantic_evaluator

                self._evaluator = get_semantic_evaluator()
                logger.info("AIEngine: Semantic Evaluator loaded for validation")
            except Exception as e:
                logger.warning(f"Could not load Semantic Evaluator: {e}")

            return self._evaluator

    async def chat(
        self,
        message: str,
        conversation_id: str | None = None,
        user_id: str | None = None,
        config: GenerationConfig | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> FormattedResponse:
        """
        Process a chat message and return formatted response.

        Args:
            message: User's message
            conversation_id: Optional conversation ID for context
            user_id: Optional user ID
            config: Generation configuration
            context_data: Additional context (e.g., grade level, language)

        Returns:
            FormattedResponse with content, metadata, and sources
        """
        self._ensure_initialized()
        config = config or GenerationConfig()
        start_time = time.perf_counter()

        # FAST PATH: Check for simple arithmetic that can be computed directly
        fast_result = self._try_fast_arithmetic(message)
        if fast_result:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return self.formatter.format_response(
                content=fast_result,
                query=message,
                sources=[],
                tokens_prompt=0,
                tokens_completion=_count_tokens(fast_result),
                latency_ms=latency_ms,
                model_id="fast-compute",
                confidence=1.0,
            )

        # Get or create conversation context
        if conversation_id:
            context = self.context_manager.get_or_create(conversation_id, user_id)
        else:
            context = self.context_manager.get_or_create(str(uuid_module.uuid4()), user_id)

        # Add user message to context
        context.add_message(ContextRole.USER, message)

        # NOTE: Personalization removed - no grade_level, subject, learning_style
        # All students get the same quality responses

        # Detect intent and route to appropriate handler
        intent = self.formatter.detect_intent(message)

        # Classify compute class and get immutable budget
        compute_class = classify_compute(message, intent)
        budget = get_budget(compute_class)
        logger.debug(
            "Compute budget: class=%s max_tokens=%d rag=%s safety=%s tier_ceil=%s",
            budget.compute_class.value, budget.max_tokens,
            budget.use_rag, budget.run_safety, budget.model_tier_ceiling.value,
        )

        # Apply budget ceilings to config
        config.max_tokens = min(config.max_tokens, budget.max_tokens)
        if budget.use_rag is not None:
            config.use_rag = budget.use_rag

        # Build prompt with conversation history
        history = context.get_context_messages(max_messages=10)

        # Generate response
        try:
            result = await self._generate_response(
                message=message,
                history=history,
                intent=intent,
                config=config,
                context_data=context_data,  # Pass through context (subject, language, etc.)
                budget=budget,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result = GenerationResult(
                content=self._get_fallback_response(message, intent),
                tokens_prompt=0,
                tokens_completion=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                model_id="fallback",
                confidence=0.5,
            )

        # Add assistant response to context
        context.add_message(ContextRole.ASSISTANT, result.content)
        self.context_manager.save(context)

        # NOTE: Review queue removed - no personalization tracking

        # Format the response
        formatted = self.formatter.format_response(
            content=result.content,
            query=message,
            sources=result.sources,
            tokens_prompt=result.tokens_prompt,
            tokens_completion=result.tokens_completion,
            latency_ms=result.latency_ms,
            model_id=result.model_id,
            confidence=result.confidence,
        )

        # Update metrics
        self._request_count += 1
        self._total_latency_ms += result.latency_ms
        if result.cached:
            self._cache_hits += 1

        return formatted

    async def chat_stream(
        self,
        message: str,
        conversation_id: str | None = None,
        user_id: str | None = None,
        config: GenerationConfig | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response token by token.

        Yields SSE-formatted events for real-time streaming.
        """
        self._ensure_initialized()
        config = config or GenerationConfig(stream=True)
        start_time = time.perf_counter()

        # FAST PATH: Check for simple arithmetic that can be computed directly
        fast_result = self._try_fast_arithmetic(message)
        if fast_result:
            yield fast_result
            return

        # Get or create conversation context
        conv_id = conversation_id or str(uuid_module.uuid4())
        context = self.context_manager.get_or_create(conv_id, user_id)
        context.add_message(ContextRole.USER, message)

        # Build prompt
        history = context.get_context_messages(max_messages=10)
        intent = self.formatter.detect_intent(message)

        # Classify compute and apply budget ceilings
        compute_class = classify_compute(message, intent)
        budget = get_budget(compute_class)
        config.max_tokens = min(config.max_tokens, budget.max_tokens)
        if budget.use_rag is not None:
            config.use_rag = budget.use_rag

        # Stream response
        full_response = ""

        try:
            async for token in self._stream_generate(
                message=message,
                history=history,
                intent=intent,
                config=config,
                context_data=context_data,
                budget=budget,
            ):
                full_response += token
                yield self._format_sse_chunk(token)

            # Complete event
            latency_ms = (time.perf_counter() - start_time) * 1000
            yield self._format_sse_complete(full_response, latency_ms)

            # Save to context
            context.add_message(ContextRole.ASSISTANT, full_response)
            self.context_manager.save(context)

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield self._format_sse_error(str(e))

    def _allocate_compute_resources(
        self, message: str, intent: Intent, config: GenerationConfig
    ) -> Any:
        """Allocate context tokens and predict resource requirements.

        Returns the context allocation object (or None).
        """
        context_allocator = self._get_context_allocator()
        context_allocation = None
        if context_allocator:
            try:
                context_allocation = context_allocator.allocate(message, intent.value)
                logger.debug(
                    f"Context allocation: {context_allocation.total_tokens} tokens, priority={context_allocation.priority}"
                )
            except Exception as e:
                logger.debug(f"Context allocation failed: {e}")

        resource_scheduler = self._get_resource_scheduler()
        if resource_scheduler:
            try:
                prediction = resource_scheduler.predict_resources(
                    queue_length=1,
                    avg_tokens=config.max_tokens,
                    current_memory_pressure=0.5,
                )
                if prediction and hasattr(prediction, "optimal_batch_size"):
                    logger.debug(
                        f"Resource prediction: batch_size={prediction.optimal_batch_size}"
                    )
            except Exception as e:
                logger.debug(f"Resource scheduling failed: {e}")

        return context_allocation

    async def _retrieve_rag_sources(
        self, message: str
    ) -> tuple[list, str, bool, int]:
        """Retrieve RAG sources using self-optimizing loop with standard fallback.

        Returns: (sources, augmented_context, rag_empty, retrieval_iterations)
        """
        sources: list = []
        augmented_context = ""
        rag_empty = False
        retrieval_iterations = 0

        retrieval_loop = self._get_retrieval_loop()

        if retrieval_loop:
            try:
                def search_fn(query: str, top_k: int = 5):
                    return self._rag_service.search(query, top_k=top_k, rerank=True)

                loop_result = await retrieval_loop.retrieve_with_refinement(
                    query=message,
                    search_fn=search_fn,
                    min_quality=0.7,
                    max_iterations=3,
                )

                if loop_result and loop_result.chunks:
                    retrieval_iterations = loop_result.iterations
                    sources = [
                        SourceReference(
                            source_id=r.chunk_id
                            if hasattr(r, "chunk_id")
                            else str(i),
                            source_type="document",
                            title=r.metadata.get("title", "Document")
                            if hasattr(r, "metadata")
                            else "Document",
                            location=r.metadata.get("location")
                            if hasattr(r, "metadata")
                            else None,
                            confidence=r.score if hasattr(r, "score") else 0.8,
                            quote=r.text[:200]
                            if hasattr(r, "text") and len(r.text) > 200
                            else (r.text if hasattr(r, "text") else str(r)),
                            is_inferred=(r.score if hasattr(r, "score") else 0.8)
                            < 0.8,
                        )
                        for i, r in enumerate(loop_result.chunks[:3])
                    ]
                    augmented_context = "\n\n".join(
                        [
                            r.text if hasattr(r, "text") else str(r)
                            for r in loop_result.chunks[:3]
                        ]
                    )
                    logger.debug(
                        f"Retrieval loop completed in {retrieval_iterations} iterations"
                    )
                else:
                    rag_empty = True
            except Exception as e:
                logger.warning(
                    f"Self-optimizing retrieval failed, falling back: {e}"
                )
                retrieval_loop = None  # Fall back to standard retrieval

        # Fallback to standard RAG if loop not available or failed
        if not retrieval_loop or (not sources and not rag_empty):
            try:
                rag_result = self._rag_service.search(message, top_k=5, rerank=True)
                if rag_result:
                    sources = [
                        SourceReference(
                            source_id=r.chunk_id,
                            source_type="document",
                            title=r.metadata.get("title", "Document"),
                            location=r.metadata.get("location"),
                            confidence=r.score,
                            quote=r.text[:200] if len(r.text) > 200 else r.text,
                            is_inferred=r.score < 0.8,
                        )
                        for r in rag_result[:3]
                    ]
                    augmented_context = "\n\n".join(
                        [r.text for r in rag_result[:3]]
                    )
                else:
                    rag_empty = True
                    logger.info(f"RAG returned no results for: {message[:50]}...")
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
                rag_empty = True

        return sources, augmented_context, rag_empty, retrieval_iterations

    def _calculate_response_confidence(
        self,
        response_text: str,
        sources: list,
        rag_attempted: bool,
        rag_empty: bool,
        is_unrestricted: bool,
    ) -> tuple[float, int]:
        """Calculate confidence score and uncertainty count for a response.

        Returns: (confidence, uncertainty_count)
        """
        confidence = 0.7  # Base confidence

        if sources:
            avg_source_score = sum(s.confidence for s in sources) / len(sources)
            confidence = 0.75 + (avg_source_score * 0.2)  # 0.75-0.95 range
        elif rag_attempted and rag_empty:
            confidence = 0.6

        # Common uncertainty phrases that reduce confidence
        uncertainty_phrases = [
            "i'm not sure",
            "i don't know",
            "i'm not certain",
            "might be",
            "could be",
            "possibly",
            "perhaps",
            "i think",
            "i believe",
            "it seems",
            "general knowledge",
        ]
        if not is_unrestricted:
            uncertainty_phrases.extend(
                ["verify with your teacher", "check your textbook"]
            )

        response_lower = response_text.lower()
        uncertainty_count = sum(
            1 for phrase in uncertainty_phrases if phrase in response_lower
        )
        if uncertainty_count >= 2:
            confidence = min(confidence, 0.5)
        elif uncertainty_count == 1:
            confidence = min(confidence, 0.6)

        # Specific factual claims without sources = lower confidence
        factual_indicators = [
            "in 1", "in 2", "was born", "died in", "founded in",
            "discovered in", "invented in", "established in",
            "population of", "capital of", "president of", "prime minister of",
        ]
        if not sources and any(ind in response_lower for ind in factual_indicators):
            confidence = min(confidence, 0.55)

        return confidence, uncertainty_count

    async def _generate_response(
        self,
        message: str,
        history: list[dict[str, str]],
        intent: Intent,
        config: GenerationConfig,
        context_data: dict[str, Any] | None = None,
        budget: ComputeBudget | None = None,
    ) -> GenerationResult:
        """Generate a complete response with self-optimization and safety verification."""
        start_time = time.perf_counter()

        # Apply self-optimized parameters
        config = self._apply_optimized_params(config)

        # Route to appropriate model — enforce budget tier ceiling
        task_type = self._intent_to_task_type(intent)
        routing = self.router.route(
            message,
            task_type,
            max_latency_ms=budget.max_latency_ms if budget else None,
            model_tier_ceiling=budget.model_tier_ceiling if budget else None,
        )

        # Compute token ceiling: router estimate capped by budget
        dynamic_max_tokens = routing.estimated_max_tokens
        if budget:
            # Budget is a hard ceiling — over-computation is a failure
            config.max_tokens = min(budget.max_tokens, dynamic_max_tokens)
        elif config.max_tokens < dynamic_max_tokens:
            config.max_tokens = dynamic_max_tokens

        # Check if RAG is needed - respect config.use_rag if explicitly set
        if config.use_rag is not None:
            use_rag = config.use_rag
        else:
            use_rag = self._should_use_rag(message, intent)

        sources: list[dict[str, Any]] = []
        augmented_context = ""
        rag_attempted = False
        rag_empty = False
        retrieval_iterations = 0

        # Allocate compute resources (context tokens + resource scheduling)
        context_allocation = self._allocate_compute_resources(message, intent, config)

        if use_rag and self._rag_service:
            rag_attempted = True
            sources, augmented_context, rag_empty, retrieval_iterations = (
                await self._retrieve_rag_sources(message)
            )

        # Build prompt with RAG context if available
        # Use compute budget to determine prompt complexity
        is_simple = budget.compute_class == ComputeClass.TRIVIAL if budget else self._is_simple_query(message, intent)
        system_prompt = self._build_system_prompt(
            intent, context_data, is_simple_query=is_simple
        )

        # Check policy mode for context modification
        policy = self._get_policy_engine()
        is_unrestricted = policy and policy.mode == PolicyMode.UNRESTRICTED

        # Modify system prompt based on RAG results
        if augmented_context:
            # We have verified sources - instruct model to use them
            system_prompt += (
                "\n\nVERIFIED REFERENCE DOCUMENTS PROVIDED:\n"
                "You MUST base your answer primarily on the provided context below. "
                "These are trusted sources. Cite specific information from them.\n"
            )
        elif rag_attempted and rag_empty and not is_unrestricted:
            # RAG was tried but no documents found - make model more cautious (only in restricted mode)
            system_prompt += (
                "\n\nIMPORTANT: No verified reference documents are available for this query.\n"
                "You should:\n"
                "1. Clearly state that you're providing general knowledge, not verified information\n"
                "2. Be extra cautious with specific facts, dates, and numbers\n"
                "3. Recommend the student verify important facts with their textbook or teacher\n"
                "4. Focus on explaining concepts and reasoning rather than specific facts\n"
            )

        prompt = self._build_prompt(message, history, augmented_context, system_prompt)

        # Generate with LLM
        llm = self._get_llm_client()

        try:
            response_text = await self._run_generation(llm, prompt, config)

            # SafetyGuard: proactively scrubs code-execution patterns
            # (os.system/subprocess/eval/exec) from responses.  This is NOT
            # duplicated by SafetyPipeline — Pass 3 covers hate-speech/PII/
            # hallucination, not code injection.  Both layers are required.
            if self._should_block_harmful(config) and self._safety_guard:
                response_text = self._safety_guard.filter_response(response_text)

            # Calculate confidence based on multiple factors
            confidence, uncertainty_count = self._calculate_response_confidence(
                response_text, sources, rag_attempted, rag_empty, is_unrestricted
            )

            # 3-Pass Safety Pipeline Verification
            # Safety is never optional — even TRIVIAL queries run through it.
            # The budget.run_safety flag is preserved for future use but
            # currently always True (see quality audit: PII/toxicity can
            # appear in any LLM response regardless of query complexity).
            rag_chunks = list(sources) if sources else []
            (
                verified_response,
                is_safe,
                safety_result,
            ) = await self._verify_response_safety(
                query=message,
                response=response_text,
                context_chunks=rag_chunks,
            )

            if not is_safe:
                response_text = verified_response
                confidence = min(confidence, 0.4)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Get speculative decoding stats if available
            spec_decoder = self._get_speculative_decoder()
            spec_stats = spec_decoder.get_stats() if spec_decoder else None

            spec_decoding_meta = None
            if spec_stats:
                spec_decoding_meta = {
                    "used": spec_stats.total_generations > 0,
                    "acceptance_rate": spec_stats.acceptance_rate,
                    "speedup": spec_stats.avg_speedup,
                }

            alloc_tokens = context_allocation.total_tokens if context_allocation else None
            safety_level = safety_result.overall_level.value if safety_result else "unknown"

            result = GenerationResult(
                content=response_text,
                tokens_prompt=_count_tokens(prompt),
                tokens_completion=_count_tokens(response_text),
                latency_ms=latency_ms,
                model_id=routing.model_id,
                sources=sources,
                confidence=confidence,
                metadata={
                    "compute_class": budget.compute_class.value if budget else "unclassified",
                    "compute_budget_max_tokens": budget.max_tokens if budget else None,
                    "rag_attempted": rag_attempted,
                    "rag_found_sources": len(sources) > 0,
                    "retrieval_iterations": retrieval_iterations,
                    "context_allocation": alloc_tokens,
                    "uncertainty_detected": uncertainty_count > 0,
                    "safety_verified": safety_result is not None,
                    "safety_level": safety_level,
                    "speculative_decoding": spec_decoding_meta,
                },
            )

            # Record metrics for self-optimization
            self._record_optimization_metrics(result, sources, cache_hit=False)

            return result

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def _fetch_stream_rag_context(self, message: str, use_rag: bool) -> tuple[str, bool]:
        """Fetch RAG context for streaming. Returns (augmented_context, rag_empty)."""
        if not (use_rag and self._rag_service):
            return "", False
        try:
            rag_result = self._rag_service.search(message, top_k=3, rerank=True)
            if rag_result:
                return "\n\n".join([r.text for r in rag_result[:3]]), False
            return "", True
        except Exception as e:
            logger.warning(f"RAG search failed in streaming: {e}")
            return "", True

    async def _stream_generate(
        self,
        message: str,
        history: list[dict[str, str]],
        intent: Intent,
        config: GenerationConfig,
        context_data: dict[str, Any] | None = None,
        budget: ComputeBudget | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM with RAG support."""
        # Get dynamic token allocation for streaming too
        task_type = self._intent_to_task_type(intent)
        routing = self.router.route(
            message,
            task_type,
            max_latency_ms=budget.max_latency_ms if budget else None,
            model_tier_ceiling=budget.model_tier_ceiling if budget else None,
        )

        # Token ceiling: router estimate capped by budget
        dynamic_max_tokens = routing.estimated_max_tokens
        if budget:
            config.max_tokens = min(budget.max_tokens, dynamic_max_tokens)
        elif config.max_tokens < dynamic_max_tokens:
            config.max_tokens = dynamic_max_tokens

        # Check if RAG is needed - respect config.use_rag if explicitly set
        use_rag = config.use_rag if config.use_rag is not None else self._should_use_rag(message, intent)

        augmented_context, rag_empty = self._fetch_stream_rag_context(message, use_rag)

        # Build system prompt with RAG awareness
        is_simple = budget.compute_class == ComputeClass.TRIVIAL if budget else self._is_simple_query(message, intent)
        system_prompt = self._build_system_prompt(
            intent, context_data, is_simple_query=is_simple
        )

        if augmented_context:
            system_prompt += (
                "\n\nVERIFIED REFERENCE DOCUMENTS PROVIDED:\n"
                "Base your answer on the provided context. Cite specific information.\n"
            )
        elif use_rag and rag_empty:
            system_prompt += (
                "\n\nNo verified sources available. Be cautious with specific facts.\n"
            )

        prompt = self._build_prompt(message, history, augmented_context, system_prompt)

        llm = self._get_llm_client()

        # Check if LLM supports streaming
        if hasattr(llm, "generate_stream"):
            async for token in llm.generate_stream(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            ):
                yield token
        else:
            # Fallback: generate full response and simulate streaming
            response = await self._run_generation(llm, prompt, config)
            # Chunk the response for pseudo-streaming
            chunk_size = 4
            for i in range(0, len(response), chunk_size):
                yield response[i : i + chunk_size]
                await asyncio.sleep(0.01)  # Small delay for UX

    async def _run_generation(
        self,
        llm: Any,
        prompt: str,
        config: GenerationConfig,
        use_speculative: bool = True,
    ) -> str:
        """Run LLM generation with timeout and optional speculative decoding.

        Args:
            llm: The LLM client
            prompt: The prompt to generate from
            config: Generation configuration
            use_speculative: Whether to try speculative decoding (Metal/ANE)
        """
        # Try speculative decoding for Metal/ANE acceleration
        speculative_decoder = (
            self._get_speculative_decoder() if use_speculative else None
        )

        if speculative_decoder and speculative_decoder.is_available():
            try:
                # Use speculative decoding for potentially faster generation
                result = await speculative_decoder.generate_speculative(
                    prompt=prompt,
                    target_model=llm,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
                if result:
                    logger.debug(
                        f"Speculative decoding: {speculative_decoder.get_stats()}"
                    )
                    return str(result)
            except Exception as e:
                logger.debug(f"Speculative decoding failed, falling back: {e}")

        try:
            # Prefer generate_async for backward compatibility with max_tokens param
            if hasattr(llm, "generate_async") and asyncio.iscoroutinefunction(
                llm.generate_async
            ):
                result = await asyncio.wait_for(
                    llm.generate_async(
                        prompt,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                    ),
                    timeout=config.timeout_seconds,
                )
            elif hasattr(llm, "generate") and asyncio.iscoroutinefunction(llm.generate):
                # Fallback to generate with config object
                from ..inference.unified_engine import GenerationConfig as InferenceGenConfig

                inf_config = InferenceGenConfig(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
                result = await asyncio.wait_for(
                    llm.generate(prompt, inf_config), timeout=config.timeout_seconds
                )
            elif hasattr(llm, "generate"):
                # Run sync method in thread pool
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: llm.generate(
                        prompt,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                    ),
                )
            else:
                # Direct call fallback
                result = str(llm(prompt))

            return str(result)

        except TimeoutError:
            logger.warning(f"Generation timed out after {config.timeout_seconds}s")
            return "I apologize, but the response took too long. Please try a simpler question."

    def _build_system_prompt(
        self,
        intent: Intent,
        context_data: dict[str, Any] | None = None,
        is_simple_query: bool = False,
    ) -> str:
        """Build system prompt based on intent and policy mode.

        In UNRESTRICTED mode: Acts as a general-purpose AI assistant
        In RESTRICTED mode: Education-focused with verification requirements

        Args:
            intent: Detected user intent
            context_data: Additional context (subject, language, etc.)
            is_simple_query: If True, use minimal prompt for speed
        """
        policy = self._get_policy_engine()
        is_unrestricted = policy and policy.mode == PolicyMode.UNRESTRICTED

        # FAST PATH: Minimal prompt for simple queries
        if is_simple_query:
            return (
                "You are a helpful AI assistant. "
                "Answer concisely and accurately. "
                "For math, just compute and show the result."
            )

        if is_unrestricted:
            # General-purpose AI assistant prompt (like ChatGPT/Claude)
            base_prompt = (
                "You are a helpful, knowledgeable AI assistant. "
                "You can help with any topic the user asks about.\n\n"
                "GUIDELINES:\n"
                "- Be helpful, accurate, and thorough in your responses\n"
                "- Use clear, well-structured explanations\n"
                "- Admit when you're uncertain about something\n"
                "- For code, provide complete, working examples\n"
                "- For math, use LaTeX: inline $x^2$ or block $$\\frac{a}{b}$$\n"
                "- Use markdown for formatting (headers, lists, code blocks)\n\n"
            )
        else:
            # Education-focused prompt with verification requirements
            base_prompt = (
                "You are ShikshaSetu, an intelligent AI assistant. "
                "You MUST provide accurate, factual, and verified information only.\n\n"
                "CRITICAL RULES:\n"
                "1. NEVER make up facts, dates, names, or statistics\n"
                "2. If you don't know something with certainty, say 'I'm not certain about this'\n"
                "3. For factual questions, only state what is definitively true\n"
                "4. If context/documents are provided, base your answer on them\n"
                "5. Distinguish between established facts and your reasoning\n"
                "6. For controversial topics, present multiple perspectives\n\n"
                "FORMATTING:\n"
                "- Use markdown for structure (headers, lists, bold)\n"
                "- Use LaTeX for math: inline $x^2$ or block $$\\frac{a}{b}$$\n"
                "- Show step-by-step solutions for problems\n"
                "- Use code blocks with language tags for code\n\n"
            )

        context_data = context_data or {}
        subject = context_data.get("subject")

        # Add subject context if provided
        if subject and subject != "General":
            base_prompt += f"CONTEXT: The user is asking about {subject}.\n\n"

        intent_prompts = {
            Intent.CODE_REQUEST: (
                "When providing code:\n"
                "1. Include complete, runnable examples\n"
                "2. Add clear comments explaining the logic\n"
                "3. Suggest tests when appropriate\n"
                "4. Explain the code after the code block\n"
            ),
            Intent.EXPLANATION: (
                "Explain concepts clearly. "
                "Use simple language and examples. "
                "Break down complex topics into digestible parts. "
                "For math topics, include worked examples with LaTeX. "
            ),
            Intent.COMPARISON: (
                "When comparing:\n"
                "1. Create a clear table if comparing multiple items\n"
                "2. List pros and cons for each option\n"
                "3. Provide a recommendation when asked\n"
            ),
            Intent.SMALL_TALK: (
                "Be friendly and conversational while remaining helpful. "
            ),
        }

        return base_prompt + intent_prompts.get(intent, "")

    def _build_prompt(
        self,
        message: str,
        history: list[dict[str, str]],
        augmented_context: str,
        system_prompt: str,
    ) -> str:
        """Build the full prompt for the LLM using ChatML format.

        Uses <|im_start|>/<|im_end|> tokens expected by Qwen2.5-Instruct.
        This format is passed through as pre-formatted by the inference backends.
        """
        parts = []

        # System prompt with RAG context embedded
        full_system = system_prompt
        if augmented_context:
            full_system += f"\n\nREFERENCE CONTEXT:\n{augmented_context}"

        parts.append(f"<|im_start|>system\n{full_system}<|im_end|>")

        # Conversation history (last 3 turns = 6 messages)
        for msg in history[-6:]:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"\n<|im_start|>{role}\n{content}<|im_end|>")

        # Current message (if not already in history)
        if not history or history[-1]["content"] != message:
            parts.append(f"\n<|im_start|>user\n{message}<|im_end|>")

        # Generation prompt
        parts.append("\n<|im_start|>assistant\n")

        return "".join(parts)

    def _intent_to_task_type(self, intent: Intent) -> TaskType:
        """Map intent to task type for routing."""
        mapping = {
            Intent.CODE_REQUEST: TaskType.CODE,
            Intent.EXPLANATION: TaskType.REASONING,
            Intent.COMPARISON: TaskType.REASONING,
            Intent.TRANSLATION: TaskType.TRANSLATION,
            Intent.SIMPLIFICATION: TaskType.SUMMARIZATION,
            Intent.QUESTION: TaskType.CHAT,
            Intent.SMALL_TALK: TaskType.CHAT,
            Intent.TASK: TaskType.REASONING,
        }
        return mapping.get(intent, TaskType.CHAT)

    def _should_use_rag(self, message: str, intent: Optional["Intent"] = None) -> bool:
        """Determine if RAG should be used for this query.

        RAG adds latency, so only use when likely to find relevant documents.
        For general knowledge questions, the LLM is sufficient.

        Args:
            message: The user's message
            intent: Optional detected intent (reserved for future use)
        """
        # Suppress unused argument warning - intent reserved for future enhancements
        _ = intent

        # Only use RAG if the service is available
        if not self._rag_service:
            return False

        message_lower = message.lower()

        # Skip RAG for simple greetings and meta-questions
        skip_patterns = [
            "hello",
            "hi ",
            "hey",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "how are you",
            "what can you do",
            "help me",
            "who are you",
        ]
        if any(pat in message_lower for pat in skip_patterns):
            return False

        # Skip RAG for very short queries (likely not educational)
        if len(message) < 15:
            return False

        # Keywords that REQUIRE document lookup
        doc_required_keywords = [
            "document",
            "file",
            "uploaded",
            "according to",
            "from the",
            "based on the",
            "in my notes",
            "curriculum",
            "ncert",
            "textbook",
            "chapter",
            "lesson",
            "syllabus",
        ]
        # Return condition directly
        return any(kw in message_lower for kw in doc_required_keywords)

    def _should_validate(self, message: str, intent: Intent) -> bool:
        """Determine if multi-model validation should be used.

        Validation is used for:
        - Factual/educational queries that need accuracy
        - Longer responses that could contain errors
        - Queries without RAG sources (higher hallucination risk)
        """
        # Always validate educational content
        educational_keywords = [
            "explain",
            "what is",
            "define",
            "describe",
            "how does",
            "calculate",
            "solve",
            "formula",
            "equation",
            "theorem",
            "history",
            "science",
            "math",
            "physics",
            "chemistry",
            "biology",
            "geography",
            "economics",
            "concept",
        ]
        message_lower = message.lower()
        if any(kw in message_lower for kw in educational_keywords):
            return True

        # Validate for question/explanation intents
        return intent in [Intent.QUESTION, Intent.EXPLANATION, Intent.COMPARISON]

    def _get_fallback_response(self, _message: str, _intent: Intent) -> str:
        """Generate fallback response when LLM fails.

        Keep it simple - no hardcoded responses per intent.
        The LLM should handle everything when available.
        Args:
            _message: User message (reserved for future context-aware fallbacks)
            _intent: Detected intent (reserved for future intent-specific fallbacks)
        """
        return "I'm having trouble processing your request right now. Please try again in a moment."

    def _try_fast_arithmetic(self, message: str) -> str | None:
        """
        Try to compute simple arithmetic directly without LLM.

        This provides instant responses for queries like:
        - "2+2"
        - "10 * 5"
        - "100/4"
        - "what is 25 + 75"

        Returns:
            Computed result as string, or None if not a simple arithmetic query
        """
        import ast
        import operator
        import re

        msg = message.lower().strip()

        # Remove common prefixes
        prefixes = [
            "what is ",
            "what's ",
            "calculate ",
            "compute ",
            "solve ",
            "evaluate ",
            "how much is ",
            "=",
        ]
        for prefix in prefixes:
            if msg.startswith(prefix):
                msg = msg[len(prefix) :].strip()
                break

        # Remove trailing question mark
        msg = msg.rstrip("?").strip()

        # Convert word-based math to symbols
        word_to_op = [
            (r"\s+plus\s+", "+"),
            (r"\s+minus\s+", "-"),
            (r"\s+times\s+", "*"),
            (r"\s+multiplied\s+by\s+", "*"),
            (r"\s+divided\s+by\s+", "/"),
            (r"\s+over\s+", "/"),
        ]
        for pattern, op in word_to_op:
            msg = re.sub(pattern, op, msg)

        # Only attempt if it looks like a simple expression
        # Allowed: digits, operators, spaces, parentheses, decimal points
        if not re.match(r"^[\d\s\+\-\*\/\^\(\)\.]+$", msg):
            return None

        # Safety: limit length and depth
        if len(msg) > 50 or msg.count("(") > 3:
            return None

        try:
            # Replace ^ with ** for exponentiation
            expr = msg.replace("^", "**")

            # Safe AST-based evaluation (no eval())
            result = self._safe_math_eval(expr)
            if result is None:
                return None

            # Format result
            if isinstance(result, float):
                result = int(result) if result == int(result) else round(result, 6)

            return f"**{message.strip()}** = **{result}**"

        except (SyntaxError, TypeError, ZeroDivisionError, NameError, ValueError):
            return None
        except Exception:
            return None

    @staticmethod
    def _eval_math_node(node, ops: dict):
        """Recursively evaluate a single AST math node.

        Returns the numeric result, or None for unsupported nodes.
        """
        import ast

        if isinstance(node, ast.Expression):
            return AIEngine._eval_math_node(node.body, ops)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left = AIEngine._eval_math_node(node.left, ops)
            right = AIEngine._eval_math_node(node.right, ops)
            if left is None or right is None:
                return None
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            val = AIEngine._eval_math_node(node.operand, ops)
            return ops[type(node.op)](val) if val is not None else None
        return None  # Unsupported node type

    @staticmethod
    def _safe_math_eval(expr: str):
        """Safely evaluate a math expression using AST parsing.

        Only allows: numbers, +, -, *, /, **, unary -, parentheses.
        No function calls, no attribute access, no imports.
        """
        import ast
        import operator

        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        try:
            tree = ast.parse(expr.strip(), mode='eval')
            return AIEngine._eval_math_node(tree, ops)
        except (SyntaxError, ValueError):
            return None

    def _is_simple_query(self, message: str, intent: Intent) -> bool:
        """
        Detect if query is simple enough to use fast path settings.

        Simple queries get:
        - Reduced max_tokens (256 instead of 512+)
        - RAG disabled
        - Minimal system prompt

        Returns:
            True if query should use fast path
        """
        msg_lower = message.lower().strip()

        # Very short messages are simple
        if len(message) < 30:
            return True

        # Greetings and small talk
        greetings = ["hello", "hi", "hey", "thanks", "thank you", "bye", "ok", "okay"]
        if any(msg_lower.startswith(g) or msg_lower == g for g in greetings):
            return True

        # Simple questions (yes/no, definitions)
        if msg_lower.startswith(("is ", "are ", "can ", "do ", "does ", "will ")):
            if len(message) < 50:
                return True

        # Intent-based detection
        return intent in [Intent.SMALL_TALK]

    def _format_sse_chunk(self, text: str) -> str:
        """Format text chunk as SSE event."""
        return f"data: {json.dumps({'type': 'chunk', 'data': {'text': text}})}\n\n"

    def _format_sse_complete(self, text: str, latency_ms: float) -> str:
        """Format completion as SSE event."""
        return f"data: {json.dumps({'type': 'complete', 'data': {'message_id': str(uuid_module.uuid4()), 'text': text, 'latency_ms': latency_ms}})}\n\n"

    def _format_sse_error(self, error: str) -> str:
        """Format error as SSE event."""
        return f"data: {json.dumps({'type': 'error', 'data': {'error': error}})}\n\n"

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        # Get optimization component stats
        spec_decoder = self._get_speculative_decoder()
        spec_stats = spec_decoder.get_stats() if spec_decoder else None

        resource_scheduler = self._get_resource_scheduler()
        scheduler_stats = (
            resource_scheduler.get_stats()
            if resource_scheduler and hasattr(resource_scheduler, "get_stats")
            else None
        )

        spec_decoding_stats = None
        if spec_stats:
            spec_decoding_stats = {
                "total_generations": spec_stats.total_generations,
                "acceptance_rate": spec_stats.acceptance_rate,
                "avg_speedup": spec_stats.avg_speedup,
            }

        return {
            "request_count": self._request_count,
            "avg_latency_ms": self._total_latency_ms / max(1, self._request_count),
            "cache_hit_rate": self._cache_hits / max(1, self._request_count),
            "active_contexts": self.context_manager.get_active_count(),
            "models": self.router.get_stats(),
            "optimization": {
                "speculative_decoding": spec_decoding_stats,
                "resource_scheduler": scheduler_stats,
            },
        }

    def record_feedback(
        self,
        query: str,
        response_id: str,
        feedback_type: str,
        rating: float | None = None,
        comment: str | None = None,
    ) -> bool:
        """Record user feedback for continuous learning.

        Args:
            query: The original query
            response_id: ID of the response being rated
            feedback_type: Type of feedback (e.g., "helpful", "unhelpful", "inaccurate")
            rating: Optional numeric rating (0.0-1.0)
            comment: Optional user comment

        Returns:
            True if feedback was recorded successfully
        """
        feedback_learner = self._get_feedback_learner()
        if not feedback_learner:
            logger.debug("Feedback learner not available")
            return False

        try:
            from ...core.optimized.self_optimizer import UserFeedback

            import hashlib as _fb_hashlib

            feedback = UserFeedback(
                query_hash=_fb_hashlib.md5(
                    (query or "").encode(), usedforsecurity=False
                ).hexdigest()[:12],
                feedback_type=feedback_type or "helpful",
                rating=int(rating) if rating is not None else 3,
                metadata={
                    "response_id": response_id,
                    "comment": comment,
                    "original_query": query,
                },
            )

            feedback_learner.record_feedback(feedback)
            logger.debug(f"Recorded feedback: {feedback_type} for {response_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to record feedback: {e}")
            return False

    def get_optimization_insights(self) -> dict[str, Any]:
        """Get insights from all optimization components.

        Returns detailed metrics and recommendations from:
        - Self-optimizing retrieval loop
        - Predictive resource scheduler
        - Adaptive context allocator
        - Speculative decoding
        """
        insights = {}

        # Retrieval loop insights
        retrieval_loop = self._get_retrieval_loop()
        if retrieval_loop and hasattr(retrieval_loop, "get_insights"):
            insights["retrieval_loop"] = retrieval_loop.get_insights()

        # Feedback learner insights
        feedback_learner = self._get_feedback_learner()
        if feedback_learner and hasattr(feedback_learner, "get_patterns"):
            insights["feedback_patterns"] = feedback_learner.get_patterns()

        # Resource scheduler insights
        scheduler = self._get_resource_scheduler()
        if scheduler and hasattr(scheduler, "get_insights"):
            insights["resource_scheduler"] = scheduler.get_insights()

        # Speculative decoder insights
        spec_decoder = self._get_speculative_decoder()
        if spec_decoder:
            stats = spec_decoder.get_stats()
            insights["speculative_decoding"] = {
                "total_generations": stats.total_generations,
                "acceptance_rate": stats.acceptance_rate,
                "avg_speedup": stats.avg_speedup,
                "avg_draft_tokens": stats.avg_draft_tokens,
            }

        # Context allocator insights
        allocator = self._get_context_allocator()
        if allocator and hasattr(allocator, "get_stats"):
            insights["context_allocator"] = allocator.get_stats()

        return insights


# Singleton instance
_engine_instance: AIEngine | None = None
_engine_lock = threading.Lock()


def get_ai_engine(redis_client: Any | None = None) -> AIEngine:
    """Get or create the global AI engine instance."""
    global _engine_instance

    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = AIEngine(redis_client=redis_client)

    return _engine_instance
