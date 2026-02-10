"""
Self-Optimizer Module
=====================

Implements dynamic self-optimization loops that tune system parameters
based on runtime performance metrics. This closes the "final boss" gap
in enterprise RAG alignment.

Key Features:
- Dynamic chunk sizing based on retrieval quality
- Adaptive threshold tuning for reranking
- Temperature adjustment for generation quality
- Automatic batch size optimization
- Memory-aware scaling
- **Self-optimizing retrieval loop with iterative refinement**
- **Query intent classification for strategy selection**
- **Feedback-driven continuous improvement**

Unlike static configuration, this module learns from each request cycle
and adjusts parameters to optimize latency, quality, and resource usage.
"""

import asyncio
import hashlib
import logging
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# QUERY INTENT CLASSIFICATION
# ============================================================================


class QueryIntent:
    """Classifications for query types to optimize retrieval strategy."""

    FACTUAL = "factual"  # Direct fact lookup
    ANALYTICAL = "analytical"  # Requires reasoning
    COMPARATIVE = "comparative"  # Compare multiple items
    PROCEDURAL = "procedural"  # Step-by-step instructions
    CREATIVE = "creative"  # Open-ended generation
    CONVERSATIONAL = "conversational"  # Chat/follow-up


class QueryClassifier:
    """
    Classifies queries to select optimal retrieval strategy.

    Uses keyword patterns and query structure analysis.
    """

    INTENT_PATTERNS = {
        QueryIntent.FACTUAL: [
            "what is",
            "who is",
            "when did",
            "where is",
            "define",
            "meaning of",
            "what are",
            "how many",
            "which",
        ],
        QueryIntent.ANALYTICAL: [
            "why",
            "explain",
            "analyze",
            "how does",
            "what causes",
            "understand",
            "reason",
            "because",
            "impact of",
        ],
        QueryIntent.COMPARATIVE: [
            "compare",
            "difference between",
            "vs",
            "versus",
            "better",
            "similar to",
            "contrast",
            "pros and cons",
            "advantages",
        ],
        QueryIntent.PROCEDURAL: [
            "how to",
            "steps to",
            "guide",
            "tutorial",
            "process",
            "instructions",
            "method",
            "procedure",
            "way to",
        ],
        QueryIntent.CREATIVE: [
            "write",
            "create",
            "generate",
            "compose",
            "design",
            "imagine",
            "story",
            "poem",
            "suggest ideas",
        ],
    }

    # Strategy configs per intent
    INTENT_STRATEGIES = {
        QueryIntent.FACTUAL: {
            "retrieval_top_k": 10,
            "rerank_top_k": 3,
            "temperature": 0.3,
            "require_exact_match": True,
        },
        QueryIntent.ANALYTICAL: {
            "retrieval_top_k": 25,
            "rerank_top_k": 7,
            "temperature": 0.5,
            "require_exact_match": False,
        },
        QueryIntent.COMPARATIVE: {
            "retrieval_top_k": 30,
            "rerank_top_k": 10,
            "temperature": 0.4,
            "multi_source": True,
        },
        QueryIntent.PROCEDURAL: {
            "retrieval_top_k": 15,
            "rerank_top_k": 5,
            "temperature": 0.3,
            "preserve_order": True,
        },
        QueryIntent.CREATIVE: {
            "retrieval_top_k": 20,
            "rerank_top_k": 5,
            "temperature": 0.8,
            "allow_diverse": True,
        },
        QueryIntent.CONVERSATIONAL: {
            "retrieval_top_k": 10,
            "rerank_top_k": 3,
            "temperature": 0.7,
            "use_context": True,
        },
    }

    def classify(self, query: str) -> tuple[str, float, dict[str, Any]]:
        """
        Classify query intent.

        Returns:
            (intent, confidence, strategy_config)
        """
        query_lower = query.lower().strip()
        scores = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if p in query_lower)
            scores[intent] = score

        # Default to conversational if no strong signal
        if max(scores.values()) == 0:
            return (
                QueryIntent.CONVERSATIONAL,
                0.5,
                self.INTENT_STRATEGIES[QueryIntent.CONVERSATIONAL],
            )

        best_intent = max(scores, key=lambda k: scores[k])
        confidence = min(1.0, scores[best_intent] / 3.0)  # Max 3 matches = 100%

        return (best_intent, confidence, self.INTENT_STRATEGIES.get(best_intent, {}))


# ============================================================================
# RETRIEVAL LOOP OPTIMIZER
# ============================================================================


@dataclass
class RetrievalAttempt:
    """Record of a single retrieval attempt in the loop."""

    query: str
    reformulated_query: str | None = None
    results_count: int = 0
    relevance_score: float = 0.0
    latency_ms: float = 0.0
    strategy_used: str = "default"
    iteration: int = 0


class SelfOptimizingRetrievalLoop:
    """
    Implements iterative retrieval with self-optimization.

    Features:
    - Query reformulation on low-relevance results
    - Dynamic strategy switching based on results
    - Multi-hop retrieval for complex queries
    - Automatic fallback chains
    """

    MAX_ITERATIONS = 3
    MIN_RELEVANCE_THRESHOLD = 0.6
    REFORMULATION_TEMPLATES = [
        "Rephrase: {query}",
        "Related to: {query}",
        "Explain: {query}",
        "Details about: {query}",
    ]

    def __init__(
        self,
        retriever: Callable | None = None,
        reranker: Callable | None = None,
        classifier: QueryClassifier | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.classifier = classifier or QueryClassifier()
        self._attempts: deque = deque(maxlen=100)
        self._lock = threading.Lock()

    async def _execute_retrieval(
        self, query: str, top_k: int,
    ) -> list[Any]:
        """Execute retrieval using sync or async retriever."""
        if not self.retriever:
            return []
        if asyncio.iscoroutinefunction(self.retriever):
            result: list[Any] = await self.retriever(query, top_k=top_k)
            return result
        result = list(self.retriever(query, top_k=top_k))
        return result

    async def _apply_reranking(
        self, query: str, results: list[Any], top_k: int,
    ) -> list[Any]:
        """Apply reranking using sync or async reranker."""
        if not self.reranker or not results:
            return results
        if asyncio.iscoroutinefunction(self.reranker):
            reranked: list[Any] = await self.reranker(query, results, top_k=top_k)
            return reranked
        reranked = list(self.reranker(query, results, top_k=top_k))
        return reranked

    @staticmethod
    def _deduplicate_results(
        results: list[Any], all_results: list[Any], seen_ids: set[str],
        get_id: Callable[..., str],
    ) -> None:
        """Deduplicate and accumulate results in-place."""
        for r in results:
            r_id = get_id(r)
            if r_id not in seen_ids:
                all_results.append(r)
                seen_ids.add(r_id)

    async def retrieve_with_optimization(
        self,
        query: str,
        min_results: int = 3,
        max_iterations: int | None = None,
    ) -> tuple[list[Any], list[RetrievalAttempt]]:
        """
        Perform optimized retrieval with iterative refinement.

        Args:
            query: User query
            min_results: Minimum number of relevant results required
            max_iterations: Max refinement iterations

        Returns:
            (results, attempts_history)
        """
        max_iter = max_iterations or self.MAX_ITERATIONS
        attempts = []
        all_results: list[Any] = []
        seen_ids: set[str] = set()

        # Classify query intent
        intent, _confidence, strategy = self.classifier.classify(query)

        current_query = query
        retrieval_top_k = strategy.get("retrieval_top_k", 20)
        for iteration in range(max_iter):
            attempt = RetrievalAttempt(
                query=query,
                reformulated_query=current_query if iteration > 0 else None,
                iteration=iteration,
                strategy_used=intent,
            )

            start = time.perf_counter()
            results = await self._execute_retrieval(current_query, retrieval_top_k)
            attempt.latency_ms = (time.perf_counter() - start) * 1000
            attempt.results_count = len(results)

            # Calculate relevance (simplified - in production use embeddings)
            if results:
                attempt.relevance_score = self._estimate_relevance(query, results)

            attempts.append(attempt)
            self._deduplicate_results(results, all_results, seen_ids, self._get_result_id)

            # Exit when we have both sufficient coverage AND quality.
            has_enough = len(all_results) >= min_results
            has_quality = attempt.relevance_score >= self.MIN_RELEVANCE_THRESHOLD
            if has_enough and has_quality:
                break

            # Reformulate query for next iteration
            if iteration < max_iter - 1:
                current_query = self._reformulate_query(query, iteration, results)

        # Apply reranking if available
        rerank_k = strategy.get("rerank_top_k", 5)
        all_results = await self._apply_reranking(query, all_results, rerank_k)

        # Record attempts for learning
        with self._lock:
            for a in attempts:
                self._attempts.append(a)

        return all_results, attempts

    def _estimate_relevance(self, query: str, results: list[Any]) -> float:
        """Estimate relevance score (simplified)."""
        if not results:
            return 0.0

        query_words = set(query.lower().split())
        scores = []

        for r in results[:5]:  # Check top 5
            content = str(r.get("content", r) if isinstance(r, dict) else r)
            content_words = set(content.lower().split())
            overlap = len(query_words & content_words) / max(len(query_words), 1)
            scores.append(min(1.0, overlap * 2))  # Scale up

        return sum(scores) / len(scores) if scores else 0.0

    def _reformulate_query(
        self, original: str, iteration: int, _prev_results: list[Any]
    ) -> str:
        """Reformulate query for next iteration."""
        template = self.REFORMULATION_TEMPLATES[
            iteration % len(self.REFORMULATION_TEMPLATES)
        ]
        return template.format(query=original)

    def _get_result_id(self, result: Any) -> str:
        """Get unique ID for a result."""
        if isinstance(result, dict):
            return str(result.get("id", result.get("chunk_id", str(hash(str(result))))))
        return str(hash(str(result)))

    def get_learning_stats(self) -> dict[str, Any]:
        """Get statistics from retrieval attempts for learning."""
        with self._lock:
            if not self._attempts:
                return {}

            attempts_list = list(self._attempts)
            return {
                "total_attempts": len(attempts_list),
                "avg_iterations": statistics.mean(
                    a.iteration + 1 for a in attempts_list
                ),
                "avg_relevance": statistics.mean(
                    a.relevance_score for a in attempts_list
                ),
                "avg_latency_ms": statistics.mean(a.latency_ms for a in attempts_list),
                "reformulation_rate": sum(1 for a in attempts_list if a.iteration > 0)
                / len(attempts_list),
            }


@dataclass
class OptimizationMetrics:
    """Metrics collected for optimization decisions."""

    # Quality metrics
    retrieval_precision: float = 0.0
    rerank_improvement: float = 0.0
    generation_quality: float = 0.0
    semantic_coherence: float = 0.0

    # Latency metrics (ms)
    embedding_latency: float = 0.0
    retrieval_latency: float = 0.0
    rerank_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0

    # Resource metrics
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0

    # Request context
    input_tokens: int = 0
    output_tokens: int = 0
    chunks_retrieved: int = 0
    chunks_after_rerank: int = 0

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizedParameters:
    """Self-tuned parameters for the RAG pipeline."""

    # Chunk sizing
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    retrieval_top_k: int = 20
    retrieval_score_threshold: float = 0.5

    # Reranking
    rerank_top_k: int = 5
    rerank_threshold: float = 0.7

    # Generation
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024

    # Batching
    batch_size: int = 32

    # Memory
    max_context_length: int = 4096

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_score_threshold": self.retrieval_score_threshold,
            "rerank_top_k": self.rerank_top_k,
            "rerank_threshold": self.rerank_threshold,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "max_context_length": self.max_context_length,
        }


class SelfOptimizer:
    """
    Self-optimizing parameter tuner for the RAG pipeline.

    Implements closed-loop optimization:
    1. Collect metrics from each request cycle
    2. Analyze patterns and detect degradation
    3. Adjust parameters to optimize target metrics
    4. Validate changes and rollback if needed

    Optimization strategies:
    - Gradient-free optimization (no training required)
    - Exponential moving average for trend detection
    - Bounded parameter exploration
    - Automatic rollback on quality regression
    """

    # Parameter bounds for safe optimization
    PARAM_BOUNDS = {
        "chunk_size": (256, 1024),
        "chunk_overlap": (25, 100),
        "retrieval_top_k": (5, 50),
        "retrieval_score_threshold": (0.3, 0.8),
        "rerank_top_k": (3, 15),
        "rerank_threshold": (0.5, 0.9),
        "temperature": (0.1, 1.2),
        "top_p": (0.7, 1.0),
        "max_tokens": (256, 4096),
        "batch_size": (8, 128),
        "max_context_length": (2048, 8192),
    }

    # Target latencies (ms)
    TARGET_LATENCIES = {
        "embedding": 50,
        "retrieval": 100,
        "rerank": 150,
        "generation": 2000,
        "total": 3000,
    }

    # Quality thresholds
    QUALITY_THRESHOLDS = {
        "retrieval_precision": 0.7,
        "rerank_improvement": 0.1,  # Reranking should improve by at least 10%
        "generation_quality": 0.75,
        "semantic_coherence": 0.8,
    }

    def __init__(
        self,
        history_window: int = 100,
        update_interval: int = 10,
        enable_auto_tune: bool = True,
        ema_alpha: float = 0.2,
    ):
        """
        Initialize self-optimizer.

        Args:
            history_window: Number of recent metrics to keep for analysis
            update_interval: Update parameters every N requests
            enable_auto_tune: Whether to automatically adjust parameters
            ema_alpha: Exponential moving average smoothing factor
        """
        self.history_window = history_window
        self.update_interval = update_interval
        self.enable_auto_tune = enable_auto_tune
        self.ema_alpha = ema_alpha

        # Current optimized parameters
        self.params = OptimizedParameters()

        # Metrics history
        self._metrics_history: deque[OptimizationMetrics] = deque(maxlen=history_window)
        self._request_count = 0

        # EMA tracking for trend detection
        self._ema_metrics: dict[str, float] = {}

        # Previous parameters for rollback
        self._previous_params: OptimizedParameters | None = None
        self._last_optimization_time: datetime | None = None

        # Performance baseline
        self._baseline_quality: float | None = None
        self._baseline_latency: float | None = None

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            f"SelfOptimizer initialized: window={history_window}, "
            f"interval={update_interval}, auto_tune={enable_auto_tune}"
        )

    def record_metrics(self, metrics: OptimizationMetrics) -> None:
        """
        Record metrics from a request cycle.

        Args:
            metrics: Metrics from the completed request
        """
        with self._lock:
            self._metrics_history.append(metrics)
            self._request_count += 1

            # Update EMAs
            self._update_ema(metrics)

            # Check if optimization should run
            if (
                self.enable_auto_tune
                and self._request_count % self.update_interval == 0
                and len(self._metrics_history) >= self.update_interval
            ):
                self._run_optimization_cycle()

    def _update_ema(self, metrics: OptimizationMetrics) -> None:
        """Update exponential moving averages for trend detection."""
        metric_values = {
            "retrieval_precision": metrics.retrieval_precision,
            "rerank_improvement": metrics.rerank_improvement,
            "generation_quality": metrics.generation_quality,
            "total_latency": metrics.total_latency,
            "cache_hit_rate": metrics.cache_hit_rate,
        }

        for name, value in metric_values.items():
            if name not in self._ema_metrics:
                self._ema_metrics[name] = value
            else:
                self._ema_metrics[name] = (
                    self.ema_alpha * value
                    + (1 - self.ema_alpha) * self._ema_metrics[name]
                )

    def _run_optimization_cycle(self) -> None:
        """Run a complete optimization cycle."""
        logger.debug("Running optimization cycle...")

        # Analyze current performance
        analysis = self._analyze_performance()

        # Store baseline if not set
        if self._baseline_quality is None:
            self._baseline_quality = analysis["avg_quality"]
            self._baseline_latency = analysis["avg_latency"]
            logger.info(
                f"Baseline set: quality={self._baseline_quality:.3f}, "
                f"latency={self._baseline_latency:.1f}ms"
            )
            return

        # Check for regression and rollback if needed
        if self._previous_params and self._should_rollback(analysis):
            self._rollback_parameters()
            return

        # Store current params before modification
        self._previous_params = OptimizedParameters(**self.params.to_dict())

        # Determine optimization direction
        optimizations = self._determine_optimizations(analysis)

        # Apply optimizations
        if optimizations:
            self._apply_optimizations(optimizations)
            self._last_optimization_time = datetime.now()
            logger.info(f"Applied optimizations: {optimizations}")

    def _analyze_performance(self) -> dict[str, Any]:
        """Analyze recent performance metrics."""
        recent = list(self._metrics_history)
        if not recent:
            return {"avg_quality": 0, "avg_latency": 0, "issues": []}

        # Calculate averages
        avg_quality = statistics.mean(
            [
                (m.retrieval_precision + m.generation_quality + m.semantic_coherence)
                / 3
                for m in recent
            ]
        )
        avg_latency = statistics.mean([m.total_latency for m in recent])
        avg_cache_hit = statistics.mean([m.cache_hit_rate for m in recent])

        # Detect issues
        issues = []

        # Quality issues
        if avg_quality < self.QUALITY_THRESHOLDS["retrieval_precision"]:
            issues.append("low_retrieval_quality")

        avg_rerank_improvement = statistics.mean([m.rerank_improvement for m in recent])
        if avg_rerank_improvement < self.QUALITY_THRESHOLDS["rerank_improvement"]:
            issues.append("low_rerank_benefit")

        # Latency issues
        avg_gen_latency = statistics.mean([m.generation_latency for m in recent])
        if avg_gen_latency > self.TARGET_LATENCIES["generation"]:
            issues.append("high_generation_latency")

        if avg_latency > self.TARGET_LATENCIES["total"]:
            issues.append("high_total_latency")

        # Resource issues
        avg_memory = statistics.mean([m.memory_usage_mb for m in recent])
        if avg_memory > 8000:  # 8GB threshold
            issues.append("high_memory_usage")

        # Cache efficiency
        if avg_cache_hit < 0.3:
            issues.append("low_cache_efficiency")

        return {
            "avg_quality": avg_quality,
            "avg_latency": avg_latency,
            "avg_cache_hit": avg_cache_hit,
            "avg_rerank_improvement": avg_rerank_improvement,
            "avg_memory": avg_memory,
            "issues": issues,
            "sample_size": len(recent),
        }

    def _should_rollback(self, analysis: dict[str, Any]) -> bool:
        """Check if we should rollback to previous parameters."""
        if self._baseline_quality is None:
            return False

        # Rollback if quality dropped significantly
        quality_drop = (
            self._baseline_quality - analysis["avg_quality"]
        ) / self._baseline_quality
        if quality_drop > 0.1:  # 10% quality regression
            logger.warning(
                f"Quality regression detected: {quality_drop:.1%} drop. "
                f"Rolling back parameters."
            )
            return True

        # Rollback if latency increased significantly without quality gain
        latency_increase = (
            analysis["avg_latency"] - self._baseline_latency
        ) / self._baseline_latency
        quality_gain = (
            analysis["avg_quality"] - self._baseline_quality
        ) / self._baseline_quality

        if (
            latency_increase > 0.3 and quality_gain < 0.05
        ):  # 30% latency increase with <5% quality gain
            logger.warning("Latency regression without quality gain. Rolling back.")
            return True

        return False

    def _rollback_parameters(self) -> None:
        """Rollback to previous parameters."""
        if self._previous_params:
            self.params = self._previous_params
            self._previous_params = None
            logger.info("Rolled back to previous parameters")

    def _determine_optimizations(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Determine what optimizations to apply based on analysis."""
        optimizations = {}
        issues = analysis["issues"]

        # Low retrieval quality -> increase retrieval K, decrease threshold
        if "low_retrieval_quality" in issues:
            new_k = min(
                self.params.retrieval_top_k + 5, self.PARAM_BOUNDS["retrieval_top_k"][1]
            )
            new_threshold = max(
                self.params.retrieval_score_threshold - 0.05,
                self.PARAM_BOUNDS["retrieval_score_threshold"][0],
            )
            optimizations["retrieval_top_k"] = new_k
            optimizations["retrieval_score_threshold"] = new_threshold

        # Low rerank benefit -> reduce rerank K (save compute) or adjust threshold
        if "low_rerank_benefit" in issues:
            # If reranking isn't helping, we might be over-filtering
            new_threshold = max(
                self.params.rerank_threshold - 0.05,
                self.PARAM_BOUNDS["rerank_threshold"][0],
            )
            optimizations["rerank_threshold"] = new_threshold

        # High generation latency -> reduce max tokens or lower temperature
        if "high_generation_latency" in issues:
            new_max_tokens = max(
                self.params.max_tokens - 256, self.PARAM_BOUNDS["max_tokens"][0]
            )
            optimizations["max_tokens"] = new_max_tokens

        # High total latency -> reduce retrieval K
        if "high_total_latency" in issues and "low_retrieval_quality" not in issues:
            new_k = max(
                self.params.retrieval_top_k - 3, self.PARAM_BOUNDS["retrieval_top_k"][0]
            )
            optimizations["retrieval_top_k"] = new_k

        # High memory usage -> reduce batch size and context length
        if "high_memory_usage" in issues:
            new_batch = max(
                self.params.batch_size // 2, self.PARAM_BOUNDS["batch_size"][0]
            )
            new_context = max(
                self.params.max_context_length - 1024,
                self.PARAM_BOUNDS["max_context_length"][0],
            )
            optimizations["batch_size"] = new_batch
            optimizations["max_context_length"] = new_context

        # Low cache efficiency -> increase chunk overlap for better reuse
        if "low_cache_efficiency" in issues:
            new_overlap = min(
                self.params.chunk_overlap + 10, self.PARAM_BOUNDS["chunk_overlap"][1]
            )
            optimizations["chunk_overlap"] = new_overlap

        return optimizations

    def _apply_optimizations(self, optimizations: dict[str, Any]) -> None:
        """Apply optimization updates to parameters."""
        for param, value in optimizations.items():
            if hasattr(self.params, param):
                setattr(self.params, param, value)

    def get_current_params(self) -> OptimizedParameters:
        """Get current optimized parameters."""
        with self._lock:
            return OptimizedParameters(**self.params.to_dict())

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get optimization statistics and current state."""
        with self._lock:
            analysis = self._analyze_performance() if self._metrics_history else {}

            return {
                "enabled": self.enable_auto_tune,
                "request_count": self._request_count,
                "history_size": len(self._metrics_history),
                "current_params": self.params.to_dict(),
                "ema_metrics": self._ema_metrics.copy(),
                "baseline_quality": self._baseline_quality,
                "baseline_latency": self._baseline_latency,
                "last_optimization": (
                    self._last_optimization_time.isoformat()
                    if self._last_optimization_time
                    else None
                ),
                "current_analysis": analysis,
            }

    def force_optimize(self) -> dict[str, Any]:
        """Force an optimization cycle (for manual intervention)."""
        with self._lock:
            if len(self._metrics_history) < 5:
                return {
                    "status": "insufficient_data",
                    "required": 5,
                    "current": len(self._metrics_history),
                }

            self._run_optimization_cycle()
            return {
                "status": "optimized",
                "params": self.params.to_dict(),
                "stats": self.get_optimization_stats(),
            }

    def reset_baseline(self) -> None:
        """Reset baseline metrics (useful after major changes)."""
        with self._lock:
            self._baseline_quality = None
            self._baseline_latency = None
            self._previous_params = None
            logger.info("Baseline reset - will be recalculated on next cycle")

    def set_target_latency(self, component: str, target_ms: float) -> None:
        """Update target latency for a component."""
        if component in self.TARGET_LATENCIES:
            self.TARGET_LATENCIES[component] = int(target_ms)
            logger.info(f"Updated target latency for {component}: {target_ms}ms")


# ============================================================================
# FEEDBACK-DRIVEN LEARNING
# ============================================================================


@dataclass
class UserFeedback:
    """User feedback for response quality."""

    query_hash: str
    rating: int  # 1-5
    feedback_type: str  # "helpful", "accurate", "complete", "fast"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackLearner:
    """
    Learns from user feedback to improve parameters.

    Correlates feedback with query types and parameter settings
    to optimize future responses.
    """

    def __init__(self, window_size: int = 500):
        self._feedback: deque = deque(maxlen=window_size)
        self._query_params: dict[str, dict[str, Any]] = {}  # query_hash -> params used
        self._lock = threading.Lock()

    def record_query(
        self, query: str, params: dict[str, Any], intent: str = "unknown"
    ) -> str:
        """Record query and parameters used. Returns query hash."""
        query_hash = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:12]

        with self._lock:
            self._query_params[query_hash] = {
                "params": params,
                "intent": intent,
                "timestamp": datetime.now(),
            }

        return query_hash

    def record_feedback(
        self, query_hash: str, rating: int, feedback_type: str = "helpful"
    ) -> None:
        """Record user feedback for a query."""
        feedback = UserFeedback(
            query_hash=query_hash,
            rating=rating,
            feedback_type=feedback_type,
        )

        with self._lock:
            self._feedback.append(feedback)

    def get_optimal_params_for_intent(self, intent: str) -> dict[str, Any] | None:
        """
        Get best-performing parameters for a query intent.

        Analyzes feedback to find which parameters work best
        for each query type.
        """
        with self._lock:
            # Find queries with this intent and good feedback
            good_params = []

            for feedback in self._feedback:
                is_good = feedback.rating >= 4
                has_params = feedback.query_hash in self._query_params
                if is_good and has_params:
                    query_info = self._query_params[feedback.query_hash]
                    if query_info.get("intent") == intent:
                        good_params.append(query_info["params"])

            if not good_params:
                return None

            # Average the good parameters
            averaged = {}
            for key in good_params[0]:
                values = [p.get(key) for p in good_params if key in p]
                if values and all(isinstance(v, (int, float)) for v in values):
                    averaged[key] = sum(values) / len(values)

            return averaged if averaged else None

    def get_feedback_stats(self) -> dict[str, Any]:
        """Get feedback statistics."""
        with self._lock:
            if not self._feedback:
                return {"total": 0}

            feedback_list = list(self._feedback)
            return {
                "total": len(feedback_list),
                "avg_rating": statistics.mean(f.rating for f in feedback_list),
                "by_type": {
                    ftype: statistics.mean(
                        f.rating for f in feedback_list if f.feedback_type == ftype
                    )
                    for ftype in {f.feedback_type for f in feedback_list}
                },
            }


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

_optimizer_instance: SelfOptimizer | None = None
_optimizer_lock = threading.Lock()
_retrieval_loop: SelfOptimizingRetrievalLoop | None = None
_feedback_learner: FeedbackLearner | None = None


def get_self_optimizer(
    history_window: int = 100,
    update_interval: int = 10,
    enable_auto_tune: bool = True,
) -> SelfOptimizer:
    """Get or create the global SelfOptimizer instance."""
    global _optimizer_instance

    with _optimizer_lock:
        if _optimizer_instance is None:
            _optimizer_instance = SelfOptimizer(
                history_window=history_window,
                update_interval=update_interval,
                enable_auto_tune=enable_auto_tune,
            )
            logger.info("Created SelfOptimizer singleton")
        return _optimizer_instance


def get_retrieval_loop(
    retriever: Callable | None = None,
    reranker: Callable | None = None,
) -> SelfOptimizingRetrievalLoop:
    """Get or create the global retrieval loop instance."""
    global _retrieval_loop

    with _optimizer_lock:
        if _retrieval_loop is None:
            _retrieval_loop = SelfOptimizingRetrievalLoop(
                retriever=retriever,
                reranker=reranker,
            )
            logger.info("Created SelfOptimizingRetrievalLoop singleton")
        return _retrieval_loop


def get_feedback_learner() -> FeedbackLearner:
    """Get or create the global feedback learner instance."""
    global _feedback_learner

    with _optimizer_lock:
        if _feedback_learner is None:
            _feedback_learner = FeedbackLearner()
            logger.info("Created FeedbackLearner singleton")
        return _feedback_learner


def reset_optimizer() -> None:
    """Reset the optimizer singleton (for testing)."""
    global _optimizer_instance, _retrieval_loop, _feedback_learner
    with _optimizer_lock:
        _optimizer_instance = None
        _retrieval_loop = None
        _feedback_learner = None


__all__ = [
    "FeedbackLearner",
    # Core optimizer
    "OptimizationMetrics",
    "OptimizedParameters",
    "QueryClassifier",
    # Query classification
    "QueryIntent",
    # Self-optimizing retrieval loop
    "RetrievalAttempt",
    "SelfOptimizer",
    "SelfOptimizingRetrievalLoop",
    # Feedback learning
    "UserFeedback",
    "get_feedback_learner",
    "get_retrieval_loop",
    "get_self_optimizer",
    "reset_optimizer",
]
