"""
Prometheus Metrics & Monitoring (Principle S)
=============================================
Log inference latency and track performance metrics.

Strategy:
- Prometheus metrics for latency, throughput, errors
- Per-model, per-task metrics
- Memory usage tracking
- Request/response logging

Reference: "Log inference latency p50/p95"
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Try to import prometheus_client, gracefully handle if not installed
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
        multiprocess,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, metrics will be no-op")


class MetricType(Enum):
    """Types of metrics."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR = "error"
    MEMORY = "memory"
    QUEUE = "queue"


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    enabled: bool = True
    prefix: str = "ssetu"

    # Histogram buckets for latency (seconds)
    latency_buckets: tuple = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

    # Memory tracking interval (seconds)
    memory_check_interval: float = 10.0

    # Log slow requests threshold (seconds)
    slow_request_threshold: float = 5.0


# Global metrics registry
_metrics: dict[str, Any] = {}
_config = MetricsConfig()


def _create_metrics():
    """Create Prometheus metrics if available."""
    global _metrics

    if not PROMETHEUS_AVAILABLE or not _config.enabled:
        return

    prefix = _config.prefix

    # Inference latency histogram (Principle S)
    _metrics["inference_latency"] = Histogram(
        f"{prefix}_inference_latency_seconds",
        "Inference latency in seconds",
        ["model", "task", "status"],
        buckets=_config.latency_buckets,
    )

    # Request counter
    _metrics["requests_total"] = Counter(
        f"{prefix}_requests_total",
        "Total requests processed",
        ["model", "task", "status"],
    )

    # Active requests gauge
    _metrics["active_requests"] = Gauge(
        f"{prefix}_active_requests", "Currently active requests", ["model", "task"]
    )

    # Tokens processed
    _metrics["tokens_processed"] = Counter(
        f"{prefix}_tokens_processed_total",
        "Total tokens processed",
        ["model", "direction"],  # direction: input/output
    )

    # Model memory usage
    _metrics["model_memory_bytes"] = Gauge(
        f"{prefix}_model_memory_bytes", "Memory used by models", ["model"]
    )

    # Model load time
    _metrics["model_load_seconds"] = Histogram(
        f"{prefix}_model_load_seconds",
        "Time to load models",
        ["model"],
        buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    )

    # Queue metrics
    _metrics["queue_length"] = Gauge(
        f"{prefix}_queue_length", "Current queue length", ["queue"]
    )

    _metrics["queue_wait_seconds"] = Histogram(
        f"{prefix}_queue_wait_seconds",
        "Time spent waiting in queue",
        ["queue"],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    )

    # Error metrics
    _metrics["errors_total"] = Counter(
        f"{prefix}_errors_total", "Total errors", ["model", "task", "error_type"]
    )

    # Cache metrics
    _metrics["cache_hits"] = Counter(
        f"{prefix}_cache_hits_total", "Cache hits", ["cache_type"]
    )

    _metrics["cache_misses"] = Counter(
        f"{prefix}_cache_misses_total", "Cache misses", ["cache_type"]
    )

    # System memory
    _metrics["system_memory_bytes"] = Gauge(
        f"{prefix}_system_memory_bytes",
        "System memory usage",
        ["type"],  # used, available, total
    )

    # GPU memory (if available)
    _metrics["gpu_memory_bytes"] = Gauge(
        f"{prefix}_gpu_memory_bytes",
        "GPU memory usage",
        ["device", "type"],  # type: used, total
    )

    logger.info(f"Prometheus metrics initialized with prefix: {prefix}")


# Initialize metrics
_create_metrics()


# =============================================================================
# Metric Recording Functions
# =============================================================================


def record_inference_latency(
    model: str, task: str, latency_seconds: float, status: str = "success"
):
    """
    Record inference latency (Principle S).

    Args:
        model: Model name
        task: Task type (simplify, translate, etc.)
        latency_seconds: Latency in seconds
        status: success or error
    """
    if not PROMETHEUS_AVAILABLE or "inference_latency" not in _metrics:
        return

    _metrics["inference_latency"].labels(model=model, task=task, status=status).observe(
        latency_seconds
    )

    _metrics["requests_total"].labels(model=model, task=task, status=status).inc()

    # Log slow requests
    if latency_seconds > _config.slow_request_threshold:
        logger.warning(
            f"Slow inference: model={model} task={task} latency={latency_seconds:.2f}s"
        )


def record_tokens(model: str, input_tokens: int, output_tokens: int):
    """Record tokens processed."""
    if not PROMETHEUS_AVAILABLE or "tokens_processed" not in _metrics:
        return

    _metrics["tokens_processed"].labels(model=model, direction="input").inc(
        input_tokens
    )
    _metrics["tokens_processed"].labels(model=model, direction="output").inc(
        output_tokens
    )


def record_model_memory(model: str, memory_bytes: int):
    """Record model memory usage."""
    if not PROMETHEUS_AVAILABLE or "model_memory_bytes" not in _metrics:
        return

    _metrics["model_memory_bytes"].labels(model=model).set(memory_bytes)


def record_model_load_time(model: str, load_seconds: float):
    """Record model load time."""
    if not PROMETHEUS_AVAILABLE or "model_load_seconds" not in _metrics:
        return

    _metrics["model_load_seconds"].labels(model=model).observe(load_seconds)


def record_queue_length(queue: str, length: int):
    """Record queue length."""
    if not PROMETHEUS_AVAILABLE or "queue_length" not in _metrics:
        return

    _metrics["queue_length"].labels(queue=queue).set(length)


def record_queue_wait(queue: str, wait_seconds: float):
    """Record queue wait time."""
    if not PROMETHEUS_AVAILABLE or "queue_wait_seconds" not in _metrics:
        return

    _metrics["queue_wait_seconds"].labels(queue=queue).observe(wait_seconds)


def record_error(model: str, task: str, error_type: str):
    """Record error."""
    if not PROMETHEUS_AVAILABLE or "errors_total" not in _metrics:
        return

    _metrics["errors_total"].labels(model=model, task=task, error_type=error_type).inc()


def record_cache_hit(cache_type: str):
    """Record cache hit."""
    if not PROMETHEUS_AVAILABLE or "cache_hits" not in _metrics:
        return

    _metrics["cache_hits"].labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str):
    """Record cache miss."""
    if not PROMETHEUS_AVAILABLE or "cache_misses" not in _metrics:
        return

    _metrics["cache_misses"].labels(cache_type=cache_type).inc()


def record_system_memory():
    """Record system memory usage."""
    if not PROMETHEUS_AVAILABLE or "system_memory_bytes" not in _metrics:
        return

    try:
        import psutil

        mem = psutil.virtual_memory()

        _metrics["system_memory_bytes"].labels(type="used").set(mem.used)
        _metrics["system_memory_bytes"].labels(type="available").set(mem.available)
        _metrics["system_memory_bytes"].labels(type="total").set(mem.total)
    except ImportError:
        pass


def record_gpu_memory():
    """Record GPU memory usage."""
    if not PROMETHEUS_AVAILABLE or "gpu_memory_bytes" not in _metrics:
        return

    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory

                _metrics["gpu_memory_bytes"].labels(
                    device=f"cuda:{i}", type="used"
                ).set(allocated)
                _metrics["gpu_memory_bytes"].labels(
                    device=f"cuda:{i}", type="total"
                ).set(total)
    except ImportError:
        pass


# =============================================================================
# Context Managers and Decorators
# =============================================================================


@contextmanager
def track_inference(model: str, task: str):
    """
    Context manager to track inference latency.

    Usage:
        with track_inference("qwen3-8b", "simplify"):
            result = await model.generate(...)
    """
    start_time = time.time()
    status = "success"

    # Track active requests
    if PROMETHEUS_AVAILABLE and "active_requests" in _metrics:
        _metrics["active_requests"].labels(model=model, task=task).inc()

    try:
        yield
    except Exception as e:
        status = "error"
        record_error(model, task, type(e).__name__)
        raise
    finally:
        latency = time.time() - start_time
        record_inference_latency(model, task, latency, status)

        if PROMETHEUS_AVAILABLE and "active_requests" in _metrics:
            _metrics["active_requests"].labels(model=model, task=task).dec()


def track_latency(model: str, task: str):
    """
    Decorator to track function latency.

    Usage:
        @track_latency("qwen3-8b", "simplify")
        async def simplify_text(text):
            ...
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with track_inference(model, task):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with track_inference(model, task):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Metrics Endpoint
# =============================================================================


def get_metrics() -> bytes:
    """
    Get metrics in Prometheus format.

    Returns:
        Prometheus metrics as bytes
    """
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus not available\n"

    # Update system metrics
    record_system_memory()
    record_gpu_memory()

    return generate_latest(REGISTRY)


def get_metrics_content_type() -> str:
    """Get content type for metrics endpoint."""
    if PROMETHEUS_AVAILABLE:
        return CONTENT_TYPE_LATEST
    return "text/plain"


# =============================================================================
# FastAPI Integration
# =============================================================================


def setup_metrics_endpoint(app):
    """
    Setup metrics endpoint for FastAPI.

    Args:
        app: FastAPI application instance
    """
    from fastapi import Response

    @app.get("/metrics")
    async def metrics():
        return Response(content=get_metrics(), media_type=get_metrics_content_type())

    logger.info("Metrics endpoint registered at /metrics")


# =============================================================================
# Latency Summary Statistics
# =============================================================================


class LatencyTracker:
    """
    Track latency statistics for reporting.

    Provides p50, p95, p99 latency statistics.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latencies: dict[str, list] = {}

    def record(self, key: str, latency_ms: float):
        """Record a latency measurement."""
        if key not in self._latencies:
            self._latencies[key] = []

        self._latencies[key].append(latency_ms)

        # Keep only window_size most recent
        if len(self._latencies[key]) > self.window_size:
            self._latencies[key] = self._latencies[key][-self.window_size :]

    def get_stats(self, key: str) -> dict[str, float]:
        """Get latency statistics for a key."""
        if key not in self._latencies or not self._latencies[key]:
            return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "count": 0}

        import numpy as np

        data = np.array(self._latencies[key])

        return {
            "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "mean": float(np.mean(data)),
            "count": len(data),
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all tracked keys."""
        return {key: self.get_stats(key) for key in self._latencies}


# Global latency tracker
latency_tracker = LatencyTracker()


def log_latency(model: str, task: str, latency_ms: float):
    """
    Log latency for statistics tracking.

    Args:
        model: Model name
        task: Task type
        latency_ms: Latency in milliseconds
    """
    key = f"{model}:{task}"
    latency_tracker.record(key, latency_ms)

    # Also record to Prometheus
    record_inference_latency(model, task, latency_ms / 1000.0)


def get_latency_stats() -> dict[str, dict[str, float]]:
    """Get latency statistics for all models/tasks."""
    return latency_tracker.get_all_stats()
