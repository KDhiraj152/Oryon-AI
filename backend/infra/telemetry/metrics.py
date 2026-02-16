"""
Metrics Collector — Prometheus + Internal Metrics
===================================================

Centralized metrics registry for all backend layers.
Provides typed metric primitives (counters, gauges, histograms)
with automatic label injection and Prometheus exposition.

Design:
  - Single registry, no scattered metric creation
  - Pre-defined metrics for inference, hardware, queue, cache
  - Thread-safe and async-compatible
  - Latency percentile tracking (p50, p95, p99)
  - Token throughput accounting

Metric Naming Convention:
  - oryon_{layer}_{component}_{metric}_{unit}
  - e.g., oryon_execution_inference_latency_seconds
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from backend.infra.telemetry.logger import get_logger

logger = get_logger(__name__)

# ── Try importing Prometheus ───────────────────────────────────────

try:
    from prometheus_client import (
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# ── Percentile Tracker ─────────────────────────────────────────────

class PercentileTracker:
    """Thread-safe rolling window percentile calculator with cached sorting."""

    __slots__ = ("_lock", "_maxlen", "_sorted_cache", "_sorted_dirty", "_values")

    def __init__(self, window_size: int = 1000):
        self._values: deque[float] = deque(maxlen=window_size)
        self._maxlen = window_size
        self._lock = threading.Lock()
        self._sorted_dirty = True
        self._sorted_cache: list[float] = []

    def record(self, value: float) -> None:
        with self._lock:
            self._values.append(value)
            self._sorted_dirty = True

    def percentile(self, p: float) -> float:
        """Get percentile value (0-100). Uses cached sort; only re-sorts when data changes."""
        with self._lock:
            if not self._values:
                return 0.0
            if self._sorted_dirty:
                self._sorted_cache = sorted(self._values)
                self._sorted_dirty = False
            idx = int(len(self._sorted_cache) * p / 100)
            return self._sorted_cache[min(idx, len(self._sorted_cache) - 1)]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def count(self) -> int:
        return len(self._values)

    def mean(self) -> float:
        with self._lock:
            if not self._values:
                return 0.0
            return sum(self._values) / len(self._values)

# ── Throughput Tracker ─────────────────────────────────────────────

class ThroughputTracker:
    """Tracks requests/sec and tokens/sec over a sliding window."""

    __slots__ = ("_lock", "_timestamps", "_tokens", "_window")

    def __init__(self, window_seconds: float = 60.0):
        self._timestamps: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._window = window_seconds
        self._lock = threading.Lock()

    def record_request(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._timestamps.append(now)
            self._prune(now)

    def record_tokens(self, count: int) -> None:
        with self._lock:
            now = time.monotonic()
            self._tokens.append((now, count))
            self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        while self._tokens and self._tokens[0][0] < cutoff:
            self._tokens.popleft()

    @property
    def requests_per_second(self) -> float:
        with self._lock:
            self._prune(time.monotonic())
            if not self._timestamps:
                return 0.0
            return len(self._timestamps) / self._window

    @property
    def tokens_per_second(self) -> float:
        with self._lock:
            self._prune(time.monotonic())
            if not self._tokens:
                return 0.0
            total = sum(t[1] for t in self._tokens)
            return total / self._window

# ── Metrics Collector ──────────────────────────────────────────────

class MetricsCollector:
    """
    Centralized metrics collection.

    Pre-defines all application metrics with proper labels.
    Supports both Prometheus export and internal percentile tracking.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Internal percentile trackers (per model)
        self._latency_trackers: dict[str, PercentileTracker] = {}
        self._throughput = ThroughputTracker()
        self._gpu_tracker = PercentileTracker(window_size=720)

        if not HAS_PROMETHEUS:
            logger.info("prometheus_not_installed", msg="Using internal metrics only")
            return

        # ── Inference Metrics ──
        self.inference_latency = Histogram(
            "oryon_execution_inference_latency_seconds",
            "Model inference latency",
            labelnames=["model", "task", "device"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )

        self.inference_requests = Counter(
            "oryon_execution_inference_requests_total",
            "Total inference requests",
            labelnames=["model", "task", "status"],
        )

        self.inference_active = Gauge(
            "oryon_execution_inference_active",
            "Currently active inference tasks",
            labelnames=["model"],
        )

        self.inference_tokens = Counter(
            "oryon_execution_tokens_processed_total",
            "Total tokens processed",
            labelnames=["model", "direction"],  # direction: input/output
        )

        self.inference_batch_size = Histogram(
            "oryon_execution_batch_size",
            "Inference batch sizes",
            labelnames=["model"],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        )

        # ── Queue Metrics ──
        self.queue_depth = Gauge(
            "oryon_orchestration_queue_depth",
            "Current queue depth",
            labelnames=["priority"],
        )

        self.queue_wait_time = Histogram(
            "oryon_orchestration_queue_wait_seconds",
            "Time spent waiting in queue",
            labelnames=["priority"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
        )

        self.queue_shed = Counter(
            "oryon_orchestration_requests_shed_total",
            "Requests shed due to overload",
            labelnames=["reason"],
        )

        # ── Hardware Metrics ──
        self.gpu_memory_used = Gauge(
            "oryon_hardware_gpu_memory_used_bytes",
            "GPU memory used",
            labelnames=["device"],
        )

        self.gpu_memory_total = Gauge(
            "oryon_hardware_gpu_memory_total_bytes",
            "GPU memory total",
            labelnames=["device"],
        )

        self.gpu_utilization = Gauge(
            "oryon_hardware_gpu_utilization_ratio",
            "GPU utilization ratio (0-1)",
            labelnames=["device"],
        )

        self.system_memory_used = Gauge(
            "oryon_hardware_system_memory_used_bytes",
            "System memory used",
        )

        self.system_cpu_percent = Gauge(
            "oryon_hardware_cpu_utilization_percent",
            "CPU utilization percentage",
        )

        self.active_models = Gauge(
            "oryon_hardware_active_models",
            "Number of models currently loaded",
        )

        # ── Cache Metrics ──
        self.cache_hits = Counter(
            "oryon_memory_cache_hits_total",
            "Cache hits",
            labelnames=["tier", "cache_type"],
        )

        self.cache_misses = Counter(
            "oryon_memory_cache_misses_total",
            "Cache misses",
            labelnames=["tier", "cache_type"],
        )

        self.cache_evictions = Counter(
            "oryon_memory_cache_evictions_total",
            "Cache evictions",
            labelnames=["cache_type", "reason"],
        )

        # ── API Metrics ──
        self.http_requests = Counter(
            "oryon_api_http_requests_total",
            "Total HTTP requests",
            labelnames=["method", "path", "status"],
        )

        self.http_latency = Histogram(
            "oryon_api_http_latency_seconds",
            "HTTP request latency",
            labelnames=["method", "path"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        self.http_active = Gauge(
            "oryon_api_http_active_requests",
            "Active HTTP requests",
        )

        # ── Circuit Breaker Metrics ──
        self.circuit_state = Gauge(
            "oryon_orchestration_circuit_state",
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
            labelnames=["circuit"],
        )

    # ── Recording Methods ──────────────────────────────────────────

    def record_inference(
        self,
        *,
        model: str,
        task: str,
        device: str,
        latency_s: float,
        status: str = "success",
        input_tokens: int = 0,
        output_tokens: int = 0,
        batch_size: int = 1,
    ) -> None:
        """Record a completed inference request."""
        # Internal tracker
        tracker = self._get_latency_tracker(model)
        tracker.record(latency_s)
        self._throughput.record_request()
        if output_tokens > 0:
            self._throughput.record_tokens(output_tokens)

        if not HAS_PROMETHEUS:
            return

        self.inference_latency.labels(
            model=model, task=task, device=device
        ).observe(latency_s)
        self.inference_requests.labels(
            model=model, task=task, status=status
        ).inc()
        if input_tokens > 0:
            self.inference_tokens.labels(model=model, direction="input").inc(
                input_tokens
            )
        if output_tokens > 0:
            self.inference_tokens.labels(model=model, direction="output").inc(
                output_tokens
            )
        self.inference_batch_size.labels(model=model).observe(batch_size)

    @contextmanager
    def track_inference(
        self, *, model: str, task: str, device: str
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager to track inference timing and metadata."""
        meta: dict[str, Any] = {"start": time.monotonic()}
        if HAS_PROMETHEUS:
            self.inference_active.labels(model=model).inc()
        try:
            yield meta
            status = "success"
        except (RuntimeError, OSError):
            status = "error"
            raise
        finally:
            latency = time.monotonic() - meta["start"]
            if HAS_PROMETHEUS:
                self.inference_active.labels(model=model).dec()
            self.record_inference(
                model=model,
                task=task,
                device=device,
                latency_s=latency,
                status=status,
                input_tokens=meta.get("input_tokens", 0),
                output_tokens=meta.get("output_tokens", 0),
                batch_size=meta.get("batch_size", 1),
            )

    def record_gpu_utilization(self, device: str, utilization: float, memory_used: int, memory_total: int) -> None:
        """Record GPU utilization snapshot."""
        self._gpu_tracker.record(utilization)
        if not HAS_PROMETHEUS:
            return
        self.gpu_utilization.labels(device=device).set(utilization)
        self.gpu_memory_used.labels(device=device).set(memory_used)
        self.gpu_memory_total.labels(device=device).set(memory_total)

    def record_queue_depth(self, priority: str, depth: int) -> None:
        if HAS_PROMETHEUS:
            self.queue_depth.labels(priority=priority).set(depth)

    def record_cache_access(self, *, tier: str, cache_type: str, hit: bool) -> None:
        if not HAS_PROMETHEUS:
            return
        if hit:
            self.cache_hits.labels(tier=tier, cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(tier=tier, cache_type=cache_type).inc()

    def record_http_request(self, *, method: str, path: str, status: int, latency_s: float) -> None:
        if not HAS_PROMETHEUS:
            return
        self.http_requests.labels(method=method, path=path, status=str(status)).inc()
        self.http_latency.labels(method=method, path=path).observe(latency_s)

    # ── Percentile Access ──────────────────────────────────────────

    def _get_latency_tracker(self, model: str) -> PercentileTracker:
        if model not in self._latency_trackers:
            with self._lock:
                if model not in self._latency_trackers:
                    self._latency_trackers[model] = PercentileTracker()
        return self._latency_trackers[model]

    def get_latency_percentiles(self, model: str) -> dict[str, float]:
        tracker = self._get_latency_tracker(model)
        return {
            "p50": tracker.p50,
            "p95": tracker.p95,
            "p99": tracker.p99,
            "mean": tracker.mean(),
            "count": tracker.count,
        }

    def get_throughput(self) -> dict[str, float]:
        return {
            "requests_per_second": self._throughput.requests_per_second,
            "tokens_per_second": self._throughput.tokens_per_second,
        }

    def get_gpu_percentiles(self) -> dict[str, float]:
        return {
            "utilization_p50": self._gpu_tracker.p50,
            "utilization_p95": self._gpu_tracker.p95,
            "utilization_p99": self._gpu_tracker.p99,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get a complete metrics summary for dashboards."""
        summary: dict[str, Any] = {
            "throughput": self.get_throughput(),
            "gpu": self.get_gpu_percentiles(),
            "models": {},
        }
        for model, tracker in self._latency_trackers.items():
            summary["models"][model] = {
                "p50": tracker.p50,
                "p95": tracker.p95,
                "p99": tracker.p99,
                "count": tracker.count,
            }
        return summary

    def export_prometheus(self) -> bytes:
        """Export metrics in Prometheus text format."""
        if HAS_PROMETHEUS:
            return generate_latest(REGISTRY)
        return b""

# ── Singleton ──────────────────────────────────────────────────────

_metrics: MetricsCollector | None = None

def get_metrics() -> MetricsCollector:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _metrics
    if _metrics is not None:
        return _metrics
    _metrics = MetricsCollector()
    return _metrics