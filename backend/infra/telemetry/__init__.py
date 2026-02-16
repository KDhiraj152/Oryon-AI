"""
Telemetry Layer — Unified Observability
========================================

LAYER 6 in the architecture stack. All other layers depend on this.

Provides:
  - Structured logging with correlation IDs
  - Distributed tracing (OpenTelemetry)
  - Metrics collection (Prometheus)
  - GPU/hardware utilization tracking
  - Token throughput accounting

Usage:
    from backend.infra.telemetry import get_logger, trace, metrics

    logger = get_logger(__name__)
    with trace.span("my_operation") as span:
        span.set_attribute("model", "qwen3-8b")
        metrics.inference_latency.observe(0.42)
        logger.info("completed", extra={"tokens": 128})
"""

from backend.infra.telemetry.dashboard import register_stats_provider
from backend.infra.telemetry.logger import StructuredLogger, get_logger
from backend.infra.telemetry.metrics import MetricsCollector, get_metrics
from backend.infra.telemetry.profiler import PerformanceProfiler, get_profiler
from backend.infra.telemetry.tracer import Tracer, get_tracer, trace_span

__all__ = [
    "MetricsCollector",
    "PerformanceProfiler",
    "StructuredLogger",
    "Tracer",
    "get_logger",
    "get_metrics",
    "get_profiler",
    "get_tracer",
    "register_stats_provider",
    "trace_span",
]
