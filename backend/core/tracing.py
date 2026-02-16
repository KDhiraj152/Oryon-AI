"""
OpenTelemetry Tracing Configuration
====================================

Distributed tracing for request flow through all pipeline stages.
Integrates with Jaeger, Zipkin, or any OTLP-compatible backend.

Traces include:
- HTTP request/response
- Database queries
- Redis operations
- ML model inference
- Pipeline processing stages

Usage:
    from backend.core.tracing import tracer, trace_span

    @trace_span("process_content")
    async def process_content(text: str):
        with tracer.start_as_current_span("simplify") as span:
            span.set_attribute("text.length", len(text))
            result = await simplify(text)
        return result
"""

import functools
import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime, timezone
from typing import Any, TypeVar

from ..core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Check if OpenTelemetry is available
OTEL_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Span, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    OTEL_AVAILABLE = True
except ImportError:
    logger.info(
        "OpenTelemetry not installed. Tracing disabled. Install with: pip install opentelemetry-api opentelemetry-sdk"
    )

class NoOpSpan:
    """No-op span when tracing is disabled. All methods are intentionally empty/no-op."""

    def __init__(self, name: str = ""):
        self.name = name
        self._attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> "NoOpSpan":
        self._attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> "NoOpSpan":
        self._attributes.update(attributes)
        return self

    def add_event(
        self, _name: str, _attributes: dict[str, Any] | None = None
    ) -> "NoOpSpan":
        """No-op: Event logging disabled when tracing unavailable."""
        return self

    def record_exception(self, _exception: Exception) -> "NoOpSpan":
        """No-op: Exception recording disabled when tracing unavailable."""
        return self

    def set_status(self, _status: Any, _description: str | None = None) -> "NoOpSpan":
        """No-op: Status setting disabled when tracing unavailable."""
        return self

    def end(self) -> None:
        """No-op: Span already ended or tracing disabled."""
        # Intentionally empty - NoOp pattern for disabled tracing

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """No-op: Context exit when tracing disabled."""
        # Intentionally empty - NoOp pattern for disabled tracing

class NoOpTracer:
    """No-op tracer when OpenTelemetry is not available."""

    def start_as_current_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan(name)

    def start_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan(name)

    @contextmanager
    def start_as_current_span_context(self, name: str, **kwargs):
        yield NoOpSpan(name)

class TracingManager:
    """
    Manages OpenTelemetry tracing configuration and lifecycle.

    Supports multiple exporters:
    - Console (development)
    - OTLP (production - Jaeger, Zipkin, etc.)
    """

    _instance: "TracingManager | None" = None
    _initialized: bool = False

    def __new__(cls) -> "TracingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if TracingManager._initialized:
            return

        self._tracer: trace.Tracer | NoOpTracer = NoOpTracer()
        self._enabled = False
        TracingManager._initialized = True

    def initialize(
        self,
        service_name: str = "oryon-ai",
        environment: str = "production",
        otlp_endpoint: str | None = None,
        _sample_rate: float = 1.0,  # Reserved for future sampling configuration
    ) -> None:
        """
        Initialize OpenTelemetry tracing.

        Args:
            service_name: Name of the service for traces
            environment: Environment (production, staging)
            otlp_endpoint: OTLP collector endpoint (e.g., "http://localhost:4317")
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available. Using no-op tracer.")
            return

        try:
            # Create resource with service info
            resource = Resource.create(
                {
                    SERVICE_NAME: service_name,
                    "service.version": "2.0.0",
                    "deployment.environment": environment,
                    "service.instance.id": os.getenv("HOSTNAME", "local"),
                }
            )

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Add console exporter for development
            if environment == "development":
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(console_exporter))

            # Add OTLP exporter if endpoint provided
            if otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                    logger.info("OTLP exporter configured: %s", otlp_endpoint)
                except ImportError:
                    logger.warning(
                        "OTLP exporter not available. Install opentelemetry-exporter-otlp"
                    )

            # Set global tracer provider
            trace.set_tracer_provider(provider)

            # Set up context propagation
            set_global_textmap(TraceContextTextMapPropagator())

            # Get tracer
            self._tracer = trace.get_tracer(service_name, "2.0.0")
            self._enabled = True

            logger.info("OpenTelemetry tracing initialized for %s", service_name)

        except (RuntimeError, OSError) as e:
            logger.error("Failed to initialize tracing: %s", e)
            self._tracer = NoOpTracer()

    @property
    def tracer(self) -> "trace.Tracer" | NoOpTracer:
        """Get the tracer instance."""
        return self._tracer

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return bool(self._enabled)

    def get_current_span(self) -> "trace.Span" | NoOpSpan:
        """Get the current active span."""
        if not OTEL_AVAILABLE or not self._enabled:
            return NoOpSpan()
        return trace.get_current_span()

    def inject_context(self, carrier: dict[str, str]) -> None:
        """Inject trace context into a carrier (headers)."""
        if OTEL_AVAILABLE and self._enabled:
            from opentelemetry.propagate import inject

            inject(carrier)

    def extract_context(self, carrier: dict[str, str]):
        """Extract trace context from a carrier (headers)."""
        if OTEL_AVAILABLE and self._enabled:
            from opentelemetry.propagate import extract

            return extract(carrier)
        return None

# Global tracing manager instance
_tracing_manager: TracingManager | None = None

def get_tracing_manager() -> TracingManager:
    """Get or create the global tracing manager."""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager

def get_tracer() -> "trace.Tracer" | NoOpTracer:
    """Get the global tracer instance."""
    return get_tracing_manager().tracer

# Convenience alias
tracer = property(lambda self: get_tracer())

def _configure_span(
    span: "NoOpSpan | Any", func: Callable, attributes: dict[str, Any] | None
) -> None:
    """Configure span with function info and custom attributes."""
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    span.set_attribute("code.function", func.__name__)
    span.set_attribute("code.namespace", func.__module__)

def _finalize_span_success(span: "NoOpSpan | Any", start_time: datetime) -> None:
    """Finalize span on successful execution."""
    duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
    span.set_attribute("duration_ms", duration_ms)
    if OTEL_AVAILABLE:
        span.set_status(Status(StatusCode.OK))

def _finalize_span_error(
    span: "NoOpSpan | Any", exception: Exception, record_exception: bool
) -> None:
    """Finalize span on error."""
    if record_exception:
        span.record_exception(exception)
        if OTEL_AVAILABLE:
            span.set_status(Status(StatusCode.ERROR, str(exception)))

def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
) -> Callable:
    """
    Decorator to trace a function.

    Usage:
        @trace_span("process_document", {"doc.type": "pdf"})
        async def process_document(doc_id: str):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                _configure_span(span, func, attributes)
                try:
                    start_time = datetime.now(UTC)
                    result = await func(*args, **kwargs)
                    _finalize_span_success(span, start_time)
                    return result
                except Exception as e:
                    _finalize_span_error(span, e, record_exception)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                _configure_span(span, func, attributes)
                try:
                    start_time = datetime.now(UTC)
                    result = func(*args, **kwargs)
                    _finalize_span_success(span, start_time)
                    return result
                except Exception as e:
                    _finalize_span_error(span, e, record_exception)
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

def add_span_attributes(attributes: dict[str, Any]) -> None:
    """Add attributes to the current span."""
    span = get_tracing_manager().get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, value)

def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Add an event to the current span."""
    span = get_tracing_manager().get_current_span()
    if span:
        span.add_event(name, attributes or {})

# Initialize tracing on module load if enabled
def init_tracing():
    """Initialize tracing with settings from config."""
    manager = get_tracing_manager()

    # Get configuration from environment or settings
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    sample_rate = float(os.getenv("OTEL_SAMPLE_RATE", "1.0"))

    manager.initialize(
        service_name=getattr(settings, "APP_NAME", "oryon-ai"),
        environment=getattr(settings, "ENVIRONMENT", "development"),
        otlp_endpoint=otlp_endpoint,
        _sample_rate=sample_rate,
    )

# Export commonly used items
__all__ = [
    "OTEL_AVAILABLE",
    "TracingManager",
    "add_span_attributes",
    "add_span_event",
    "get_tracer",
    "get_tracing_manager",
    "init_tracing",
    "trace_span",
]
