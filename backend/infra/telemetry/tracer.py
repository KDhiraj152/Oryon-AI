"""
Distributed Tracing — OpenTelemetry Integration
==================================================

Provides span-based distributed tracing with automatic context propagation.
Falls back to no-op implementations when OpenTelemetry is not installed.

Design:
  - Zero-cost no-op when tracing is disabled
  - Automatic parent-child span relationships
  - Attribute injection for model, hardware, and request metadata
  - Compatible with Jaeger, Zipkin, OTLP backends

Usage:
    from backend.infra.telemetry.tracer import get_tracer, trace_span

    tracer = get_tracer(__name__)

    async def inference(request):
        with tracer.span("model.inference", attributes={"model": "qwen3-8b"}) as span:
            result = await run_model(request)
            span.set_attribute("tokens", result.token_count)
            return result

    # Or as decorator:
    @trace_span("embedding.generate")
    async def generate_embedding(text: str) -> list[float]:
        ...
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from backend.infra.telemetry.logger import get_logger

logger = get_logger(__name__)

# ── Try importing OpenTelemetry ────────────────────────────────────

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation import set_span_in_context

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

# ── No-Op Implementations ─────────────────────────────────────────

class NoOpSpan:
    """Zero-cost span when tracing is disabled."""

    __slots__ = ("attributes", "name", "start_time")

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = time.monotonic()
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: Any, description: str | None = None) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def record_exception(self, exception: BaseException, **kwargs: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end()

# ── Tracer ─────────────────────────────────────────────────────────

class Tracer:
    """
    Unified tracing interface.

    Wraps OpenTelemetry when available, falls back to no-ops.
    Provides consistent API regardless of tracing backend.
    """

    def __init__(self, name: str, *, enabled: bool = True):
        self._name = name
        self._enabled = enabled and HAS_OTEL
        self._tracer = None

        if self._enabled:
            self._tracer = otel_trace.get_tracer(name)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        kind: str = "internal",
    ) -> Generator[Any, None, None]:
        """
        Create a traced span.

        Args:
            name: Span name (e.g., "model.inference", "db.query")
            attributes: Initial span attributes
            kind: Span kind — "internal", "server", "client", "producer", "consumer"
        """
        if not self._enabled or self._tracer is None:
            noop_span = NoOpSpan(name)
            if attributes:
                noop_span.attributes.update(attributes)
            yield noop_span
            return

        otel_kind = {
            "internal": otel_trace.SpanKind.INTERNAL,
            "server": otel_trace.SpanKind.SERVER,
            "client": otel_trace.SpanKind.CLIENT,
            "producer": otel_trace.SpanKind.PRODUCER,
            "consumer": otel_trace.SpanKind.CONSUMER,
        }.get(kind, otel_trace.SpanKind.INTERNAL)

        with self._tracer.start_as_current_span(
            name, kind=otel_kind, attributes=attributes
        ) as span:
            try:
                yield span
            except (RuntimeError, OSError) as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def current_span(self) -> Any:
        """Get the current active span."""
        if self._enabled:
            return otel_trace.get_current_span()
        return NoOpSpan()

# ── Global Tracer Registry ────────────────────────────────────────

_tracers: dict[str, Tracer] = {}
_tracing_enabled: bool = False

def init_tracing(
    *,
    service_name: str = "oryon-ai",
    enabled: bool = True,
    exporter: str = "console",
    endpoint: str | None = None,
    configure_provider: bool = False,
) -> None:
    """
    Initialize the tracing system. Call once at application startup.

    If ``core.tracing.init_tracing()`` has already configured an
    OpenTelemetry TracerProvider, pass ``configure_provider=False``
    (the default) to reuse that provider. Set ``configure_provider=True``
    only when this module is the sole tracing initializer.

    Args:
        service_name: Service name for span attribution
        enabled: Enable/disable tracing globally
        exporter: Exporter type — "console", "otlp", "jaeger"
        endpoint: Exporter endpoint URL
        configure_provider: If True, set up a new TracerProvider.
            If False, just enable the global flag so Tracer instances
            pick up the already-configured provider.
    """
    global _tracing_enabled
    _tracing_enabled = enabled

    if not enabled or not HAS_OTEL:
        if not HAS_OTEL:
            logger.info("opentelemetry_not_installed", msg="Tracing disabled (install opentelemetry-sdk)")
        return

    # When core.tracing already configured the provider, skip re-configuration.
    if not configure_provider:
        logger.info(
            "tracing_layer_enabled",
            msg="Telemetry layer tracing enabled (using existing provider)",
        )
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp" and endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
        except ImportError:
            logger.warning("otlp_exporter_not_installed")
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    otel_trace.set_tracer_provider(provider)
    logger.info("tracing_initialized", exporter=exporter, service=service_name)

def get_tracer(name: str) -> Tracer:
    """Get or create a tracer for the given module."""
    if name not in _tracers:
        _tracers[name] = Tracer(name, enabled=_tracing_enabled)
    return _tracers[name]

# ── Decorator ──────────────────────────────────────────────────────

def trace_span(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to trace a function call.

    @trace_span("model.generate")
    async def generate(prompt: str) -> str:
        ...
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        tracer = get_tracer(func.__module__ or __name__)

        if asyncio_iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.span(span_name, attributes=attributes):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.span(span_name, attributes=attributes):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator

def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if function is async without importing asyncio at module level."""
    import asyncio

    return asyncio.iscoroutinefunction(func)
