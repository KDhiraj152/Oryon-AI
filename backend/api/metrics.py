"""Prometheus metrics endpoint for FastAPI monitoring."""

import logging
import time
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.routing import APIRoute
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

# Define metrics
http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["method", "endpoint"],
)

# ML model metrics
ml_inference_duration_seconds = Histogram(
    "ml_inference_duration_seconds",
    "ML model inference duration",
    ["model", "operation"],
)

ml_inference_total = Counter(
    "ml_inference_total", "Total ML inferences", ["model", "operation", "status"]
)

# Celery task metrics
celery_task_duration_seconds = Histogram(
    "celery_task_duration_seconds", "Celery task duration", ["task_name"]
)

celery_tasks_total = Counter(
    "celery_tasks_total", "Total Celery tasks", ["task_name", "status"]
)

# Database metrics
db_query_duration_seconds = Histogram(
    "db_query_duration_seconds", "Database query duration", ["operation"]
)

db_connections_active = Gauge(
    "db_connections_active", "Number of active database connections"
)

# Cache metrics
cache_hits_total = Counter("cache_hits_total", "Total cache hits", ["cache_type"])

cache_misses_total = Counter("cache_misses_total", "Total cache misses", ["cache_type"])

cache_operations_duration_seconds = Histogram(
    "cache_operations_duration_seconds",
    "Cache operation duration",
    ["operation", "cache_type"],
)


class MetricsRoute(APIRoute):
    """Custom route class that records metrics for each request."""

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            # Extract route info
            method = request.method
            endpoint = request.url.path

            # Track in-progress requests
            http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()

            # Time the request
            start_time = time.time()

            try:
                response = await original_route_handler(request)
                status = response.status_code

                # Record metrics
                http_requests_total.labels(
                    method=method, endpoint=endpoint, status=status
                ).inc()

                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method, endpoint=endpoint
                ).observe(duration)

                return response

            except Exception:
                # Record error
                http_requests_total.labels(
                    method=method, endpoint=endpoint, status=500
                ).inc()
                raise

            finally:
                # Decrement in-progress counter
                http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()

        return custom_route_handler


def metrics_endpoint() -> Response:
    """Expose Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def track_ml_inference(model: str, operation: str):
    """Context manager to track ML inference metrics."""
    from contextlib import contextmanager

    @contextmanager
    def tracker():
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time

            ml_inference_duration_seconds.labels(
                model=model, operation=operation
            ).observe(duration)

            ml_inference_total.labels(
                model=model, operation=operation, status=status
            ).inc()

    return tracker()


__all__ = [
    "MetricsRoute",
    "cache_hits_total",
    "cache_misses_total",
    "cache_operations_duration_seconds",
    "celery_tasks_total",
    "http_requests_total",
    "metrics_endpoint",
    "ml_inference_total",
    "track_ml_inference",
]
