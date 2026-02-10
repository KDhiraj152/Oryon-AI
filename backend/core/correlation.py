"""
Correlation ID Middleware and Logging Context.

Provides request tracing across the entire pipeline with:
- Unique request IDs for every request
- Propagation through all pipeline stages
- Structured logging with correlation context
"""

import logging
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, Optional, TypeVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Context variable for request ID - accessible anywhere in the async context
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
request_context_var: ContextVar[dict[str, Any] | None] = ContextVar(
    "request_context", default=None
)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# CONTEXT MANAGEMENT
# =============================================================================


def get_request_id() -> str | None:
    """Get current request ID from context."""
    return request_id_var.get()


def get_request_context() -> dict[str, Any]:
    """Get current request context."""
    ctx = request_context_var.get()
    return ctx if ctx is not None else {}


def set_request_id(request_id: str) -> str:
    """Set request ID in context."""
    request_id_var.set(request_id)
    return request_id


def set_request_context(context: dict[str, Any]):
    """Set request context."""
    request_context_var.set(context)


def generate_request_id() -> str:
    """Generate a new unique request ID."""
    return str(uuid.uuid4())[:12]  # Short but unique enough


# =============================================================================
# CORRELATION LOGGING FILTER
# =============================================================================


class CorrelationLogFilter(logging.Filter):
    """
    Logging filter that adds request_id to all log records.

    Usage:
        handler.addFilter(CorrelationLogFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        record.context = request_context_var.get() or {}
        return True


class CorrelationLogFormatter(logging.Formatter):
    """
    Formatter that includes request_id in log output.

    Example output:
        2025-12-02 10:30:00 | INFO | a1b2c3d4 | [pipeline] Processing started
    """

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        default_fmt = (
            "%(asctime)s | %(levelname)s | %(request_id)s | %(name)s | %(message)s"
        )
        super().__init__(fmt or default_fmt, datefmt or "%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return super().format(record)


def setup_correlation_logging(logger_name: str | None = None):
    """
    Setup correlation ID logging for the application.

    Args:
        logger_name: Specific logger to configure, or None for root
    """
    logger = logging.getLogger(logger_name)

    # Add filter to all handlers
    for handler in logger.handlers:
        handler.addFilter(CorrelationLogFilter())
        if not isinstance(handler.formatter, CorrelationLogFormatter):
            handler.setFormatter(CorrelationLogFormatter())

    # If no handlers, add a default one
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(CorrelationLogFilter())
        handler.setFormatter(CorrelationLogFormatter())
        logger.addHandler(handler)


# =============================================================================
# FASTAPI MIDDLEWARE
# =============================================================================


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that adds correlation ID to every request.

    Usage:
        app.add_middleware(CorrelationIdMiddleware)
    """

    HEADER_NAME = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Get or generate request ID
        request_id = request.headers.get(self.HEADER_NAME) or generate_request_id()

        # Set in context
        set_request_id(request_id)
        set_request_context(
            {
                "path": request.url.path,
                "method": request.method,
                "client_ip": request.client.host if request.client else None,
            }
        )

        # Store in request state for handlers
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers[self.HEADER_NAME] = request_id

        return response


# =============================================================================
# DECORATOR FOR PROPAGATING CONTEXT
# =============================================================================


def with_correlation(func: F) -> F:
    """
    Decorator to ensure correlation ID is logged with function execution.

    Example:
        @with_correlation
        async def process_content(text: str) -> str:
            logger.info("Processing...")  # Will include request_id
            ...
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        request_id = get_request_id()
        logger = logging.getLogger(func.__module__)

        # Log entry
        logger.debug(
            f"Entering {func.__name__}",
            extra={"request_id": request_id, "args_count": len(args)},
        )

        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {e}", extra={"request_id": request_id}
            )
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # request_id available via get_request_id() if needed for logging
        logger = logging.getLogger(func.__module__)

        logger.debug(f"Entering {func.__name__}")

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise

    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# =============================================================================
# STRUCTURED LOG HELPER
# =============================================================================


class CorrelatedLogger:
    """
    Logger wrapper that automatically includes correlation context.

    Usage:
        log = CorrelatedLogger(__name__)
        log.info("Processing content", content_id="123", stage="simplification")
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, msg: str, **kwargs):
        request_id = get_request_id()
        context = get_request_context()

        extra_msg = ""
        if kwargs:
            extra_msg = " | " + " ".join(f"{k}={v}" for k, v in kwargs.items())

        self._logger.log(
            level,
            f"{msg}{extra_msg}",
            extra={"request_id": request_id, "context": context, **kwargs},
        )

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)

    def exception(self, msg: str, **kwargs):
        self._logger.exception(msg, extra={"request_id": get_request_id(), **kwargs})


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CorrelatedLogger",
    # Middleware
    "CorrelationIdMiddleware",
    # Logging
    "CorrelationLogFilter",
    "CorrelationLogFormatter",
    "generate_request_id",
    "get_request_context",
    "get_request_id",
    "request_context_var",
    # Context
    "request_id_var",
    "set_request_context",
    "set_request_id",
    "setup_correlation_logging",
    # Decorators
    "with_correlation",
]
