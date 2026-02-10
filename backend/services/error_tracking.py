"""
Error Tracking and Monitoring Integration

Issue: CODE-REVIEW-GPT #9 (HIGH)
Problem: No error tracking system (Sentry SDK installed but not configured)

Solution: Comprehensive Sentry integration with context capture
"""

import logging
import os
from functools import wraps
from typing import Any, Dict, Optional

# Safe import of sentry_sdk - gracefully handle if not installed
try:
    import sentry_sdk
    from sentry_sdk.integrations.celery import CeleryIntegration
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.redis import RedisIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    sentry_sdk = None

from ..core.config import get_settings

logger = logging.getLogger(__name__)

# Lazy settings access to avoid module-level instantiation
settings = None


def _get_settings():
    global settings
    if settings is None:
        settings = get_settings()
    return settings


def init_sentry():
    """Initialize Sentry error tracking."""

    # Check if sentry_sdk is available
    if not SENTRY_AVAILABLE:
        logger.warning("sentry-sdk not installed. Run: pip install sentry-sdk[fastapi]")
        return

    sentry_dsn = os.getenv("SENTRY_DSN")
    environment = _get_settings().ENVIRONMENT

    if not sentry_dsn:
        logger.warning("SENTRY_DSN not set, error tracking disabled")
        return

    # Sentry configuration
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=environment,
        # Integrations
        integrations=[
            FastApiIntegration(transaction_style="url"),
            SqlalchemyIntegration(),
            CeleryIntegration(),
            RedisIntegration(),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        # Performance monitoring
        traces_sample_rate=0.1 if environment == "production" else 1.0,
        profiles_sample_rate=0.1 if environment == "production" else 1.0,
        # Error sampling
        sample_rate=1.0,
        # Release tracking
        release=os.getenv("APP_VERSION", "dev"),
        # Additional options
        attach_stacktrace=True,
        send_default_pii=False,  # Don't send PII
        max_breadcrumbs=50,
        # Before send hook
        before_send=before_send_hook,
    )

    logger.info(f"Sentry initialized for environment: {environment}")


def before_send_hook(
    event: dict[str, Any], hint: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Filter and modify events before sending to Sentry.

    Use this to:
    - Filter out sensitive data
    - Ignore certain error types
    - Add custom context
    """

    # Ignore specific errors
    if "exc_info" in hint:
        exc_type, exc_value, _ = hint["exc_info"]

        # Ignore common/expected errors
        ignored_exceptions = (
            "ConnectionError",
            "TimeoutError",
            "HTTPException",  # FastAPI HTTP exceptions
        )

        if exc_type.__name__ in ignored_exceptions:
            # Only log to Sentry if it's a server error (5xx)
            if hasattr(exc_value, "status_code") and exc_value.status_code < 500:
                return None

    # Filter sensitive data from request
    if "request" in event:
        request = event["request"]

        # Remove sensitive headers
        if "headers" in request:
            sensitive_headers = ["authorization", "cookie", "x-api-key"]
            for header in sensitive_headers:
                if header in request["headers"]:
                    request["headers"][header] = "[FILTERED]"

        # Remove sensitive query params
        if "query_string" in request:
            sensitive_params = ["password", "token", "api_key", "secret"]
            # Simple filtering (production should use proper URL parsing)
            for param in sensitive_params:
                if param in str(request["query_string"]).lower():
                    request["query_string"] = "[FILTERED]"

    return event


def capture_exception(
    error: Exception,
    context: dict[str, Any] | None = None,
    level: str = "error",
    tags: dict[str, str] | None = None,
) -> str | None:
    """
    Capture exception with additional context.

    Args:
        error: Exception to capture
        context: Additional context data
        level: Error level (error, warning, info)
        tags: Custom tags for filtering

    Returns:
        Event ID from Sentry
    """
    with sentry_sdk.push_scope() as scope:
        # Set level
        scope.level = level

        # Add context
        if context:
            for key, value in context.items():
                scope.set_context(key, value)

        # Add tags
        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)

        # Capture exception
        event_id = sentry_sdk.capture_exception(error)
        logger.error(f"Exception captured with Sentry ID: {event_id}")
        return event_id


def capture_message(
    message: str,
    level: str = "info",
    context: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str | None:
    """
    Capture message with context.

    Args:
        message: Message to capture
        level: Message level
        context: Additional context
        tags: Custom tags

    Returns:
        Event ID from Sentry
    """
    with sentry_sdk.push_scope() as scope:
        scope.level = level

        if context:
            for key, value in context.items():
                scope.set_context(key, value)

        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)

        return sentry_sdk.capture_message(message, level=level)


def add_breadcrumb(
    message: str,
    category: str = "default",
    level: str = "info",
    data: dict[str, Any] | None = None,
):
    """
    Add breadcrumb for debugging context.

    Args:
        message: Breadcrumb message
        category: Category (api, db, cache, etc.)
        level: Breadcrumb level
        data: Additional data
    """
    sentry_sdk.add_breadcrumb(
        message=message, category=category, level=level, data=data or {}
    )


def set_user_context(
    user_id: str | None = None,
    email: str | None = None,
    username: str | None = None,
    ip_address: str | None = None,
):
    """Set user context for error tracking."""
    sentry_sdk.set_user(
        {"id": user_id, "email": email, "username": username, "ip_address": ip_address}
    )


def set_tag(key: str, value: str):
    """Set custom tag for filtering."""
    sentry_sdk.set_tag(key, value)


def set_context(name: str, context: dict[str, Any]):
    """Set custom context."""
    sentry_sdk.set_context(name, context)


# Decorator for automatic error capture
def monitor_errors(
    operation: str, capture_args: bool = False, tags: dict[str, str] | None = None
):
    """
    Decorator to automatically capture errors.

    Usage:
        @monitor_errors("content_processing", capture_args=True)
        def process_content(content_id):
            ...
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                add_breadcrumb(
                    message=f"Starting {operation}", category="operation", level="info"
                )
                result = await func(*args, **kwargs)
                add_breadcrumb(
                    message=f"Completed {operation}", category="operation", level="info"
                )
                return result
            except Exception as e:
                context = {"operation": operation}
                if capture_args:
                    context["args"] = str(args)[:500]  # Limit size
                    context["kwargs"] = str(kwargs)[:500]

                capture_exception(e, context=context, tags=tags or {})
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                add_breadcrumb(
                    message=f"Starting {operation}", category="operation", level="info"
                )
                result = func(*args, **kwargs)
                add_breadcrumb(
                    message=f"Completed {operation}", category="operation", level="info"
                )
                return result
            except Exception as e:
                context = {"operation": operation}
                if capture_args:
                    context["args"] = str(args)[:500]
                    context["kwargs"] = str(kwargs)[:500]

                capture_exception(e, context=context, tags=tags or {})
                raise

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Performance monitoring
class PerformanceMonitor:
    """Context manager for performance monitoring."""

    def __init__(self, operation: str, op_type: str = "function"):
        self.operation = operation
        self.op_type = op_type
        self.transaction = None

    def __enter__(self):
        self.transaction = sentry_sdk.start_transaction(
            op=self.op_type, name=self.operation
        )
        self.transaction.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.transaction.set_status("internal_error")
        else:
            self.transaction.set_status("ok")
        self.transaction.__exit__(exc_type, exc_val, exc_tb)

    def add_span(self, op: str, description: str):
        """Add a span to the transaction."""
        return self.transaction.start_child(op=op, description=description)

    def span(self, description: str):
        """Create a span context manager."""
        return SpanContext(self, description)


class SpanContext:
    """Context manager for spans within a performance monitor."""

    def __init__(self, monitor: PerformanceMonitor, description: str):
        self.monitor = monitor
        self.description = description
        self.span = None

    def __enter__(self):
        if self.monitor.transaction:
            self.span = self.monitor.transaction.start_child(
                op="span", description=self.description
            )
            self.span.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.span.__exit__(exc_type, exc_val, exc_tb)


# Health check for Sentry
def sentry_health_check() -> dict[str, Any]:
    """Check if Sentry is properly configured."""
    try:
        # Send a test message
        event_id = capture_message("Sentry health check", level="info")
        return {
            "status": "healthy",
            "configured": True,
            "environment": _get_settings().ENVIRONMENT,
            "test_event_id": event_id,
        }
    except Exception as e:
        return {"status": "unhealthy", "configured": False, "error": str(e)}
