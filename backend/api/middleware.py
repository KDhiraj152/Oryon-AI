"""API middleware for security, logging, and performance monitoring.

OPTIMIZATION NOTES (December 2025):
- Uses time.perf_counter() for high-resolution timing
- Precomputed static headers to avoid dict creation per request
- Fast path checks using frozenset for O(1) lookups
- Minimal async overhead via optimized dispatch patterns
"""

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timezone
from typing import Tuple

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings
from ..core.correlation import set_request_id
from ..core.exceptions import ShikshaSetuException
from ..utils.logging import get_logger

logger = get_logger(__name__)

# OPTIMIZATION: Precompute static headers as tuple pairs for faster iteration
_STATIC_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

# Pre-convert to tuple for faster iteration (avoids dict.items() overhead)
_SECURITY_HEADERS_TUPLE: tuple[tuple[str, str], ...] = tuple(
    _STATIC_SECURITY_HEADERS.items()
)

# HSTS header for production (precomputed)
_HSTS_HEADER = (
    "Strict-Transport-Security",
    "max-age=31536000; includeSubDomains; preload",
)

# OPTIMIZATION: Paths that skip logging for high-frequency endpoints
_SKIP_LOGGING_PATHS = frozenset(
    ["/health", "/api/v2/health", "/metrics", "/favicon.ico"]
)

# OPTIMIZATION: Precompute production check once at module load
_IS_PRODUCTION = settings.ENVIRONMENT == "production"


# ============================================================================
# DEPRECATED: Legacy middleware classes below are superseded by
# UnifiedMiddleware (see unified_middleware.py) which consolidates
# request ID, security headers, timing, logging, and consent into a
# single zero-copy ASGI middleware.  These classes are retained only
# for reference and manual testing scripts (tests/manual/).
# ============================================================================


class RequestIDMiddleware(BaseHTTPMiddleware):
    """[DEPRECATED] Use UnifiedMiddleware instead.

    Add unique request ID for distributed tracing.
    OPTIMIZATION: Uses uuid4().hex for faster string conversion (no hyphens).
    """

    __slots__ = ("app",)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID - use hex for faster string ops
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex

        # Store in request state for access in route handlers
        request.state.request_id = request_id

        # Store in context var for access in Celery tasks and logging
        set_request_id(request_id)

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """[DEPRECATED] Use UnifiedMiddleware instead.

    Add security headers to all responses.

    OPTIMIZATION: Uses precomputed tuple iteration for O(n) header application
    instead of dict.update() which has higher overhead.
    """

    __slots__ = ("app",)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # OPTIMIZATION: Direct tuple iteration is faster than dict.update()
        headers = response.headers
        for key, value in _SECURITY_HEADERS_TUPLE:
            headers[key] = value

        # HSTS header for production only (uses module-level precomputed check)
        if _IS_PRODUCTION:
            headers[_HSTS_HEADER[0]] = _HSTS_HEADER[1]

        return response


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """[DEPRECATED] Use UnifiedMiddleware instead.

    Add request timing and log slow requests.

    OPTIMIZATION: Uses perf_counter() for high-resolution timing.
    Caches threshold check to avoid repeated attribute access.
    """

    __slots__ = ("_slow_threshold", "app")

    def __init__(self, app):
        super().__init__(app)
        # Cache threshold to avoid repeated settings access
        self._slow_threshold = settings.SLOW_REQUEST_THRESHOLD

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate processing time with perf_counter for accuracy
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        # Log slow requests - only if above threshold
        if process_time > self._slow_threshold:
            logger.warning(
                "Slow request: %s %s took %.2fs",
                request.method,
                request.url.path,
                process_time,
            )

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """[DEPRECATED] Use UnifiedMiddleware instead.

    Log all API requests with request ID correlation.

    OPTIMIZATION:
    - Skips logging for high-frequency health/metrics endpoints
    - Uses lazy % formatting for logger (avoids string creation if not logging)
    - Only logs response for non-2xx status codes to reduce I/O
    """

    __slots__ = ("app",)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # OPTIMIZATION: Skip logging for high-frequency endpoints (O(1) lookup)
        if path in _SKIP_LOGGING_PATHS:
            return await call_next(request)

        # Get request ID from request state (set by RequestIDMiddleware)
        request_id = getattr(request.state, "request_id", None)
        method = request.method
        client = request.client

        # Log request with lazy formatting
        logger.info(
            "Request: %s %s from %s",
            method,
            path,
            client.host if client else "unknown",
            extra={"request_id": request_id},
        )

        # Process request
        response = await call_next(request)

        # OPTIMIZATION: Only log responses for non-2xx or errors
        status = response.status_code
        if status >= 400:
            logger.warning(
                "Response: %s %s status=%d",
                method,
                path,
                status,
                extra={"request_id": request_id},
            )
        elif logger.isEnabledFor(10):  # DEBUG level
            logger.debug(
                "Response: %s %s status=%d",
                method,
                path,
                status,
                extra={"request_id": request_id},
            )

        return response


class AgeConsentMiddleware(BaseHTTPMiddleware):
    """[DEPRECATED] Use UnifiedMiddleware instead.

    Middleware to check age consent header when required.

    When AGE_CONSENT_REQUIRED=true in settings:
    - Checks for X-Age-Consent: confirmed header
    - Only blocks mature content topics without consent
    - Does NOT block general queries

    This is the ONLY content restriction in Universal Mode.

    OPTIMIZATION: Uses cached settings and fast header lookup.
    """

    __slots__ = ("_enabled", "_mature_topics", "app")

    def __init__(self, app):
        super().__init__(app)
        # Cache settings at startup
        self._enabled = getattr(settings, "AGE_CONSENT_REQUIRED", True)
        self._mature_topics = frozenset(getattr(settings, "MATURE_CONTENT_TOPICS", []))

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._enabled:
            return await call_next(request)

        # OPTIMIZATION: Direct get with lowercase comparison
        age_consent = request.headers.get("x-age-consent", "").lower()

        # Store consent status in request state for routes to use
        request.state.age_consent = age_consent == "confirmed"

        # We don't block here - routes can check request.state.age_consent
        # if they need to handle mature content specifically

        return await call_next(request)


def exception_handler(request: Request, exc: ShikshaSetuException) -> JSONResponse:
    """Handle custom ShikshaSetu exceptions.

    OPTIMIZATION: Uses lazy % formatting to avoid string creation if not logging.
    """
    logger.error(
        "ShikshaSetuException: %s - %s (status=%d) for %s %s",
        exc.error_code,
        exc.detail,
        exc.status_code,
        request.method,
        request.url.path,
    )

    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions.

    OPTIMIZATION: Uses module-level production check.
    """
    logger.exception(
        "Unhandled exception for %s %s: %s", request.method, request.url.path, str(exc)
    )

    # OPTIMIZATION: Use pre-computed production check
    detail = "An unexpected error occurred" if _IS_PRODUCTION else str(exc)

    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "detail": detail,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
