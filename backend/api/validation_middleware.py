"""
Request Validation Middleware
=============================

Provides consistent validation error formatting across all endpoints.
Integrates with Pydantic validation errors for standardized responses.

Features:
- Consistent error format for all validation errors
- Field-level error details
- Request body validation
- Query parameter validation
- Path parameter validation
"""

from collections.abc import Callable
from datetime import UTC, datetime, timezone
from typing import Any

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logging import get_logger

logger = get_logger(__name__)

class ValidationErrorDetail:
    """Structured validation error detail."""

    def __init__(
        self,
        field: str,
        message: str,
        error_type: str,
        value: Any | None = None,
        constraint: str | None = None,
    ):
        self.field = field
        self.message = message
        self.error_type = error_type
        self.value = value
        self.constraint = constraint

    def to_dict(self) -> dict[str, Any]:
        result = {"field": self.field, "message": self.message, "type": self.error_type}
        if self.value is not None:
            # Truncate long values
            str_value = str(self.value)
            result["value"] = (
                str_value[:100] + "..." if len(str_value) > 100 else str_value
            )
        if self.constraint:
            result["constraint"] = self.constraint
        return result

class ValidationErrorResponse:
    """Standardized validation error response."""

    def __init__(
        self,
        message: str = "Validation failed",
        errors: list[ValidationErrorDetail] | None = None,
        request_id: str | None = None,
    ):
        self.message = message
        self.errors = errors or []
        self.request_id = request_id
        self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "VALIDATION_ERROR",
            "message": self.message,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "details": [e.to_dict() for e in self.errors],
            "error_count": len(self.errors),
        }

def parse_pydantic_errors(errors: list[dict[str, Any]]) -> list[ValidationErrorDetail]:
    """Parse Pydantic validation errors into structured format."""
    details = []

    for error in errors:
        # Get field path
        loc = error.get("loc", [])
        field = ".".join(
            str(loc_part)
            for loc_part in loc
            if loc_part not in ("body", "query", "path")
        )
        if not field:
            field = "request"

        # Get error message
        msg = error.get("msg", "Invalid value")

        # Get error type
        error_type = error.get("type", "value_error")

        # Get constraint info if available
        constraint = None
        ctx = error.get("ctx", {})
        if "limit_value" in ctx:
            constraint = f"limit: {ctx['limit_value']}"
        elif "pattern" in ctx:
            constraint = f"pattern: {ctx['pattern']}"
        elif "min_length" in ctx:
            constraint = f"min_length: {ctx['min_length']}"
        elif "max_length" in ctx:
            constraint = f"max_length: {ctx['max_length']}"

        # Get input value (sanitized)
        value = error.get("input")

        details.append(
            ValidationErrorDetail(
                field=field,
                message=msg,
                error_type=error_type,
                value=value,
                constraint=constraint,
            )
        )

    return details

def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle FastAPI validation errors with consistent format.

    Returns:
        JSONResponse with structured validation errors
    """
    request_id = getattr(request.state, "request_id", None)

    # Parse errors
    errors = parse_pydantic_errors(exc.errors())

    # Create response
    response = ValidationErrorResponse(
        message=f"Request validation failed with {len(errors)} error(s)",
        errors=errors,
        request_id=request_id,
    )

    # Log validation error
    logger.warning(
        f"Validation error on {request.method} {request.url.path}: "
        f"{len(errors)} errors",
        extra={
            "request_id": request_id,
            "errors": [e.to_dict() for e in errors[:5]],  # Log first 5
        },
    )

    return JSONResponse(status_code=422, content=response.to_dict())

def pydantic_validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors (from manual validation).
    """
    request_id = getattr(request.state, "request_id", None)

    errors = parse_pydantic_errors(exc.errors())

    response = ValidationErrorResponse(
        message=f"Data validation failed with {len(errors)} error(s)",
        errors=errors,
        request_id=request_id,
    )

    return JSONResponse(status_code=422, content=response.to_dict())

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add consistent validation context.

    This middleware:
    - Adds request context for validation errors
    - Logs validation failures
    - Tracks validation metrics

    OPTIMIZATION: Uses __slots__ to reduce memory overhead per instance.
    """

    __slots__ = ("_total_requests", "_validation_errors", "app", "track_metrics")

    def __init__(self, app, track_metrics: bool = True):
        super().__init__(app)
        self.track_metrics = track_metrics
        self._validation_errors = 0
        self._total_requests = 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        self._total_requests += 1

        try:
            response = await call_next(request)

            # Track validation errors from response status
            if response.status_code == 422:
                self._validation_errors += 1

            return response

        except RequestValidationError as exc:
            self._validation_errors += 1
            return await validation_exception_handler(request, exc)

    @property
    def validation_error_rate(self) -> float:
        """Get validation error rate."""
        if self._total_requests == 0:
            return 0.0
        return self._validation_errors / self._total_requests

def register_validation_handlers(app) -> None:
    """
    Register validation exception handlers with FastAPI app.

    Usage:
        from backend.api.validation_middleware import register_validation_handlers
        register_validation_handlers(app)
    """
    from fastapi.exceptions import RequestValidationError
    from pydantic import ValidationError

    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_handler)

# Export
__all__ = [
    "RequestValidationMiddleware",
    "ValidationErrorDetail",
    "ValidationErrorResponse",
    "parse_pydantic_errors",
    "pydantic_validation_handler",
    "register_validation_handlers",
    "validation_exception_handler",
]
