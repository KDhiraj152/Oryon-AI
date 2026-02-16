"""Unit tests for custom exceptions."""

import pytest

from backend.core.exceptions import (
    AuthenticationError,
    ContentNotFoundError,
    RateLimitError,
    OryonException,
    ValidationError,
)


def test_base_exception():
    """Test base OryonException."""
    exc = OryonException("Test error", status_code=500, error_code="TEST_ERROR")

    assert exc.detail == "Test error"
    assert exc.status_code == 500
    assert exc.error_code == "TEST_ERROR"
    assert exc.timestamp is not None

    exc_dict = exc.to_dict()
    assert exc_dict["error"] == "TEST_ERROR"
    assert exc_dict["detail"] == "Test error"
    assert exc_dict["status_code"] == 500


def test_content_not_found_error():
    """Test ContentNotFoundError."""
    exc = ContentNotFoundError("test-content-id")

    assert exc.status_code == 404
    assert exc.error_code == "CONTENT_NOT_FOUND"
    assert "test-content-id" in exc.detail


def test_authentication_error():
    """Test AuthenticationError."""
    exc = AuthenticationError()

    assert exc.status_code == 401
    assert exc.error_code == "AUTHENTICATION_FAILED"

    exc_custom = AuthenticationError("Invalid token")
    assert "Invalid token" in exc_custom.detail


def test_validation_error():
    """Test ValidationError."""
    exc = ValidationError("Invalid input format")

    assert exc.status_code == 422
    assert exc.error_code == "VALIDATION_ERROR"
    assert "Invalid input" in exc.detail


def test_rate_limit_error():
    """Test RateLimitError."""
    exc = RateLimitError()

    assert exc.status_code == 429
    assert exc.error_code == "RATE_LIMIT_EXCEEDED"
