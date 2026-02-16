"""
Unit Tests for Backend Core Components
======================================

Tests for:
- Circuit breakers
- Tracing
- Validation middleware
- API versioning
- Device router
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

# ==================== Circuit Breaker Tests ====================


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_creation(self):
        """Test circuit breaker initialization."""
        from backend.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=3, timeout_seconds=10, success_threshold=2
        )
        breaker = CircuitBreaker("test_service", config)

        assert breaker.name == "test_service"
        assert breaker.config.failure_threshold == 3
        assert breaker.stats.state.value == "closed"

    def test_circuit_breaker_registry(self):
        """Test circuit breaker registry."""
        from backend.core.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker("registry_test")

        # Should be in registry
        assert CircuitBreaker.get("registry_test") is breaker

        # Get all stats
        stats = CircuitBreaker.get_all_stats()
        assert "registry_test" in stats

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        from backend.core.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("success_test")

        async def successful_func():
            await asyncio.sleep(0)  # Minimal async operation
            return "success"

        result = await breaker.execute(successful_func)

        assert result == "success"
        assert breaker.stats.total_successes == 1
        assert breaker.stats.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        from backend.core.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitBreakerError,
            CircuitState,
        )

        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=60)
        breaker = CircuitBreaker("failure_test", config)

        async def failing_func():
            raise ValueError("Test failure")

        # First failure
        with pytest.raises(ValueError):
            await breaker.execute(failing_func)
        assert breaker.stats.state == CircuitState.CLOSED

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await breaker.execute(failing_func)
        assert breaker.stats.state == CircuitState.OPEN

        # Next call should be rejected
        with pytest.raises(CircuitBreakerError):
            await breaker.execute(failing_func)

    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics."""
        from backend.core.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker("stats_test")
        stats = breaker.stats.to_dict()

        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "total_failures" in stats

    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset."""
        from backend.core.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("reset_test")
        breaker.stats.failure_count = 10
        breaker.stats.state = CircuitState.OPEN

        breaker.reset()

        assert breaker.stats.state == CircuitState.CLOSED
        assert breaker.stats.failure_count == 0

    def test_predefined_breakers(self):
        """Test predefined circuit breaker factories."""
        from backend.core.circuit_breaker import (
            get_database_breaker,
            get_external_api_breaker,
            get_ml_breaker,
            get_redis_breaker,
        )

        db_breaker = get_database_breaker()
        assert db_breaker.name == "database"

        redis_breaker = get_redis_breaker()
        assert redis_breaker.name == "redis"

        ml_breaker = get_ml_breaker()
        assert ml_breaker.name == "ml_model"

        api_breaker = get_external_api_breaker()
        assert api_breaker.name == "external_api"


# ==================== Tracing Tests ====================


class TestTracing:
    """Test OpenTelemetry tracing functionality."""

    def test_noop_span(self):
        """Test NoOpSpan when tracing is disabled."""
        from backend.core.tracing import NoOpSpan

        span = NoOpSpan("test")

        # Should not raise
        span.set_attribute("key", "value")
        span.set_attributes({"a": 1, "b": 2})
        span.add_event("event")
        span.record_exception(ValueError("test"))
        span.set_status("OK")
        span.end()

        # Context manager should work
        with span:
            span.set_attribute("inside", True)  # Do something inside the context

    def test_noop_tracer(self):
        """Test NoOpTracer when OpenTelemetry is not available."""
        from backend.core.tracing import NoOpTracer

        tracer = NoOpTracer()

        span = tracer.start_as_current_span("test")
        assert span is not None

        span2 = tracer.start_span("test2")
        assert span2 is not None

    def test_tracing_manager_singleton(self):
        """Test TracingManager is singleton."""
        from backend.core.tracing import TracingManager

        manager1 = TracingManager()
        manager2 = TracingManager()

        assert manager1 is manager2

    def test_get_tracer(self):
        """Test get_tracer returns a tracer."""
        from backend.core.tracing import get_tracer

        tracer = get_tracer()
        assert tracer is not None

        # Should have start_as_current_span method
        assert hasattr(tracer, "start_as_current_span")

    @pytest.mark.asyncio
    async def test_trace_span_decorator(self):
        """Test trace_span decorator."""
        from backend.core.tracing import trace_span

        @trace_span("test_operation")
        async def test_func(x: int) -> int:
            return x * 2

        result = await test_func(5)
        assert result == 10

    def test_add_span_attributes(self):
        """Test adding span attributes."""
        from backend.core.tracing import add_span_attributes

        # Should not raise even without active span
        add_span_attributes({"key": "value", "count": 42})

    def test_add_span_event(self):
        """Test adding span events."""
        from backend.core.tracing import add_span_event

        # Should not raise even without active span
        add_span_event("test_event", {"detail": "value"})


# ==================== Validation Middleware Tests ====================


class TestValidationMiddleware:
    """Test validation middleware functionality."""

    def test_validation_error_detail(self):
        """Test ValidationErrorDetail structure."""
        from backend.api.validation_middleware import ValidationErrorDetail

        error = ValidationErrorDetail(
            field="email",
            message="Invalid email format",
            error_type="value_error",
            value="invalid",
            constraint="pattern: ^[a-z]+@[a-z]+\\.[a-z]+$",
        )

        result = error.to_dict()

        assert result["field"] == "email"
        assert result["message"] == "Invalid email format"
        assert result["type"] == "value_error"
        assert result["value"] == "invalid"
        assert "constraint" in result

    def test_validation_error_response(self):
        """Test ValidationErrorResponse structure."""
        from backend.api.validation_middleware import (
            ValidationErrorDetail,
            ValidationErrorResponse,
        )

        errors = [
            ValidationErrorDetail("field1", "error1", "type1"),
            ValidationErrorDetail("field2", "error2", "type2"),
        ]

        response = ValidationErrorResponse(
            message="Validation failed", errors=errors, request_id="req-123"
        )

        result = response.to_dict()

        assert result["error"] == "VALIDATION_ERROR"
        assert result["message"] == "Validation failed"
        assert result["request_id"] == "req-123"
        assert result["error_count"] == 2
        assert len(result["details"]) == 2

    def test_parse_pydantic_errors(self):
        """Test Pydantic error parsing."""
        from backend.api.validation_middleware import parse_pydantic_errors

        pydantic_errors = [
            {
                "loc": ["body", "email"],
                "msg": "Invalid email",
                "type": "value_error.email",
                "ctx": {},
            },
            {
                "loc": ["body", "age"],
                "msg": "Value must be >= 0",
                "type": "value_error.number.not_ge",
                "ctx": {"limit_value": 0},
            },
        ]

        details = parse_pydantic_errors(pydantic_errors)

        assert len(details) == 2
        assert details[0].field == "email"
        assert details[1].field == "age"
        assert "limit: 0" in details[1].constraint


# ==================== API Version Middleware Tests ====================


class TestAPIVersionMiddleware:
    """Test API versioning middleware."""

    def test_get_api_version_info(self):
        """Test API version info retrieval."""
        from backend.api.version_middleware import get_api_version_info

        info = get_api_version_info()

        assert "current_version" in info
        assert "versions" in info
        assert "v2" in info["versions"]

    def test_check_api_version_supported(self):
        """Test checking supported API version."""
        from backend.api.version_middleware import check_api_version

        result = check_api_version("v2")

        assert result["supported"] is True
        assert result["status"] == "current"

    def test_check_api_version_deprecated(self):
        """Test checking deprecated API version."""
        from backend.api.version_middleware import check_api_version

        result = check_api_version("v1")

        assert result["supported"] is True
        assert result["deprecated"] is True
        assert "sunset" in result

    def test_check_api_version_unsupported(self):
        """Test checking unsupported API version."""
        from backend.api.version_middleware import check_api_version

        result = check_api_version("v99")

        assert result["supported"] is False
        assert "recommended" in result


# ==================== Device Router Tests ====================


class TestDeviceRouter:
    """Test device router functionality."""

    def test_device_capabilities_aliases(self):
        """Test DeviceCapabilities alias properties."""
        from backend.core.optimized.device_router import DeviceCapabilities

        caps = DeviceCapabilities(memory_gb=16.0, mlx_available=True)

        # Test aliases - use pytest.approx for floating point comparison
        assert caps.unified_memory_gb == pytest.approx(16.0)
        assert caps.has_mlx is True

        # Should be same as original
        assert caps.unified_memory_gb == pytest.approx(caps.memory_gb)
        assert caps.has_mlx == caps.mlx_available

    def test_device_router_get_optimal_backends(self):
        """Test DeviceRouter.get_optimal_backends()."""
        from backend.core.optimized.device_router import DeviceRouter

        router = DeviceRouter()
        backends = router.get_optimal_backends()

        assert isinstance(backends, dict)
        assert len(backends) > 0

        # Should have task types as keys
        assert "llm_inference" in backends or "embedding" in backends

    def test_device_router_get_info(self):
        """Test DeviceRouter.get_info()."""
        from backend.core.optimized.device_router import DeviceRouter

        router = DeviceRouter()
        info = router.get_info()

        assert "memory_gb" in info
        assert "unified_memory_gb" in info
        assert "mlx_available" in info
        assert info["unified_memory_gb"] == info["memory_gb"]

    def test_device_router_singleton(self):
        """Test DeviceRouter can be instantiated multiple times (not a strict singleton)."""
        from backend.core.optimized.device_router import DeviceRouter

        router1 = DeviceRouter()
        router2 = DeviceRouter()

        # Both should work and have same capabilities
        assert router1.capabilities.memory_gb == router2.capabilities.memory_gb
        assert router1.capabilities.mlx_available == router2.capabilities.mlx_available


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for new components."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_tracing(self):
        """Test circuit breaker works with tracing."""
        from backend.core.circuit_breaker import circuit_breaker
        from backend.core.tracing import trace_span

        @circuit_breaker("traced_service")
        @trace_span("traced_operation")
        async def traced_and_protected():
            return "success"

        result = await traced_and_protected()
        assert result == "success"

    def test_all_exceptions_defined(self):
        """Test all custom exceptions are properly defined."""
        from backend.core.exceptions import (
            AuthenticationError,
            AuthorizationError,
            ContentNotFoundError,
            DatabaseError,
            DocumentNotFoundError,
            InvalidFileError,
            ProcessingError,
            RateLimitError,
            OryonException,
            TaskNotFoundError,
            ValidationError,
        )

        # Test base exception
        exc = OryonException("test", 500, "TEST_ERROR")
        result = exc.to_dict()
        assert result["error"] == "TEST_ERROR"
        assert result["status_code"] == 500

        # Test specific exceptions
        assert ContentNotFoundError("123").status_code == 404
        assert AuthenticationError().status_code == 401
        assert AuthorizationError().status_code == 403
        assert ValidationError("invalid").status_code == 422
        assert RateLimitError().status_code == 429


# ==================== Performance Tests ====================


class TestPerformance:
    """Performance tests for critical paths."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_overhead(self):
        """Test circuit breaker adds minimal overhead."""
        import time

        from backend.core.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker("perf_test")

        async def fast_func():
            await asyncio.sleep(0)  # Minimal async operation
            return 42

        # Warm up
        for _ in range(10):
            await breaker.execute(fast_func)

        # Measure
        start = time.perf_counter()
        for _ in range(1000):
            await breaker.execute(fast_func)
        elapsed = time.perf_counter() - start

        # Should complete 1000 calls in under 100ms
        assert elapsed < 0.1, f"Circuit breaker overhead too high: {elapsed:.3f}s"

    def test_tracing_noop_overhead(self):
        """Test NoOp tracing adds minimal overhead."""
        import time

        from backend.core.tracing import NoOpTracer

        tracer = NoOpTracer()

        start = time.perf_counter()
        for _ in range(10000):
            with tracer.start_as_current_span("test") as span:
                span.set_attribute("key", "value")
        elapsed = time.perf_counter() - start

        # Should complete 10000 spans in under 50ms
        assert elapsed < 0.05, f"NoOp tracing overhead too high: {elapsed:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
