"""
Circuit Breaker Pattern Implementation
======================================

Provides resilience for external service calls (Redis, DB, ML models).
Prevents cascade failures by stopping requests to failing services.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests are rejected immediately
- HALF_OPEN: Testing if service has recovered

Usage:
    @circuit_breaker("redis")
    async def get_from_redis(key: str):
        return await redis.get(key)
"""

import asyncio
import functools
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, ClassVar, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes in half-open before closing
    timeout_seconds: float = 30.0  # Time before attempting recovery
    half_open_max_calls: int = 3  # Max calls allowed in half-open state
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures

@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    total_failures: int = 0
    total_successes: int = 0
    total_rejected: int = 0
    opened_at: float | None = None
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=10))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for monitoring."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_rejected": self.total_rejected,
            "last_failure": datetime.fromtimestamp(
                self.last_failure_time, tz=UTC
            ).isoformat()
            if self.last_failure_time
            else None,
            "recent_failures": list(self.recent_failures),
        }

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, service_name: str, retry_after: float):
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker open for {service_name}. Retry after {retry_after:.1f}s"
        )

class CircuitBreaker:
    """
    Circuit breaker implementation for service resilience.

    Example:
        redis_breaker = CircuitBreaker("redis", config=CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=60
        ))

        @redis_breaker
        async def get_cached_value(key: str):
            return await redis.get(key)
    """

    # Global registry of all circuit breakers
    _registry: ClassVar[dict[str, "CircuitBreaker"]] = {}

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitStats()
        self._lock = asyncio.Lock()
        self._half_open_calls = 0

        # Register this breaker
        CircuitBreaker._registry[name] = self

    @classmethod
    def get(cls, name: str) -> "CircuitBreaker | None":
        """Get a circuit breaker by name."""
        return cls._registry.get(name)

    @classmethod
    def get_all_stats(cls) -> dict[str, dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {name: cb.stats.to_dict() for name, cb in cls._registry.items()}

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on state."""
        if self.stats.state == CircuitState.CLOSED:
            return True

        if self.stats.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if (
                self.stats.opened_at
                and time.time() - self.stats.opened_at >= self.config.timeout_seconds
            ):
                self.stats.state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit breaker '%s' transitioning to HALF_OPEN", self.name)
                return True
            return False

        if self.stats.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

        return False

    def _record_success(self) -> None:
        """Record a successful call."""
        self.stats.success_count += 1
        self.stats.total_successes += 1
        self.stats.last_success_time = time.time()

        if self.stats.state == CircuitState.HALF_OPEN and self.stats.success_count >= self.config.success_threshold:
                self.stats.state = CircuitState.CLOSED
                self.stats.failure_count = 0
                self.stats.success_count = 0
                logger.info("Circuit breaker '%s' CLOSED (recovered)", self.name)

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        self.stats.failure_count += 1
        self.stats.total_failures += 1
        self.stats.last_failure_time = time.time()
        self.stats.recent_failures.append(
            {"time": datetime.now(UTC).isoformat(), "error": str(error)[:100]}
        )

        if self.stats.state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately opens the circuit
            self.stats.state = CircuitState.OPEN
            self.stats.opened_at = time.time()
            self.stats.success_count = 0
            logger.warning("Circuit breaker '%s' OPEN (failed in half-open)", self.name)

        elif self.stats.state == CircuitState.CLOSED:
            if self.stats.failure_count >= self.config.failure_threshold:
                self.stats.state = CircuitState.OPEN
                self.stats.opened_at = time.time()
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN (threshold reached)"
                )

    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a function with circuit breaker protection (async)."""
        async with self._lock:
            if not self._should_allow_request():
                self.stats.total_rejected += 1
                retry_after = self.config.timeout_seconds - (
                    time.time() - (self.stats.opened_at or 0)
                )
                raise CircuitBreakerError(self.name, max(0, retry_after))

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            async with self._lock:
                self._record_success()

            return result

        except Exception as e:
            # Check if this exception should count as a failure
            if isinstance(e, self.config.excluded_exceptions):
                raise

            async with self._lock:
                self._record_failure(e)
            raise

    def call_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a function with circuit breaker protection (sync version).

        Thread-safe synchronous circuit breaker execution for embedding/sync code paths.
        """
        # Simple sync check without async lock
        if not self._should_allow_request():
            self.stats.total_rejected += 1
            retry_after = self.config.timeout_seconds - (
                time.time() - (self.stats.opened_at or 0)
            )
            raise CircuitBreakerError(self.name, max(0, retry_after))

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except Exception as e:
            if isinstance(e, self.config.excluded_exceptions):
                raise
            self._record_failure(e)
            raise

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator for protecting a function with circuit breaker."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.execute(func, *args, **kwargs)

        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self.stats = CircuitStats()
        self._half_open_calls = 0
        logger.info("Circuit breaker '%s' manually reset to CLOSED", self.name)

# Pre-configured circuit breakers for common services
def get_database_breaker() -> CircuitBreaker:
    """Get or create the database circuit breaker."""
    if "database" not in CircuitBreaker._registry:
        return CircuitBreaker(
            "database",
            CircuitBreakerConfig(
                failure_threshold=5, timeout_seconds=30, success_threshold=2
            ),
        )
    return CircuitBreaker._registry["database"]

def get_redis_breaker() -> CircuitBreaker:
    """Get or create the Redis circuit breaker."""
    if "redis" not in CircuitBreaker._registry:
        return CircuitBreaker(
            "redis",
            CircuitBreakerConfig(
                failure_threshold=3, timeout_seconds=15, success_threshold=2
            ),
        )
    return CircuitBreaker._registry["redis"]

def get_ml_breaker() -> CircuitBreaker:
    """Get or create the ML model circuit breaker."""
    if "ml_model" not in CircuitBreaker._registry:
        return CircuitBreaker(
            "ml_model",
            CircuitBreakerConfig(
                failure_threshold=3, timeout_seconds=60, success_threshold=2
            ),
        )
    return CircuitBreaker._registry["ml_model"]

def get_external_api_breaker() -> CircuitBreaker:
    """Get or create the external API circuit breaker."""
    if "external_api" not in CircuitBreaker._registry:
        return CircuitBreaker(
            "external_api",
            CircuitBreakerConfig(
                failure_threshold=5, timeout_seconds=120, success_threshold=3
            ),
        )
    return CircuitBreaker._registry["external_api"]

# Convenience decorator
def circuit_breaker(service_name: str, config: CircuitBreakerConfig | None = None):
    """
    Decorator to protect a function with a circuit breaker.

    Usage:
        @circuit_breaker("redis")
        async def get_from_cache(key: str):
            return await redis.get(key)
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        breaker = CircuitBreaker.get(service_name)
        if not breaker:
            breaker = CircuitBreaker(service_name, config)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await breaker.execute(func, *args, **kwargs)

        return wrapper

    return decorator
