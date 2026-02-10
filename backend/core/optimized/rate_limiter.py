"""
Unified Rate Limiter - Consolidates all rate limiting implementations.

This module provides a single, optimized rate limiter that:
- Uses async Redis with fallback to memory
- Supports per-user role-based limits
- Provides endpoint-specific limits
- Uses token bucket algorithm with burst support
- Includes caching for hot paths

Replaces:
- backend/core/rate_limiter.py
- backend/core/rate_limiter_async.py
- backend/core/user_rate_limits.py
- RateLimiter class in model_clients.py
- RateLimitValidator in sanitizer.py
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ==================== Configuration ====================


class UserRole(str, Enum):
    """User roles with different rate limits."""

    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"
    GUEST = "guest"
    API = "api"


@dataclass(frozen=True)
class RateLimitConfig:
    """Immutable rate limit configuration."""

    per_minute: int = 60
    per_hour: int = 600
    burst_multiplier: float = 1.5

    def with_multiplier(self, mult: float) -> "RateLimitConfig":
        """Return new config with multiplied limits."""
        return RateLimitConfig(
            per_minute=int(self.per_minute * mult),
            per_hour=int(self.per_hour * mult),
            burst_multiplier=self.burst_multiplier,
        )


# Role-based defaults
ROLE_LIMITS: dict[UserRole, RateLimitConfig] = {
    UserRole.ADMIN: RateLimitConfig(per_minute=1000, per_hour=10000),
    UserRole.TEACHER: RateLimitConfig(per_minute=200, per_hour=2000),
    UserRole.STUDENT: RateLimitConfig(per_minute=60, per_hour=600),
    UserRole.GUEST: RateLimitConfig(per_minute=20, per_hour=100),
    UserRole.API: RateLimitConfig(per_minute=500, per_hour=5000),
}

# Endpoint-specific limits (path prefix -> config)
ENDPOINT_LIMITS: dict[str, RateLimitConfig] = {
    "/api/v2/auth/login": RateLimitConfig(per_minute=10, per_hour=100),
    "/api/v2/auth/register": RateLimitConfig(per_minute=5, per_hour=50),
    "/api/v2/content/upload": RateLimitConfig(per_minute=5, per_hour=50),
    "/api/v2/content/process": RateLimitConfig(per_minute=10, per_hour=100),
    "/api/v2/process": RateLimitConfig(per_minute=30, per_hour=500),
    "/api/v2/embed": RateLimitConfig(per_minute=100, per_hour=2000),
}

# Exempt paths (no rate limiting)
EXEMPT_PATHS = frozenset(
    [
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/metrics",
        "/api/v2/health",
    ]
)


# ==================== In-Memory Store ====================


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    tokens: float
    last_update: float
    request_count: int = 0


class MemoryStore:
    """Thread-safe in-memory rate limit store."""

    def __init__(self, max_entries: int = 10000):
        self._store: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self._max_entries = max_entries
        self._cleanup_threshold = int(max_entries * 0.9)

    async def check_and_consume(
        self, key: str, limit: int, window: int, burst: float = 1.5
    ) -> tuple[bool, int]:
        """
        Check rate limit and consume a token.

        Returns:
            Tuple of (allowed: bool, remaining: int)
        """
        max_tokens = int(limit * burst)
        refill_rate = limit / window  # tokens per second

        async with self._lock:
            now = time.time()

            if key not in self._store:
                self._store[key] = TokenBucket(
                    tokens=max_tokens - 1, last_update=now, request_count=1
                )
                self._maybe_cleanup(now)
                return True, max_tokens - 1

            bucket = self._store[key]

            # Refill tokens based on elapsed time
            elapsed = now - bucket.last_update
            refill = elapsed * refill_rate
            bucket.tokens = min(max_tokens, bucket.tokens + refill)
            bucket.last_update = now

            # Try to consume a token
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                bucket.request_count += 1
                return True, int(bucket.tokens)
            else:
                return False, 0

    async def get_count(self, key: str) -> int:
        """Get request count for key."""
        async with self._lock:
            if key in self._store:
                return self._store[key].request_count
            return 0

    def _maybe_cleanup(self, current_time: float) -> None:
        """Clean up expired entries if store is getting full.

        Note: This is intentionally synchronous as cleanup happens
        within an already-acquired async lock context.
        """
        if len(self._store) < self._cleanup_threshold:
            return

        # Remove entries older than 1 hour
        cutoff = current_time - 3600
        expired = [k for k, v in self._store.items() if v.last_update < cutoff]
        for key in expired[:1000]:  # Limit cleanup batch
            del self._store[key]


# ==================== Redis Store ====================


class RedisStore:
    """Redis-backed rate limit store."""

    def __init__(self, redis_client):
        self._redis = redis_client
        # Check if client is async or sync
        self._is_async = hasattr(redis_client, "__aenter__") if redis_client else False

    async def check_and_consume(
        self, key: str, limit: int, window: int, burst: float = 1.5
    ) -> tuple[bool, int]:
        """Check rate limit using Redis with Lua script for atomicity."""
        max_tokens = int(limit * burst)

        # Lua script for atomic token bucket
        lua_script = """
        local key = KEYS[1]
        local max_tokens = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local window = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        local data = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber(data[1]) or max_tokens
        local last_update = tonumber(data[2]) or now

        -- Refill tokens
        local elapsed = now - last_update
        tokens = math.min(max_tokens, tokens + elapsed * refill_rate)

        -- Try to consume
        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
            redis.call('EXPIRE', key, window * 2)
            return {1, math.floor(tokens)}
        else
            return {0, 0}
        end
        """

        try:
            # Use async Redis client directly - no executor needed
            # The redis.asyncio client handles this natively
            if hasattr(self._redis, "eval"):
                result = await asyncio.wait_for(
                    self._redis.eval(
                        lua_script,
                        1,
                        key,
                        max_tokens,
                        limit / window,
                        window,
                        time.time(),
                    ),
                    timeout=2.0,  # Prevent hanging on Redis issues
                )
                return bool(result[0]), int(result[1])
            else:
                # Fallback for sync client - run in executor
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._redis.eval(
                        lua_script,
                        1,
                        key,
                        max_tokens,
                        limit / window,
                        window,
                        time.time(),
                    ),
                )
                return bool(result[0]), int(result[1])
        except TimeoutError:
            logger.warning(f"Redis rate limit timeout for key: {key}")
            return True, limit  # Fail open on timeout
        except Exception as e:
            logger.warning(f"Redis rate limit error: {e}")
            # Fail open
            return True, limit


# ==================== Unified Rate Limiter ====================


class UnifiedRateLimiter:
    """
    Unified rate limiter with async support.

    Features:
    - Token bucket algorithm with burst support
    - Role-based rate limits
    - Endpoint-specific limits
    - Redis with memory fallback
    - Configurable exempt paths
    """

    def __init__(
        self,
        redis_client=None,
        enabled: bool = True,
        default_config: RateLimitConfig = None,
    ):
        self._redis = RedisStore(redis_client) if redis_client else None
        self._memory = MemoryStore()
        self._enabled = enabled
        self._default_config = default_config or RateLimitConfig()

        logger.info(
            f"UnifiedRateLimiter initialized: "
            f"enabled={enabled}, redis={'yes' if redis_client else 'no'}"
        )
        # Initialize config cache (instance dict instead of lru_cache to avoid memory leak)
        self._config_cache: dict[tuple[str, UserRole], RateLimitConfig] = {}

    def _get_config(self, path: str, role: UserRole) -> RateLimitConfig:
        """Get rate limit config for path and role (cached in instance)."""
        cache_key = (path, role)
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        # Check endpoint-specific limits first
        for prefix, config in ENDPOINT_LIMITS.items():
            if path.startswith(prefix):
                self._config_cache[cache_key] = config
                return config

        # Fall back to role-based limits
        result = ROLE_LIMITS.get(role, self._default_config)
        self._config_cache[cache_key] = result
        return result

    def _get_identifier(self, request: Request) -> tuple[str, UserRole]:
        """Extract identifier and role from request."""
        # Try authenticated user
        if hasattr(request.state, "user") and request.state.user:
            user = request.state.user
            user_id = getattr(user, "id", None)
            role_str = getattr(user, "role", "student")
            try:
                role = UserRole(role_str)
            except ValueError:
                role = UserRole.STUDENT
            if user_id:
                return f"user:{user_id}", role

        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}", UserRole.GUEST

    async def check_request(self, request: Request) -> tuple[bool, dict[str, Any]]:
        """
        Check if request should be allowed.

        Returns:
            Tuple of (allowed, headers_dict)
        """
        if not self._enabled:
            return True, {}

        path = request.url.path

        # Check exempt paths
        if path in EXEMPT_PATHS or any(path.startswith(p) for p in EXEMPT_PATHS):
            return True, {}

        identifier, role = self._get_identifier(request)
        config = self._get_config(path, role)

        now = time.time()
        minute_key = f"rl:min:{identifier}:{int(now / 60)}"
        hour_key = f"rl:hr:{identifier}:{int(now / 3600)}"

        # Use Redis if available, else memory
        store = self._redis if self._redis else self._memory

        # Check minute limit
        minute_ok, minute_remaining = await store.check_and_consume(
            minute_key, config.per_minute, 60, config.burst_multiplier
        )

        if not minute_ok:
            return False, self._limit_headers(config, 0, 0, "minute")

        # Check hour limit
        hour_ok, hour_remaining = await store.check_and_consume(
            hour_key, config.per_hour, 3600, config.burst_multiplier
        )

        if not hour_ok:
            return False, self._limit_headers(config, minute_remaining, 0, "hour")

        return True, self._limit_headers(config, minute_remaining, hour_remaining)

    def _limit_headers(
        self,
        config: RateLimitConfig,
        minute_remaining: int,
        hour_remaining: int,
        exceeded: str | None = None,
    ) -> dict[str, str]:
        """Generate rate limit headers."""
        headers = {
            "X-RateLimit-Limit-Minute": str(config.per_minute),
            "X-RateLimit-Limit-Hour": str(config.per_hour),
            "X-RateLimit-Remaining-Minute": str(minute_remaining),
            "X-RateLimit-Remaining-Hour": str(hour_remaining),
        }
        if exceeded:
            headers["Retry-After"] = "60" if exceeded == "minute" else "3600"
        return headers


# ==================== Middleware ====================


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, rate_limiter: UnifiedRateLimiter = None, **kwargs):
        super().__init__(app)
        self._limiter = rate_limiter or UnifiedRateLimiter(**kwargs)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to request."""
        try:
            allowed, headers = await self._limiter.check_request(request)

            if not allowed:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Rate limit exceeded",
                        "error": "rate_limit_exceeded",
                    },
                    headers=headers,
                )

            response = await call_next(request)

            # Add rate limit headers
            for key, value in headers.items():
                response.headers[key] = value

            return response

        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # Fail open
            return await call_next(request)


# ==================== Simple Rate Limiter for Services ====================


class SimpleRateLimiter:
    """
    Simple async rate limiter for internal service use.

    Replaces the RateLimiter classes in model_clients.py
    """

    def __init__(self, max_calls: int = 100, window_seconds: int = 60):
        self._max_calls = max_calls
        self._window = window_seconds
        self._calls: list = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Acquire a rate limit slot, waiting if necessary.

        Returns:
            True when slot acquired
        """
        async with self._lock:
            now = time.time()

            # Remove expired calls
            self._calls = [t for t in self._calls if now - t < self._window]

            if len(self._calls) >= self._max_calls:
                # Wait for oldest call to expire
                sleep_time = self._window - (now - self._calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    self._calls = []

            self._calls.append(time.time())
            return True

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        pass


# ==================== Exports ====================

__all__ = [
    "ENDPOINT_LIMITS",
    "EXEMPT_PATHS",
    "ROLE_LIMITS",
    "RateLimitConfig",
    "RateLimitMiddleware",
    "SimpleRateLimiter",
    "UnifiedRateLimiter",
    "UserRole",
]
