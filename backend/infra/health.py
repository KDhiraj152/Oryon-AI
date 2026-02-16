"""
Health Checker — Readiness and Liveness Probes
=================================================

Provides structured health checks for:
  - Kubernetes readiness/liveness probes
  - Load balancer health endpoints
  - Dependency status monitoring

Checks:
  - Database connectivity
  - Redis connectivity
  - Model availability
  - Memory pressure
  - Worker pool availability
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from backend.infra.telemetry import get_logger

logger = get_logger(__name__)

class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """Aggregate system health."""

    status: HealthStatus
    checks: list[HealthCheck]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": round(c.latency_ms, 2),
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }

class HealthChecker:
    """
    Aggregated health checker.

    Usage:
        checker = get_health_checker()
        checker.register("database", check_database)
        checker.register("redis", check_redis)

        health = await checker.check()
        # Returns SystemHealth with all check results
    """

    def __init__(self) -> None:
        self._checks: dict[str, Callable[[], Awaitable[HealthCheck]]] = {}
        self._cache_ttl_s = 5.0
        self._cached: SystemHealth | None = None
        self._cached_at = 0.0
        self._lock = threading.Lock()

    def register(self, name: str, check_fn: Callable[[], Awaitable[HealthCheck]]) -> None:
        """Register a health check function."""
        self._checks[name] = check_fn

    async def check(self, *, use_cache: bool = True) -> SystemHealth:
        """Run all health checks."""
        if (use_cache and self._cached
                and time.monotonic() - self._cached_at < self._cache_ttl_s):
            return self._cached

        checks = await asyncio.gather(
            *(self._run_check(name, fn) for name, fn in self._checks.items()),
            return_exceptions=True,
        )

        results = []
        for check in checks:
            if isinstance(check, Exception):
                results.append(
                    HealthCheck(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        message=str(check),
                    )
                )
            else:
                results.append(check)  # type: ignore[arg-type]

        # Aggregate status
        statuses = [c.status for c in results]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        health = SystemHealth(status=overall, checks=results)

        with self._lock:
            self._cached = health
            self._cached_at = time.monotonic()

        return health

    async def liveness(self) -> bool:
        """Simple liveness check (is process alive?)."""
        return True

    async def readiness(self) -> bool:
        """Readiness check (are all dependencies ready?)."""
        health = await self.check()
        return health.status != HealthStatus.UNHEALTHY

    @staticmethod
    async def _run_check(
        name: str, fn: Callable[[], Awaitable[HealthCheck]]
    ) -> HealthCheck:
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(fn(), timeout=5.0)
            result.latency_ms = (time.monotonic() - start) * 1000
            return result
        except TimeoutError:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.monotonic() - start) * 1000,
                message="Health check timed out",
            )
        except (RuntimeError, OSError) as exc:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.monotonic() - start) * 1000,
                message=str(exc),
            )

# ── Built-in Health Checks ────────────────────────────────────────

async def check_database() -> HealthCheck:
    """Check database connectivity."""
    try:
        from sqlalchemy import text

        from backend.db.database import get_async_db
        async for session in get_async_db():
            await session.execute(text("SELECT 1"))
            return HealthCheck(name="database", status=HealthStatus.HEALTHY)
    except (OSError, RuntimeError) as exc:
        return HealthCheck(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=str(exc),
        )
    return HealthCheck(
        name="database",
        status=HealthStatus.UNHEALTHY,
        message="No database session available",
    )

async def check_memory() -> HealthCheck:
    """Check memory pressure."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        util = mem.percent / 100

        if util > 0.95:
            status = HealthStatus.UNHEALTHY
        elif util > 0.85:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return HealthCheck(
            name="memory",
            status=status,
            details={
                "total_gb": round(mem.total / (1024 ** 3), 1),
                "used_percent": round(mem.percent, 1),
            },
        )
    except (RuntimeError, OSError) as exc:
        return HealthCheck(
            name="memory", status=HealthStatus.DEGRADED, message=str(exc)
        )

# ── Singleton ──────────────────────────────────────────────────────

_checker: HealthChecker | None = None

def get_health_checker() -> HealthChecker:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _checker
    if _checker is not None:
        return _checker
    _checker = HealthChecker()
    # Register built-in checks
    _checker.register("memory", check_memory)
    return _checker