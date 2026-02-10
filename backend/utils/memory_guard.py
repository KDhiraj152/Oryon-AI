"""
Memory Guard - Per-Request Memory Pressure Protection
=====================================================

Provides decorators and utilities to check memory pressure before
executing memory-intensive operations like model inference.

CRITICAL FIX: Prevents OOM by refusing requests when memory is critical.

Usage:
    from backend.utils.memory_guard import require_memory, check_memory_pressure

    @require_memory(action="reject")  # Reject requests during critical pressure
    async def generate_text(prompt: str) -> str:
        ...

    # Or check manually:
    if not check_memory_pressure():
        raise HTTPException(503, "Server under memory pressure, try again later")
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Literal

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Try to import memory coordinator
_COORDINATOR_AVAILABLE = False
_coordinator = None

try:
    from backend.core.optimized.memory_coordinator import (
        MemoryPressure,
        get_memory_coordinator,
    )

    _COORDINATOR_AVAILABLE = True
except ImportError:
    # Fallback: no memory coordination
    class MemoryPressure:
        NORMAL = "normal"
        WARNING = "warning"
        CRITICAL = "critical"
        EMERGENCY = "emergency"


def _get_coordinator():
    """Lazy get memory coordinator singleton."""
    global _coordinator
    if _COORDINATOR_AVAILABLE and _coordinator is None:
        try:
            _coordinator = get_memory_coordinator()
        except Exception as e:
            logger.warning(f"Failed to get memory coordinator: {e}")
    return _coordinator


def get_memory_pressure() -> str:
    """
    Get current memory pressure level.

    Returns:
        One of: 'normal', 'warning', 'critical', 'emergency'
    """
    coordinator = _get_coordinator()
    if coordinator is None:
        return "normal"  # Assume normal if no coordinator

    try:
        pressure = coordinator.get_memory_pressure()
        return pressure.value
    except Exception:
        return "normal"


def check_memory_pressure(
    reject_on: tuple = ("critical", "emergency"),
    warn_on: tuple = ("warning",),
) -> bool:
    """
    Check if memory pressure allows new requests.

    Args:
        reject_on: Pressure levels that should reject the request
        warn_on: Pressure levels that should log a warning

    Returns:
        True if request can proceed, False if should be rejected
    """
    pressure = get_memory_pressure()

    if pressure in warn_on:
        logger.warning(f"Memory pressure is {pressure}, request may be slow")
        return True

    if pressure in reject_on:
        logger.error(f"Memory pressure is {pressure}, rejecting request")
        return False

    return True


def require_memory(
    action: Literal["reject", "warn", "queue"] = "reject",
    reject_on: tuple = ("critical", "emergency"),
    error_code: int = 503,
    error_message: str = "Server under memory pressure, please try again later",
):
    """
    Decorator to check memory pressure before executing an endpoint.

    Args:
        action: What to do under pressure - 'reject', 'warn', or 'queue'
        reject_on: Pressure levels that trigger the action
        error_code: HTTP status code for rejected requests
        error_message: Message to return on rejection

    Usage:
        @router.post("/generate")
        @require_memory(action="reject")
        async def generate_text(request: Request):
            ...
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            pressure = get_memory_pressure()

            if pressure in reject_on:
                if action == "reject":
                    logger.warning(
                        f"Rejecting request due to memory pressure: {pressure}"
                    )
                    raise HTTPException(status_code=error_code, detail=error_message)
                elif action == "warn":
                    logger.warning(
                        f"Memory pressure {pressure}, proceeding with caution"
                    )
                elif action == "queue":
                    # Wait for memory pressure to reduce
                    max_wait = 30  # seconds
                    waited = 0
                    while pressure in reject_on and waited < max_wait:
                        await asyncio.sleep(1)
                        waited += 1
                        pressure = get_memory_pressure()

                    if pressure in reject_on:
                        raise HTTPException(
                            status_code=error_code,
                            detail=f"Memory pressure persisted after {max_wait}s",
                        )

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            pressure = get_memory_pressure()

            if pressure in reject_on:
                if action == "reject":
                    raise HTTPException(status_code=error_code, detail=error_message)
                elif action == "warn":
                    logger.warning(
                        f"Memory pressure {pressure}, proceeding with caution"
                    )

            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


async def wait_for_memory(
    timeout: float = 30.0,
    check_interval: float = 1.0,
    target_pressure: tuple = ("normal", "warning"),
) -> bool:
    """
    Wait for memory pressure to reduce to acceptable level.

    Args:
        timeout: Maximum seconds to wait
        check_interval: Seconds between checks
        target_pressure: Acceptable pressure levels

    Returns:
        True if pressure reduced, False if timeout
    """
    waited = 0.0
    while waited < timeout:
        pressure = get_memory_pressure()
        if pressure in target_pressure:
            return True
        await asyncio.sleep(check_interval)
        waited += check_interval

    return False


def get_memory_stats() -> dict:
    """
    Get detailed memory statistics.

    Returns:
        Dict with memory usage info
    """
    coordinator = _get_coordinator()
    if coordinator is None:
        return {"status": "coordinator_unavailable"}

    try:
        used_gb, total_gb = coordinator.get_system_memory_gb()
        available_gb = coordinator.get_available_memory_gb()
        pressure = coordinator.get_memory_pressure()

        return {
            "pressure": pressure.value,
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "available_for_models_gb": round(available_gb, 2),
            "usage_percent": round((used_gb / total_gb) * 100, 1),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
