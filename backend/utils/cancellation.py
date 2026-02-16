"""
Request Cancellation Utility
============================

Provides request cancellation detection for long-running operations.
When a client disconnects, we should stop processing to free resources.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from functools import wraps
from typing import TypeVar

from fastapi import Request

logger = logging.getLogger(__name__)

T = TypeVar("T")

class RequestCancelled(Exception):
    """Raised when the client has disconnected."""

    pass

async def check_cancelled(request: Request) -> None:
    """
    Check if request has been cancelled by the client.

    Call this periodically in long-running operations.
    Raises RequestCancelled if client has disconnected.
    """
    if await request.is_disconnected():
        logger.info("Client disconnected: %s %s", request.method, request.url.path)
        raise RequestCancelled("Client disconnected")

async def with_cancellation(
    request: Request, generator: AsyncGenerator[T, None], cleanup_fn=None
) -> AsyncGenerator[T, None]:
    """
    Wrap an async generator with client disconnect detection.

    Usage:
        async def my_stream():
            for chunk in data:
                yield chunk

        async for chunk in with_cancellation(request, my_stream()):
            yield chunk

    Args:
        request: FastAPI Request object
        generator: The async generator to wrap
        cleanup_fn: Optional cleanup function to call on cancellation

    Yields:
        Items from the wrapped generator
    """
    try:
        async for item in generator:
            # Check if client is still connected
            if await request.is_disconnected():
                logger.info("Client disconnected during streaming: %s", request.url.path)
                break
            yield item
    except asyncio.CancelledError:
        logger.info("Request cancelled: %s", request.url.path)
        raise
    finally:
        if cleanup_fn:
            try:
                if asyncio.iscoroutinefunction(cleanup_fn):
                    await cleanup_fn()
                else:
                    cleanup_fn()
            except (RuntimeError, OSError) as e:
                logger.warning("Cleanup error on cancellation: %s", e)

def cancellable(check_interval: int = 10):
    """
    Decorator for making long-running endpoint handlers cancellable.

    Automatically checks for client disconnect every N iterations.

    Usage:
        @router.post("/long-running")
        @cancellable(check_interval=5)
        async def long_running_handler(request: Request):
            # Your long-running code
            pass

    Args:
        check_interval: Check for disconnect every N operations
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            try:
                return await func(request, *args, **kwargs)
            except RequestCancelled:
                logger.info("Request cancelled by client: %s", request.url.path)
                return None
            except asyncio.CancelledError:
                logger.info("Request task cancelled: %s", request.url.path)
                raise

        return wrapper

    return decorator

class CancellationToken:
    """
    Token for cooperative cancellation in long-running operations.

    Usage:
        token = CancellationToken(request)

        while processing:
            if await token.is_cancelled():
                break
            # do work
    """

    def __init__(self, request: Request, check_interval: float = 0.5):
        self.request = request
        self.check_interval = check_interval
        self._cancelled = False
        self._last_check: float = 0.0

    async def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        if self._cancelled:
            return True

        try:
            import time

            now = time.time()

            # Only check every check_interval seconds
            if now - self._last_check >= self.check_interval:
                self._last_check = now
                if await self.request.is_disconnected():
                    self._cancelled = True
                    logger.debug("Detected client disconnect: %s", self.request.url.path)
        except (RuntimeError, OSError):
            pass  # Ignore errors in cancellation check

        return self._cancelled

    def cancel(self) -> None:
        """Manually mark as cancelled."""
        self._cancelled = True

__all__ = [
    "CancellationToken",
    "RequestCancelled",
    "cancellable",
    "check_cancelled",
    "with_cancellation",
]
