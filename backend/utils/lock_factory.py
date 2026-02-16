"""
Centralized lock factory for dependency injection.

Provides injectable lock creation so that module-level singleton locks
can be replaced with test doubles (e.g., no-op locks) during testing.

Usage:
    from backend.utils.lock_factory import create_lock

    # Module-level singleton lock (injectable for testing)
    _my_lock = create_lock()

    # Override in tests:
    from backend.utils import lock_factory
    lock_factory.set_factory(lambda: threading.RLock())  # or a no-op lock
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Union

# threading.Lock and threading.RLock are factory functions, not types.
# Use the instance types for the type alias.
LockType = Union[threading._RLock, "threading.Lock"]

_factory: Callable[[], LockType] = threading.Lock

def create_lock() -> LockType:
    """Create a lock using the current factory.

    Returns a threading.Lock by default. Can be overridden via
    ``set_factory`` for testing or custom lock implementations.
    """
    return _factory()

def set_factory(factory: Callable[[], LockType]) -> None:
    """Override the global lock factory.

    Useful in tests to inject no-op locks or RLocks.
    """
    global _factory
    _factory = factory

def reset_factory() -> None:
    """Restore the default lock factory (threading.Lock)."""
    global _factory
    _factory = threading.Lock

class NoOpLock:
    """A no-op lock for use in single-threaded test environments."""

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return True

    def release(self) -> None:
        pass

    def __enter__(self) -> NoOpLock:
        return self

    def __exit__(self, *args: object) -> None:
        pass
