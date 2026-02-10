"""
Unified Singleton Pattern - Thread-Safe Lazy Initialization
============================================================

Provides consistent singleton pattern across all services.
Eliminates race conditions and duplicate instance creation.
"""

import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ThreadSafeSingleton(Generic[T]):
    """
    Thread-safe lazy singleton using double-checked locking.

    Prevents race conditions where multiple threads create duplicate instances.
    Essential for model loading to prevent OOM from multiple copies.

    Usage:
        _llm_singleton = ThreadSafeSingleton(lambda: load_model(), name="LLM")
        llm = _llm_singleton.get()
    """

    def __init__(self, factory: Callable[[], T], name: str = "singleton"):
        self._instance: T | None = None
        self._lock = threading.Lock()
        self._factory = factory
        self._name = name
        self._initialized = False
        self._init_time: float | None = None
        self._error: Exception | None = None

    def get(self) -> T:
        """Get or create singleton instance with double-checked locking."""
        if self._instance is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._instance is None:
                    logger.info(f"[Singleton] Creating: {self._name}")
                    start = time.perf_counter()
                    try:
                        self._instance = self._factory()
                        self._init_time = time.perf_counter() - start
                        self._initialized = True
                        logger.info(
                            f"[Singleton] {self._name} ready in {self._init_time:.2f}s"
                        )
                    except Exception as e:
                        self._error = e
                        logger.error(f"[Singleton] {self._name} failed: {e}")
                        raise
        return self._instance

    def get_or_none(self) -> T | None:
        """Get instance if initialized, else None (no creation)."""
        return self._instance

    def is_initialized(self) -> bool:
        """Check if singleton is already initialized (without creating it)."""
        return self._initialized

    @property
    def init_time(self) -> float | None:
        """Get initialization time in seconds."""
        return self._init_time

    @property
    def last_error(self) -> Exception | None:
        """Get last initialization error if any."""
        return self._error

    def reset(self):
        """Reset singleton (useful for testing or reloading)."""
        with self._lock:
            if self._instance is not None:
                # Attempt cleanup if instance has close/cleanup method
                if hasattr(self._instance, "close"):
                    try:
                        self._instance.close()
                    except Exception as e:
                        logger.warning(f"[Singleton] {self._name} cleanup error: {e}")
                elif hasattr(self._instance, "cleanup"):
                    try:
                        self._instance.cleanup()
                    except Exception as e:
                        logger.warning(f"[Singleton] {self._name} cleanup error: {e}")

            self._instance = None
            self._initialized = False
            self._init_time = None
            self._error = None
            logger.info(f"[Singleton] {self._name} reset")

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"ThreadSafeSingleton({self._name}, {status})"


def lazy_singleton(name: str | None = None):
    """
    Decorator for creating lazy singleton classes.

    Usage:
        @lazy_singleton("MyService")
        class MyService:
            def __init__(self):
                # Heavy initialization
                pass

        # Get singleton instance
        service = MyService.get_instance()
    """

    def decorator(cls):
        _instances = {}
        _lock = threading.Lock()

        @functools.wraps(cls)
        def get_instance(*args, **kwargs):
            key = (cls, args, tuple(sorted(kwargs.items())))
            if key not in _instances:
                with _lock:
                    if key not in _instances:
                        singleton_name = name or cls.__name__
                        logger.info(f"[Singleton] Creating: {singleton_name}")
                        start = time.perf_counter()
                        _instances[key] = cls(*args, **kwargs)
                        elapsed = time.perf_counter() - start
                        logger.info(
                            f"[Singleton] {singleton_name} ready in {elapsed:.2f}s"
                        )
            return _instances[key]

        cls.get_instance = staticmethod(get_instance)
        cls._singleton_instances = _instances
        cls._singleton_lock = _lock

        @staticmethod
        def reset_instance():
            with _lock:
                _instances.clear()
                logger.info(f"[Singleton] {name or cls.__name__} reset")

        cls.reset_instance = reset_instance

        return cls

    return decorator


class SingletonRegistry:
    """
    Global registry for all singletons.
    Enables centralized management and cleanup.
    """

    _registry: dict = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, name: str, singleton: ThreadSafeSingleton):
        """Register a singleton for management."""
        with cls._lock:
            cls._registry[name] = singleton
            logger.debug(f"[Registry] Registered: {name}")

    @classmethod
    def get(cls, name: str) -> ThreadSafeSingleton | None:
        """Get a registered singleton by name."""
        return cls._registry.get(name)

    @classmethod
    def reset_all(cls):
        """Reset all registered singletons."""
        with cls._lock:
            for name, singleton in cls._registry.items():
                try:
                    singleton.reset()
                except Exception as e:
                    logger.error(f"[Registry] Failed to reset {name}: {e}")
            logger.info(f"[Registry] Reset {len(cls._registry)} singletons")

    @classmethod
    def status(cls) -> dict:
        """Get status of all registered singletons."""
        return {
            name: {
                "initialized": singleton.is_initialized(),
                "init_time": singleton.init_time,
            }
            for name, singleton in cls._registry.items()
        }
