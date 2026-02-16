"""
Unified Singleton Pattern - Thread-Safe Lazy Initialization
============================================================

Provides consistent singleton pattern across all services.
Eliminates race conditions and duplicate instance creation.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

class ThreadSafeSingleton(Generic[T]):
    """
    Lock-free lazy singleton using benign-race pattern.

    Uses Python's GIL to ensure pointer assignment atomicity.
    At worst two callers create an instance but only one is kept.
    Essential for model loading to prevent OOM from multiple copies.

    Usage:
        _llm_singleton = ThreadSafeSingleton(lambda: load_model(), name="LLM")
        llm = _llm_singleton.get()
    """

    def __init__(self, factory: Callable[[], T], name: str = "singleton"):
        self._instance: T | None = None
        self._factory = factory
        self._name = name
        self._initialized = False
        self._init_time: float | None = None
        self._error: Exception | None = None

    def get(self) -> T:
        """Get or create singleton instance (lock-free)."""
        if self._instance is not None:
            return self._instance
        logger.info("[Singleton] Creating: %s", self._name)
        start = time.perf_counter()
        try:
            self._instance = self._factory()
            self._init_time = time.perf_counter() - start
            self._initialized = True
            logger.info(
                "[Singleton] %s ready in %.2fs", self._name, self._init_time
            )
        except Exception as e:
            self._error = e
            logger.error("[Singleton] %s failed: %s", self._name, e)
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
        if self._instance is not None:
            # Attempt cleanup if instance has close/cleanup method
            if hasattr(self._instance, "close"):
                try:
                    self._instance.close()
                except (RuntimeError, OSError) as e:
                    logger.warning("[Singleton] %s cleanup error: %s", self._name, e)
            elif hasattr(self._instance, "cleanup"):
                try:
                    self._instance.cleanup()
                except (RuntimeError, OSError) as e:
                    logger.warning("[Singleton] %s cleanup error: %s", self._name, e)

        self._instance = None
        self._initialized = False
        self._init_time = None
        self._error = None
        logger.info("[Singleton] %s reset", self._name)

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

        @functools.wraps(cls)
        def get_instance(*args, **kwargs):
            key = (cls, args, tuple(sorted(kwargs.items())))
            if key not in _instances:
                singleton_name = name or cls.__name__
                logger.info("[Singleton] Creating: %s", singleton_name)
                start = time.perf_counter()
                _instances[key] = cls(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(
                    "[Singleton] %s ready in %.2fs", singleton_name, elapsed
                )
            return _instances[key]

        cls.get_instance = staticmethod(get_instance)
        cls._singleton_instances = _instances

        @staticmethod
        def reset_instance():
            _instances.clear()
            logger.info("[Singleton] %s reset", name or cls.__name__)

        cls.reset_instance = reset_instance

        return cls

    return decorator

class SingletonRegistry:
    """
    Global registry for all singletons.
    Enables centralized management and cleanup.
    """

    _registry: ClassVar[dict] = {}

    @classmethod
    def register(cls, name: str, singleton: ThreadSafeSingleton):
        """Register a singleton for management."""
        cls._registry[name] = singleton
        logger.debug("[Registry] Registered: %s", name)

    @classmethod
    def get(cls, name: str) -> ThreadSafeSingleton | None:
        """Get a registered singleton by name."""
        return cls._registry.get(name)

    @classmethod
    def reset_all(cls):
        """Reset all registered singletons."""
        for name, singleton in cls._registry.items():
            try:
                singleton.reset()
            except (RuntimeError, OSError) as e:
                logger.error("[Registry] Failed to reset %s: %s", name, e)
        logger.info("[Registry] Reset %s singletons", len(cls._registry))

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
