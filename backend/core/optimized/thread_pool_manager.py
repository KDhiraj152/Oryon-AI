"""
Unified Thread Pool Manager — Consolidate All Thread Pools
============================================================

PROBLEM: The system had 6+ independent ThreadPoolExecutors:
- MLX executor (2 workers)
- GPU executor (2 workers)
- Translation executor (2 workers)
- Default asyncio executor (unbounded!)
- CoreAffinity P-core pool (8 workers)
- CoreAffinity E-core pool (12 workers)

All targeting the same Metal GPU and 10-core CPU, causing:
1. Thread contention on shared GPU resources
2. Excessive context switching overhead
3. Memory waste from idle threads (each ~8MB stack)
4. No global back-pressure on thread count

SOLUTION: Two shared pools with QoS routing:
- GPU/ML Pool: 4 workers (matches P-core count), high-priority QoS
- I/O Pool: 6 workers (matches E-core count), background QoS

All services acquire executors from this module instead of creating their own.
"""

import atexit
import logging
import os
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# Detect core topology
_IS_APPLE_SILICON = platform.machine() == "arm64" and platform.system() == "Darwin"

# M4: 4 Performance cores + 6 Efficiency cores
_P_CORES = int(os.getenv("ML_THREAD_POOL_SIZE", "4"))
_E_CORES = int(os.getenv("IO_THREAD_POOL_SIZE", "6"))


class _ThreadPoolManager:
    """Singleton manager for all thread pools in the application."""

    _instance: Optional["_ThreadPoolManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # GPU/ML inference pool — P-core affinity threads
        # Serialized GPU access means we don't need many threads;
        # but we need enough to overlap tokenization (CPU) with generation (GPU)
        self._ml_pool = ThreadPoolExecutor(
            max_workers=_P_CORES,
            thread_name_prefix="ml_pcore",
        )

        # I/O pool — E-core affinity threads for DB, Redis, filesystem, network
        self._io_pool = ThreadPoolExecutor(
            max_workers=_E_CORES,
            thread_name_prefix="io_ecore",
        )

        # Apply QoS hints on Apple Silicon
        if _IS_APPLE_SILICON:
            self._apply_qos_hints()

        self._initialized = True
        logger.info(
            f"ThreadPoolManager initialized: ML={_P_CORES} workers, "
            f"I/O={_E_CORES} workers (Apple Silicon={_IS_APPLE_SILICON})"
        )

    def _apply_qos_hints(self):
        """Apply macOS QoS class hints to thread pools via environ."""
        # QoS is applied per-thread at execution time in core_affinity.py
        # Here we just configure the pools; actual QoS is set via qos_scope()
        pass

    @property
    def ml_pool(self) -> ThreadPoolExecutor:
        """Pool for GPU/ML inference tasks (P-core affinity)."""
        return self._ml_pool

    @property
    def io_pool(self) -> ThreadPoolExecutor:
        """Pool for I/O-bound tasks: DB, Redis, filesystem, network (E-core affinity)."""
        return self._io_pool

    def shutdown(self, wait: bool = True):
        """Shutdown all thread pools."""
        logger.info("Shutting down thread pools...")
        self._ml_pool.shutdown(wait=wait)
        self._io_pool.shutdown(wait=wait)
        logger.info("Thread pools shut down")

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown(wait=False)
                cls._instance = None


def get_thread_pool_manager() -> _ThreadPoolManager:
    """Get the singleton thread pool manager."""
    return _ThreadPoolManager()


def get_ml_executor() -> ThreadPoolExecutor:
    """Get the ML/GPU thread pool executor.

    Use for: LLM inference, embedding, reranking, TTS, STT, translation.
    These are GPU-bound tasks that benefit from P-core affinity.
    """
    return get_thread_pool_manager().ml_pool


def get_io_executor() -> ThreadPoolExecutor:
    """Get the I/O thread pool executor.

    Use for: Database queries, Redis ops, filesystem reads, HTTP calls.
    These are I/O-bound tasks that benefit from E-core affinity.
    """
    return get_thread_pool_manager().io_pool


# Legacy compatibility aliases
def get_gpu_executor() -> ThreadPoolExecutor:
    """Legacy alias for get_ml_executor()."""
    return get_ml_executor()


# Register atexit cleanup
def _cleanup():
    """Cleanup thread pools on process exit."""
    try:
        manager = _ThreadPoolManager._instance
        if manager is not None:
            manager.shutdown(wait=False)
    except Exception:
        pass


atexit.register(_cleanup)
