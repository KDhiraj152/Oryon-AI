"""
GPU Coordination — M4 Optimized (Event-Loop Safe)
===================================================

Provides thread-safe GPU resource management for concurrent inference:
- Per-event-loop async semaphores (avoids cross-loop issues)
- Thread locks for synchronous GPU operations
- Global executor backed by ThreadPoolManager
- Sync/async GPU dispatch helpers

Configuration via environment variables:
    SSETU_MPS_SEMAPHORE  — concurrent MPS operations (default 2 for M4)
    SSETU_MLX_SEMAPHORE  — concurrent MLX/LLM operations (default 1)
"""

import asyncio
import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from backend.utils.lock_factory import LockType, create_lock

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MPS_SEMAPHORE_LIMIT = int(os.environ.get("SSETU_MPS_SEMAPHORE", "2"))
MLX_SEMAPHORE_LIMIT = int(os.environ.get("SSETU_MLX_SEMAPHORE", "1"))

# ---------------------------------------------------------------------------
# Locks (thread-level synchronization)
# ---------------------------------------------------------------------------
_mlx_lock = create_lock()       # MLX operations (LLM)
_mps_lock = create_lock()       # MPS operations (Embeddings/TTS)
_creation_lock = create_lock()  # Thread-safe semaphore/executor init

# ---------------------------------------------------------------------------
# Per-event-loop semaphore registries
# ---------------------------------------------------------------------------
_mlx_semaphores: dict[int, asyncio.Semaphore] = {}
_mps_semaphores: dict[int, asyncio.Semaphore] = {}

# ---------------------------------------------------------------------------
# Global executor (lazy, backed by ThreadPoolManager)
# ---------------------------------------------------------------------------
_gpu_executor: ThreadPoolExecutor | None = None

def _get_loop_id() -> int:
    """Get current event loop ID, or 0 if no loop running."""
    try:
        return id(asyncio.get_running_loop())
    except RuntimeError:
        return 0

# ---------------------------------------------------------------------------
# Public API — Locks
# ---------------------------------------------------------------------------

def get_gpu_lock() -> LockType:
    """Get the MPS GPU lock for synchronous operations (embeddings, etc)."""
    return _mps_lock

def get_mlx_lock() -> LockType:
    """Get the MLX lock for LLM operations."""
    return _mlx_lock

# ---------------------------------------------------------------------------
# Public API — Semaphores (per-event-loop)
# ---------------------------------------------------------------------------

def get_gpu_semaphore() -> asyncio.Semaphore:
    """
    Get the MPS GPU semaphore for async embedding operations.

    Creates semaphore in current event loop context.
    Each event loop gets its own semaphore to avoid cross-loop issues.
    """
    loop_id = _get_loop_id()
    if loop_id not in _mps_semaphores:
        with _creation_lock:
            if loop_id not in _mps_semaphores:
                _mps_semaphores[loop_id] = asyncio.Semaphore(MPS_SEMAPHORE_LIMIT)
    return _mps_semaphores[loop_id]

def get_mlx_semaphore() -> asyncio.Semaphore:
    """
    Get the MLX semaphore for LLM operations (serialize LLM calls).

    Creates semaphore in current event loop context.
    """
    loop_id = _get_loop_id()
    if loop_id not in _mlx_semaphores:
        with _creation_lock:
            if loop_id not in _mlx_semaphores:
                _mlx_semaphores[loop_id] = asyncio.Semaphore(MLX_SEMAPHORE_LIMIT)
    return _mlx_semaphores[loop_id]

# ---------------------------------------------------------------------------
# Public API — Executor
# ---------------------------------------------------------------------------

def get_gpu_executor() -> ThreadPoolExecutor:
    """
    Get the global GPU thread pool executor.

    Delegates to unified ThreadPoolManager.ml_pool (P-core affinity).
    No private pool — saves ~2 OS threads.
    """
    global _gpu_executor
    if _gpu_executor is None:
        with _creation_lock:
            if _gpu_executor is None:
                from ...core.optimized.thread_pool_manager import get_ml_executor
                _gpu_executor = get_ml_executor()
    return _gpu_executor

# ---------------------------------------------------------------------------
# Public API — Dispatch helpers
# ---------------------------------------------------------------------------

def run_on_gpu_sync(func, *args, **kwargs):
    """Run a function on MPS GPU with lock protection (synchronous)."""
    with _mps_lock:
        return func(*args, **kwargs)

async def run_on_gpu_async(func, *args, **kwargs):
    """Run a function on GPU with semaphore protection (async)."""
    async with get_gpu_semaphore():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_gpu_executor(), lambda: func(*args, **kwargs)
        )

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_gpu_resources():
    """Clean up GPU resources on shutdown."""
    global _gpu_executor
    _gpu_executor = None
    _mlx_semaphores.clear()
    _mps_semaphores.clear()

atexit.register(cleanup_gpu_resources)
