"""
Inference Backends Module
==========================

Provides optimized inference backends:
- MLX (Apple Silicon native)
- CoreML (Neural Engine)
- MPS (Metal Performance Shaders)
- CUDA (NVIDIA)
- ONNX (CPU optimized)

Also provides:
- Model warm-up service for reduced first-inference latency
- Event-loop safe async primitives (CRITICAL FIX)
"""

import asyncio
import atexit
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

# ============================================================================
# GPU COORDINATION - M4 Optimized (EVENT-LOOP SAFE)
# ============================================================================
# CRITICAL FIX: Async semaphores must be created within the correct event loop
# We use a per-loop registry to avoid cross-loop issues.

# Configurable semaphore limits (can be tuned via env vars)
MPS_SEMAPHORE_LIMIT = int(
    os.environ.get("SSETU_MPS_SEMAPHORE", "2")
)  # Default 2 for M4
MLX_SEMAPHORE_LIMIT = int(
    os.environ.get("SSETU_MLX_SEMAPHORE", "1")
)  # Default 1 (serialize LLM)

_mlx_lock = threading.Lock()  # MLX operations (LLM)
_mps_lock = threading.Lock()  # MPS operations (Embeddings/TTS)
_creation_lock = threading.Lock()  # For thread-safe initialization

# Per-event-loop semaphore registries (loop_id -> Semaphore)
_mlx_semaphores: dict[int, asyncio.Semaphore] = {}
_mps_semaphores: dict[int, asyncio.Semaphore] = {}

# Global executor (thread-safe, one per process)
_gpu_executor: ThreadPoolExecutor | None = None


def _get_loop_id() -> int:
    """Get current event loop ID, or 0 if no loop running."""
    try:
        return id(asyncio.get_running_loop())
    except RuntimeError:
        return 0


def get_gpu_lock() -> threading.Lock:
    """Get the MPS GPU lock for synchronous operations (embeddings, etc)."""
    return _mps_lock


def get_mlx_lock() -> threading.Lock:
    """Get the MLX lock for LLM operations."""
    return _mlx_lock


def get_gpu_semaphore() -> asyncio.Semaphore:
    """
    Get the MPS GPU semaphore for async embedding operations.

    CRITICAL: Creates semaphore in current event loop context.
    Each event loop gets its own semaphore to avoid cross-loop issues.
    """
    loop_id = _get_loop_id()

    if loop_id not in _mps_semaphores:
        with _creation_lock:
            if loop_id not in _mps_semaphores:
                # Allow configurable concurrent MPS operations (default 2 for M4)
                _mps_semaphores[loop_id] = asyncio.Semaphore(MPS_SEMAPHORE_LIMIT)
    return _mps_semaphores[loop_id]


def get_mlx_semaphore() -> asyncio.Semaphore:
    """
    Get the MLX semaphore for LLM operations (serialize LLM calls).

    CRITICAL: Creates semaphore in current event loop context.
    """
    loop_id = _get_loop_id()

    if loop_id not in _mlx_semaphores:
        with _creation_lock:
            if loop_id not in _mlx_semaphores:
                # Configurable MLX concurrency (default 1 to avoid Metal conflicts)
                _mlx_semaphores[loop_id] = asyncio.Semaphore(MLX_SEMAPHORE_LIMIT)
    return _mlx_semaphores[loop_id]


def get_gpu_executor() -> ThreadPoolExecutor:
    """
    Get the global GPU thread pool executor.

    Uses 2 workers - MLX and MPS can run on different threads.
    Executor is shared across event loops (thread-safe).
    """
    global _gpu_executor
    if _gpu_executor is None:
        with _creation_lock:
            if _gpu_executor is None:
                _gpu_executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="gpu_"
                )
    return _gpu_executor


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


def cleanup_gpu_resources():
    """Clean up GPU resources on shutdown."""
    global _gpu_executor
    if _gpu_executor is not None:
        _gpu_executor.shutdown(wait=False)
        _gpu_executor = None
    _mlx_semaphores.clear()
    _mps_semaphores.clear()


# Register cleanup on interpreter exit
atexit.register(cleanup_gpu_resources)


from .coreml_backend import CoreMLEmbeddingEngine, get_coreml_embeddings
from .mlx_backend import MLXInferenceEngine, get_mlx_engine
from .unified_engine import GenerationConfig, UnifiedInferenceEngine, get_inference_engine
from .warmup import (
    ModelPriority,
    ModelSpec,
    ModelWarmupService,
    get_warmup_service,
    warmup_model,
)

# Aliases for backwards compatibility
CoreMLInferenceEngine = CoreMLEmbeddingEngine
WarmupService = ModelWarmupService

__all__ = [
    "CoreMLEmbeddingEngine",
    "CoreMLInferenceEngine",  # Alias
    "GenerationConfig",
    # Inference engines
    "MLXInferenceEngine",
    "ModelPriority",
    "ModelSpec",
    # Warm-up
    "ModelWarmupService",
    "UnifiedInferenceEngine",
    "WarmupService",  # Alias
    "get_coreml_embeddings",
    "get_inference_engine",
    "get_mlx_engine",
    "get_warmup_service",
    "warmup_model",
]
