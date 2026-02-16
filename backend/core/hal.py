"""
Hardware Abstraction Layer (HAL) — Unified Device Detection
=============================================================

Provides simple, cached functions for hardware detection that replace
the 30+ scattered inline checks of `torch.cuda.is_available()`,
`torch.backends.mps.is_available()`, `platform.machine()` etc.

ALL device-detection queries should go through this module:
    from backend.core.hal import get_device, has_gpu, device_caps

This module is SAFE to import from any layer — it defers heavy imports
(torch, mlx) to first actual use and caches results.
"""

from __future__ import annotations

import functools
import logging
import platform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.optimized.device_router import DeviceCapabilities

logger = logging.getLogger(__name__)

# ── Cached primitives (no torch import needed) ──────────────────────────

@functools.lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    """True if running on Apple Silicon (M1/M2/M3/M4)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

@functools.lru_cache(maxsize=1)
def is_darwin() -> bool:
    return platform.system() == "Darwin"

# ── Lazy torch-dependent checks (import on first call) ──────────────────

@functools.lru_cache(maxsize=1)
def has_mps() -> bool:
    """True if Metal Performance Shaders (MPS) is available."""
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False

@functools.lru_cache(maxsize=1)
def has_cuda() -> bool:
    """True if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

@functools.lru_cache(maxsize=1)
def has_mlx() -> bool:
    """True if Apple MLX framework is available."""
    try:
        import mlx.core
        return True
    except ImportError:
        return False

@functools.lru_cache(maxsize=1)
def has_gpu() -> bool:
    """True if any GPU accelerator is available (MPS, CUDA, or MLX)."""
    return has_mps() or has_cuda() or has_mlx()

# ── Device selection ─────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def get_device() -> str:
    """
    Return the best available torch device string.

    Priority: cuda > mps > cpu
    (MLX has its own runtime and doesn't use torch devices.)
    """
    if has_cuda():
        return "cuda"
    if has_mps():
        return "mps"
    return "cpu"

def get_torch_dtype():
    """Return optimal torch dtype for current device."""
    import torch
    if get_device() in ("cuda", "mps"):
        return torch.float16
    return torch.float32

# ── Full capabilities (delegates to DeviceRouter) ───────────────────────

@functools.lru_cache(maxsize=1)
def device_caps() -> DeviceCapabilities:
    """
    Get full device capabilities (chip name, core counts, memory, etc.).

    Delegates to the existing DeviceRouter for rich detection.
    Falls back to a basic DeviceCapabilities if DeviceRouter unavailable.
    """
    try:
        from backend.core.optimized.device_router import (
            DeviceCapabilities,
            get_device_router,
        )
        router = get_device_router()
        caps: DeviceCapabilities = router.capabilities
        return caps
    except (ImportError, RuntimeError):
        # Minimal fallback
        from backend.core.optimized.device_router import DeviceCapabilities
        caps = DeviceCapabilities()
        caps.is_apple_silicon = is_apple_silicon()
        caps.has_mps = has_mps()
        caps.has_cuda = has_cuda()
        caps.mlx_available = has_mlx()
        return caps

def empty_gpu_cache() -> None:
    """
    Clear GPU cache for the current device.

    Call ONLY during model load/unload, NOT on hot paths.
    Each call costs ~5ms on MPS.
    """
    device = get_device()
    if device == "mps":
        try:
            import torch
            torch.mps.empty_cache()
        except (ImportError, RuntimeError, OSError):
            pass
    elif device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError, OSError):
            pass
    # MLX manages its own memory — no explicit cache clear needed.
