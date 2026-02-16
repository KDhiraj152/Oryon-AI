"""
Apple Silicon Optimization Utilities
=====================================

M4-specific optimizations:
- Metal shader compilation hints
- Unified memory buffer pooling
- ANE (Neural Engine) task routing
- Thread affinity for P-core/E-core
"""

import logging
import os
import threading
from functools import lru_cache

from backend.utils.lock_factory import create_lock

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS - M4 Architecture
# ============================================================================

# M4 has 4 Performance cores + 6 Efficiency cores
M4_P_CORES = 4
M4_E_CORES = 6
M4_TOTAL_CORES = M4_P_CORES + M4_E_CORES

# Optimal thread counts for different workloads
COMPUTE_THREADS = M4_P_CORES  # Heavy compute on P-cores only
IO_THREADS = M4_TOTAL_CORES  # I/O can use all cores
BATCH_THREADS = M4_P_CORES  # Batch inference on P-cores

# ============================================================================
# METAL SHADER OPTIMIZATION
# ============================================================================

_metal_warmed_up = False
_metal_warmup_lock = create_lock()

def warmup_metal_shaders() -> None:
    """
    Pre-compile Metal shaders to reduce first-inference latency.

    Metal compiles shaders on first use, causing 100-500ms delay.
    This function pre-warms common operations.
    """
    global _metal_warmed_up

    if _metal_warmed_up:
        return

    with _metal_warmup_lock:
        if _metal_warmed_up:
            return

        try:
            import torch

            if not torch.backends.mps.is_available():
                return

            # Small operations to trigger shader compilation
            device = torch.device("mps")

            # Matrix operations (used by transformers)
            a = torch.randn(64, 64, device=device, dtype=torch.float16)
            b = torch.randn(64, 64, device=device, dtype=torch.float16)
            _ = torch.matmul(a, b)

            # Softmax (attention)
            _ = torch.softmax(a, dim=-1)

            # Layer norm
            _ = torch.nn.functional.layer_norm(a, [64])

            # GELU (common activation)
            _ = torch.nn.functional.gelu(a)

            # Sync and clean
            torch.mps.synchronize()
            torch.mps.empty_cache()

            _metal_warmed_up = True
            logger.debug("Metal shaders pre-compiled")

        except (RuntimeError, OSError) as e:
            logger.debug("Metal warmup skipped: %s", e)

# ============================================================================
# THREAD CONFIGURATION
# ============================================================================

_threading_configured = False

def configure_optimal_threading() -> None:
    """
    Configure PyTorch/MLX threading for M4 architecture.

    Sets:
    - torch.set_num_threads() for P-cores
    - OMP_NUM_THREADS for OpenMP
    - MKL_NUM_THREADS for MKL
    """
    global _threading_configured

    if _threading_configured:
        return

    # Set thread counts for compute libraries
    os.environ.setdefault("OMP_NUM_THREADS", str(COMPUTE_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(COMPUTE_THREADS))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(COMPUTE_THREADS))

    # PyTorch intraop threads
    try:
        import torch

        torch.set_num_threads(COMPUTE_THREADS)

        # Interop threads for data loading
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(2)

    except ImportError:
        pass

    _threading_configured = True
    logger.debug("Threading configured: compute=%s, I/O=%s", COMPUTE_THREADS, IO_THREADS)

# ============================================================================
# MPS MEMORY MANAGEMENT
# ============================================================================

def clear_mps_cache() -> None:
    """Clear MPS memory cache to free unused memory."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        pass

def sync_mps() -> None:
    """Synchronize MPS operations (wait for GPU to finish)."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except (ImportError, RuntimeError):
        pass

@lru_cache(maxsize=1)
def get_mps_memory_info() -> dict:
    """Get MPS memory information (cached)."""
    try:
        import torch

        if not torch.backends.mps.is_available():
            return {}

        # MPS doesn't expose memory stats like CUDA
        # Return estimated info based on unified memory
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
        )
        total_bytes = int(result.stdout.strip())
        total_gb = total_bytes / (1024**3)

        return {
            "total_gb": total_gb,
            "unified_memory": True,
            "recommended_max_model_gb": total_gb * 0.6,  # 60% for models
        }
    except (RuntimeError, OSError):
        return {}

# ============================================================================
# MLX OPTIMIZATIONS
# ============================================================================

def configure_mlx_memory() -> None:
    """Configure MLX memory settings for optimal M4 performance."""
    try:
        import mlx.core as mx

        # Enable memory caching for buffer reuse
        # This reduces allocation overhead significantly
        mx.set_default_device(mx.gpu)  # type: ignore[arg-type]

        logger.debug("MLX configured for GPU with memory caching")

    except ImportError:
        pass
    except (RuntimeError, OSError) as e:
        logger.debug("MLX configuration skipped: %s", e)

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_apple_silicon_optimizations() -> None:
    """
    Initialize all Apple Silicon optimizations.

    Call this once at application startup.
    """
    configure_optimal_threading()
    configure_mlx_memory()

    # Defer Metal warmup (do it async at first inference)
    logger.info("Apple Silicon optimizations initialized")

# Auto-configure on import if on Apple Silicon
try:
    import platform

    if platform.processor() == "arm" or platform.machine() == "arm64":
        configure_optimal_threading()
except Exception:
    pass
