"""
Hardware Abstraction Layer — Unified Device Management
=========================================================

LAYER 4 in the architecture stack.

Provides:
  - Device detection and capability profiling (CPU/GPU/NPU/ANE)
  - Device placement control for model execution
  - Mixed precision management (fp32, fp16, bf16, int8, int4)
  - Memory pooling with zero-copy support
  - GPU warm pools for low-latency inference
  - Dynamic batch size optimization based on hardware

Depends on: telemetry (Layer 6)
Depended on by: execution (Layer 3)

Re-exports key utilities from existing core.optimized modules
while providing a clean, layered interface.
"""

from backend.infra.hardware.device import (
    DeviceInfo,
    DeviceManager,
    device_type,
    get_device,
    get_device_manager,
    has_gpu,
)
from backend.infra.hardware.gpu_pool import (
    GPUWarmPool,
    get_gpu_pool,
)
from backend.infra.hardware.memory import (
    MemoryManager,
    get_memory_manager,
)
from backend.infra.hardware.precision import (
    PrecisionManager,
    PrecisionMode,
    get_precision_manager,
)

__all__ = [
    "DeviceInfo",
    "DeviceManager",
    "GPUWarmPool",
    "MemoryManager",
    "PrecisionManager",
    "PrecisionMode",
    "device_type",
    "get_device",
    "get_device_manager",
    "get_gpu_pool",
    "get_memory_manager",
    "get_precision_manager",
    "has_gpu",
]
