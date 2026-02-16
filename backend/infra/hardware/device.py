"""
Device Manager — Hardware Detection and Placement
=====================================================

Provides:
  - Automatic hardware detection (CPU, CUDA, MPS, ANE, MLX)
  - Device capability profiling (memory, compute, precision)
  - Placement decisions for model loading
  - Runtime device switching

Wraps and extends the existing core.hal module with
a proper object-oriented interface.
"""

from __future__ import annotations

import functools
import os
import platform
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from backend.infra.telemetry import get_logger

logger = get_logger(__name__)

class DeviceType(StrEnum):
    """Supported device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    MLX = "mlx"
    ANE = "ane"  # Apple Neural Engine

@dataclass
class DeviceInfo:
    """Detailed device information."""

    device_type: DeviceType
    name: str = "unknown"
    compute_capability: str = ""
    total_memory_bytes: int = 0
    available_memory_bytes: int = 0
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_int8: bool = False
    supports_int4: bool = False
    is_unified_memory: bool = False
    num_cores: int = 0
    clock_mhz: int = 0
    driver_version: str = ""
    chip_family: str = ""  # M1, M2, M3, M4, A100, H100, etc.

    @property
    def memory_gb(self) -> float:
        return self.total_memory_bytes / (1024 ** 3)

    @property
    def available_gb(self) -> float:
        return self.available_memory_bytes / (1024 ** 3)

    @property
    def memory_utilization(self) -> float:
        if self.total_memory_bytes == 0:
            return 0.0
        return 1.0 - (self.available_memory_bytes / self.total_memory_bytes)

class DeviceManager:
    """
    Centralized device management.

    Discovers hardware capabilities at startup and provides
    placement decisions for model loading and inference.
    """

    def __init__(self) -> None:
        self._devices: dict[DeviceType, DeviceInfo] = {}
        self._primary: DeviceType = DeviceType.CPU
        self._lock = threading.Lock()
        self._detected = False

    def detect(self) -> None:
        """Detect all available devices."""
        if self._detected:
            return

        with self._lock:
            if self._detected:
                return

            # Always have CPU
            self._devices[DeviceType.CPU] = self._detect_cpu()

            # Check GPU
            if self._check_cuda():
                self._devices[DeviceType.CUDA] = self._detect_cuda()
                self._primary = DeviceType.CUDA
            elif self._check_mps():
                self._devices[DeviceType.MPS] = self._detect_mps()
                self._primary = DeviceType.MPS
            elif self._check_mlx():
                self._devices[DeviceType.MLX] = self._detect_mlx()
                self._primary = DeviceType.MLX

            # Check ANE (Apple Neural Engine)
            if self._check_ane():
                self._devices[DeviceType.ANE] = self._detect_ane()

            self._detected = True
            logger.info(
                "devices_detected",
                primary=self._primary.value,
                devices=[d.value for d in self._devices],
            )

    @property
    def primary_device(self) -> DeviceType:
        """Get the primary compute device."""
        self.detect()
        return self._primary

    @property
    def primary_info(self) -> DeviceInfo:
        """Get info for the primary device."""
        self.detect()
        return self._devices[self._primary]

    def get_device(self, device_type: DeviceType) -> DeviceInfo | None:
        """Get info for a specific device."""
        self.detect()
        return self._devices.get(device_type)

    def get_all_devices(self) -> dict[DeviceType, DeviceInfo]:
        """Get all detected devices."""
        self.detect()
        return dict(self._devices)

    def get_placement(
        self,
        *,
        model_type: str,
        model_size_bytes: int = 0,
        requires_fp16: bool = False,
        prefer_device: DeviceType | None = None,
    ) -> DeviceType:
        """
        Determine optimal device for model placement.

        Args:
            model_type: Type of model (llm, embedding, etc.)
            model_size_bytes: Estimated model memory footprint
            requires_fp16: Whether fp16 is required
            prefer_device: User-preferred device
        """
        self.detect()

        if prefer_device and prefer_device in self._devices:
            info = self._devices[prefer_device]
            if model_size_bytes > 0 and info.available_memory_bytes >= model_size_bytes:
                return prefer_device

        # Try primary device first
        primary = self._devices[self._primary]
        if ((model_size_bytes == 0 or primary.available_memory_bytes >= model_size_bytes)
                and (not requires_fp16 or primary.supports_fp16)):
            return self._primary

        # Fallback to CPU
        return DeviceType.CPU

    def refresh_memory(self) -> None:
        """Refresh memory statistics for all devices."""
        for dtype, info in self._devices.items():
            if dtype == DeviceType.CUDA:
                self._refresh_cuda_memory(info)
            elif dtype == DeviceType.MPS:
                self._refresh_mps_memory(info)

    # ── Detection Helpers ──────────────────────────────────────────

    def _detect_cpu(self) -> DeviceInfo:
        import psutil

        return DeviceInfo(
            device_type=DeviceType.CPU,
            name=platform.processor() or platform.machine(),
            total_memory_bytes=psutil.virtual_memory().total,
            available_memory_bytes=psutil.virtual_memory().available,
            supports_fp16=False,
            supports_bf16=False,
            num_cores=os.cpu_count() or 1,
            chip_family="apple_silicon" if self._is_apple_silicon() else "x86",
        )

    @staticmethod
    def _is_apple_silicon() -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _check_mps() -> bool:
        try:
            import torch
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except ImportError:
            return False

    @staticmethod
    def _check_mlx() -> bool:
        try:
            import mlx.core
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_ane() -> bool:
        """Check for Apple Neural Engine (CoreML backend)."""
        if platform.system() != "Darwin":
            return False
        try:
            import coremltools
            return True
        except ImportError:
            return False

    def _detect_cuda(self) -> DeviceInfo:
        import torch

        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            device_type=DeviceType.CUDA,
            name=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            total_memory_bytes=props.total_memory,
            available_memory_bytes=torch.cuda.mem_get_info(0)[0],
            supports_fp16=True,
            supports_bf16=props.major >= 8,
            supports_int8=True,
            supports_int4=props.major >= 8,
            num_cores=props.multi_processor_count,
            clock_mhz=getattr(props, 'clock_rate', 0) // 1000,
            driver_version=torch.version.cuda or "",
            chip_family=props.name.split()[0],
        )

    def _detect_mps(self) -> DeviceInfo:
        import psutil

        # MPS uses unified memory (shared with CPU)
        mem = psutil.virtual_memory()
        cpu_count = os.cpu_count() or 1
        chip = "M4" if cpu_count >= 10 else "Apple Silicon"

        # Try to detect chip family
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2
            )
            brand = result.stdout.strip()
            for family in ("M4", "M3", "M2", "M1"):
                if family in brand:
                    chip = family
                    break
        except (RuntimeError, OSError):
            pass

        return DeviceInfo(
            device_type=DeviceType.MPS,
            name=f"Apple {chip} GPU",
            total_memory_bytes=mem.total,
            available_memory_bytes=mem.available,
            supports_fp16=True,
            supports_bf16=False,
            supports_int8=True,
            supports_int4=chip in ("M3", "M4"),
            is_unified_memory=True,
            chip_family=chip,
        )

    def _detect_mlx(self) -> DeviceInfo:
        import psutil

        mem = psutil.virtual_memory()
        return DeviceInfo(
            device_type=DeviceType.MLX,
            name="MLX Accelerator",
            total_memory_bytes=mem.total,
            available_memory_bytes=mem.available,
            supports_fp16=True,
            supports_bf16=True,
            supports_int8=True,
            supports_int4=True,
            is_unified_memory=True,
            chip_family="apple_silicon",
        )

    def _detect_ane(self) -> DeviceInfo:
        import psutil

        return DeviceInfo(
            device_type=DeviceType.ANE,
            name="Apple Neural Engine",
            total_memory_bytes=psutil.virtual_memory().total,
            available_memory_bytes=psutil.virtual_memory().available,
            supports_fp16=True,
            is_unified_memory=True,
            chip_family="apple_silicon",
        )

    def _refresh_cuda_memory(self, info: DeviceInfo) -> None:
        try:
            import torch
            info.available_memory_bytes = torch.cuda.mem_get_info(0)[0]
        except (RuntimeError, OSError):
            pass

    def _refresh_mps_memory(self, info: DeviceInfo) -> None:
        try:
            import psutil
            info.available_memory_bytes = psutil.virtual_memory().available
        except (RuntimeError, OSError):
            pass

    def get_summary(self) -> dict[str, Any]:
        self.detect()
        return {
            "primary": self._primary.value,
            "devices": {
                d.value: {
                    "name": info.name,
                    "memory_gb": round(info.memory_gb, 1),
                    "available_gb": round(info.available_gb, 1),
                    "fp16": info.supports_fp16,
                    "int8": info.supports_int8,
                    "unified": info.is_unified_memory,
                    "chip": info.chip_family,
                }
                for d, info in self._devices.items()
            },
        }

# ── Convenience Functions ──────────────────────────────────────────

_manager: DeviceManager | None = None

def get_device_manager() -> DeviceManager:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _manager
    if _manager is not None:
        return _manager
    _manager = DeviceManager()
    return _manager
def get_device() -> str:
    """Get the primary device name (backward compatible)."""
    return get_device_manager().primary_device.value

def has_gpu() -> bool:
    """Check if any GPU is available."""
    mgr = get_device_manager()
    mgr.detect()
    return any(
        d in mgr._devices
        for d in (DeviceType.CUDA, DeviceType.MPS, DeviceType.MLX)
    )

def device_type() -> DeviceType:
    """Get the primary device type."""
    return get_device_manager().primary_device
