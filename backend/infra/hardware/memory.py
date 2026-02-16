"""
Memory Manager — Unified Memory Lifecycle
=============================================

Manages GPU/CPU memory allocation for model inference:
  - Memory budget tracking per model
  - Eviction policies (LRU, priority-based)
  - Memory pressure monitoring
  - Integration with existing MemoryCoordinator

This is the hardware layer's memory interface.
The existing core.optimized.memory_coordinator handles
the low-level bookkeeping; this module provides the
clean layer boundary.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from backend.infra.telemetry import get_logger, get_metrics
from backend.utils.lock_factory import create_lock

logger = get_logger(__name__)
metrics = get_metrics()

class MemoryPressure(IntEnum):
    """Memory pressure levels."""

    LOW = 0         # < 60% utilization
    MODERATE = 1    # 60-80%
    HIGH = 2        # 80-90%
    CRITICAL = 3    # 90-95%
    EMERGENCY = 4   # > 95%

@dataclass
class MemoryBudget:
    """Memory budget for a model or subsystem."""

    name: str
    allocated_bytes: int = 0
    used_bytes: int = 0
    max_bytes: int = 0
    priority: int = 5  # 0 = highest, 9 = lowest

    @property
    def utilization(self) -> float:
        if self.max_bytes == 0:
            return 0.0
        return self.used_bytes / self.max_bytes

    @property
    def available_bytes(self) -> int:
        return max(0, self.max_bytes - self.used_bytes)

@dataclass
class MemorySnapshot:
    """Point-in-time memory state."""

    timestamp: float
    total_bytes: int
    used_bytes: int
    available_bytes: int
    gpu_total_bytes: int = 0
    gpu_used_bytes: int = 0
    pressure: MemoryPressure = MemoryPressure.LOW
    active_models: int = 0
    budgets: dict[str, MemoryBudget] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes

    @property
    def gpu_utilization(self) -> float:
        if self.gpu_total_bytes == 0:
            return 0.0
        return self.gpu_used_bytes / self.gpu_total_bytes

class MemoryManager:
    """
    Unified memory management across CPU and GPU.

    Tracks memory budgets, monitors pressure, and coordinates
    eviction when memory is constrained.

    Integrates with the existing MemoryCoordinator for backward compatibility.
    """

    def __init__(self) -> None:
        self._budgets: dict[str, MemoryBudget] = {}
        self._snapshots: list[MemorySnapshot] = []
        self._max_snapshots = 720  # 1 hour at 5s intervals
        self._lock = threading.Lock()
        self._coordinator: Any = None  # Lazy binding to existing MemoryCoordinator

    def _get_coordinator(self) -> Any:
        """Lazy import of existing MemoryCoordinator."""
        if self._coordinator is None:
            try:
                from backend.core.optimized.memory_coordinator import (
                    get_memory_coordinator,
                )
                self._coordinator = get_memory_coordinator()
            except ImportError:
                pass
        return self._coordinator

    def allocate_budget(
        self,
        name: str,
        max_bytes: int,
        priority: int = 5,
    ) -> MemoryBudget:
        """Allocate a memory budget for a model or subsystem."""
        with self._lock:
            budget = MemoryBudget(
                name=name,
                max_bytes=max_bytes,
                priority=priority,
            )
            self._budgets[name] = budget
            logger.info(
                "memory_budget_allocated",
                name=name,
                max_mb=round(max_bytes / (1024 ** 2)),
                priority=priority,
            )
            return budget

    def acquire(self, name: str, size_bytes: int) -> bool:
        """Acquire memory from a budget. Returns False if insufficient."""
        with self._lock:
            budget = self._budgets.get(name)
            if budget is None:
                return True  # No budget tracking

            if budget.used_bytes + size_bytes > budget.max_bytes:
                logger.warning(
                    "memory_budget_exceeded",
                    name=name,
                    requested_mb=round(size_bytes / (1024 ** 2)),
                    available_mb=round(budget.available_bytes / (1024 ** 2)),
                )
                return False

            budget.used_bytes += size_bytes
            budget.allocated_bytes += size_bytes
            return True

    def release(self, name: str, size_bytes: int) -> None:
        """Release memory back to a budget."""
        with self._lock:
            budget = self._budgets.get(name)
            if budget:
                budget.used_bytes = max(0, budget.used_bytes - size_bytes)

    def get_pressure(self) -> MemoryPressure:
        """Get current memory pressure level."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            util = mem.percent / 100.0
        except ImportError:
            return MemoryPressure.LOW

        if util < 0.6:
            return MemoryPressure.LOW
        elif util < 0.8:
            return MemoryPressure.MODERATE
        elif util < 0.9:
            return MemoryPressure.HIGH
        elif util < 0.95:
            return MemoryPressure.CRITICAL
        else:
            return MemoryPressure.EMERGENCY

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            total = mem.total
            used = mem.used
            available = mem.available
        except ImportError:
            total = used = available = 0

        gpu_total = gpu_used = 0
        try:
            from backend.core.hal import get_device

            device = get_device()
            if device == "cuda":
                import torch
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_used = torch.cuda.memory_allocated(0)
            elif device == "mps":
                try:
                    import torch
                    gpu_used = torch.mps.current_allocated_memory()
                    gpu_total = total  # Unified memory
                except (RuntimeError, OSError):
                    gpu_total = total
        except (ImportError, RuntimeError):
            pass

        snapshot = MemorySnapshot(
            timestamp=time.monotonic(),
            total_bytes=total,
            used_bytes=used,
            available_bytes=available,
            gpu_total_bytes=gpu_total,
            gpu_used_bytes=gpu_used,
            pressure=self.get_pressure(),
            active_models=len([b for b in self._budgets.values() if b.used_bytes > 0]),
            budgets=dict(self._budgets),
        )

        with self._lock:
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self._max_snapshots:
                self._snapshots = self._snapshots[-self._max_snapshots:]

        # Report to metrics
        metrics.record_gpu_utilization(
            device=get_device() if gpu_total else "cpu",
            utilization=snapshot.gpu_utilization,
            memory_used=gpu_used,
            memory_total=gpu_total,
        )

        return snapshot

    async def evict_if_needed(self, required_bytes: int = 0) -> bool:
        """
        Evict models if memory pressure is too high.

        Returns True if eviction was performed.
        """
        pressure = self.get_pressure()
        if pressure < MemoryPressure.HIGH and required_bytes == 0:
            return False

        coordinator = self._get_coordinator()
        if coordinator and hasattr(coordinator, "emergency_evict"):
            try:
                await coordinator.emergency_evict()
                logger.info("memory_eviction_triggered", pressure=pressure.name)
                return True
            except (RuntimeError, OSError) as exc:
                logger.error("memory_eviction_failed", exc=exc)

        return False

    def empty_cache(self) -> None:
        """Clear GPU caches."""
        try:
            from backend.core.hal import empty_gpu_cache, get_device
            empty_gpu_cache()
        except ImportError:
            pass

    def get_summary(self) -> dict[str, Any]:
        snapshot = self.take_snapshot()
        return {
            "pressure": snapshot.pressure.name,
            "system": {
                "total_gb": round(snapshot.total_bytes / (1024 ** 3), 1),
                "used_gb": round(snapshot.used_bytes / (1024 ** 3), 1),
                "utilization": round(snapshot.utilization, 3),
            },
            "gpu": {
                "total_gb": round(snapshot.gpu_total_bytes / (1024 ** 3), 1),
                "used_gb": round(snapshot.gpu_used_bytes / (1024 ** 3), 1),
                "utilization": round(snapshot.gpu_utilization, 3),
            },
            "active_models": snapshot.active_models,
            "budgets": {
                name: {
                    "max_mb": round(b.max_bytes / (1024 ** 2)),
                    "used_mb": round(b.used_bytes / (1024 ** 2)),
                    "utilization": round(b.utilization, 3),
                }
                for name, b in self._budgets.items()
            },
        }

# ── Singleton ──────────────────────────────────────────────────────

_manager: MemoryManager | None = None
_lock = create_lock()

def get_memory_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        with _lock:
            if _manager is None:
                _manager = MemoryManager()
    return _manager
