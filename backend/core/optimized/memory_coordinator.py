"""
Global Memory Coordinator - Centralized Model Memory Management
================================================================

CRITICAL FIX: Prevents OOM by coordinating all model loading across the system.

This module solves the following issues:
1. Race conditions when multiple models load simultaneously
2. Memory exhaustion from uncoordinated model loading
3. Lack of memory pressure handling
4. No eviction policy for under-utilized models

Design Principles:
- Single point of control for ALL model memory allocation
- Event-loop safe async primitives
- Memory pressure callbacks before OOM
- LRU eviction with grace periods
- Proper cleanup on shutdown

M4 Memory Budget (16GB Unified):
- OS Reserved: 2.5GB
- Model Weights: 8-10GB (shared across all models)
- Inference Buffers: 3-4GB (dynamic)
- Cache: 1-2GB
- Headroom: 1GB

Usage:
    coordinator = get_memory_coordinator()

    # Request memory before loading
    async with coordinator.acquire_memory("embedding_model", 2.0):
        model = load_embedding_model()  # Actually load
        coordinator.register_model("embedding_model", model, unload_fn)
"""

import asyncio
import gc
import logging
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


class MemoryPressure(Enum):
    """Memory pressure levels."""

    NORMAL = "normal"  # < 70% used
    WARNING = "warning"  # 70-85% used
    CRITICAL = "critical"  # 85-95% used
    EMERGENCY = "emergency"  # > 95% used


class ModelState(Enum):
    """Model lifecycle states."""

    PENDING = "pending"  # Memory reserved, not yet loaded
    LOADING = "loading"  # Currently loading
    READY = "ready"  # Loaded and ready
    EVICTING = "evicting"  # Being evicted
    UNLOADED = "unloaded"  # Memory released


@dataclass
class MemoryBudgetConfig:
    """Memory budget configuration for M4."""

    total_gb: float = 16.0
    os_reserved_gb: float = 2.5
    max_model_memory_gb: float = 10.0
    inference_buffer_gb: float = 2.5
    cache_gb: float = 1.5
    headroom_gb: float = 0.5  # Always keep free

    # Pressure thresholds (percentage of available)
    warning_threshold: float = 0.70
    critical_threshold: float = 0.85
    emergency_threshold: float = 0.95

    # Eviction settings
    eviction_grace_period_s: float = 300.0  # 5 minutes
    min_model_age_for_eviction_s: float = 60.0  # 1 minute

    @property
    def available_for_models_gb(self) -> float:
        """Memory available for model loading."""
        return self.total_gb - self.os_reserved_gb - self.headroom_gb

    def validate(self) -> bool:
        """Validate budget adds up."""
        used = (
            self.os_reserved_gb
            + self.max_model_memory_gb
            + self.inference_buffer_gb
            + self.cache_gb
            + self.headroom_gb
        )
        return used <= self.total_gb


@dataclass
class ModelRegistration:
    """Registration info for a managed model."""

    name: str
    memory_gb: float
    model_ref: weakref.ref  # Weak reference to actual model
    unload_fn: Callable[[], None] | None
    state: ModelState = ModelState.PENDING
    priority: int = 0  # Higher = less likely to evict
    load_time: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0

    def touch(self):
        """Mark model as recently used."""
        self.last_used = time.time()
        self.use_count += 1

    @property
    def age_seconds(self) -> float:
        """Time since load."""
        return time.time() - self.load_time

    @property
    def idle_seconds(self) -> float:
        """Time since last use."""
        return time.time() - self.last_used


class GlobalMemoryCoordinator:
    """
    Singleton coordinator for all model memory management.

    Thread-safe and event-loop safe. Uses proper async primitives
    that are created within the correct event loop context.

    Key Principles:
    1. All model loads MUST go through acquire_memory()
    2. Models are tracked via weak references
    3. Memory pressure triggers automatic eviction
    4. Shutdown properly cleans up all resources

    FIXED: C2 - Async primitives now properly created per event loop with cleanup.
    """

    _instance = None
    _creation_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton for testing or reinitialization."""
        with cls._creation_lock:
            if cls._instance is not None:
                cls._instance._shutdown = True
                cls._instance._cleanup_async_primitives()
            cls._instance = None

    def __init__(self):
        if self._initialized:
            return

        # Configuration
        self.config = self._detect_memory_config()

        # Thread-safe state
        self._lock = threading.RLock()  # Reentrant for nested calls
        self._models: dict[str, ModelRegistration] = OrderedDict()
        self._ready_models: dict[str, ModelRegistration] = (
            OrderedDict()
        )  # FIX: Track ready models separately (renamed to avoid property conflict)
        self._allocated_gb: float = 0.0
        self._pending_gb: float = 0.0  # Reserved but not yet loaded

        # Async primitives - thread-safe storage with proper locking
        # FIX C2: Use weak event loop references to prevent stale locks
        self._async_lock_storage = threading.Lock()
        self._async_locks: dict[int, asyncio.Lock] = {}
        self._async_semaphores: dict[int, asyncio.Semaphore] = {}
        self._loop_refs: dict[int, weakref.ref] = {}  # Track event loops for cleanup

        # Memory pressure callbacks
        self._pressure_callbacks: list[Callable[[MemoryPressure], None]] = []

        # Eviction queue (model names to evict)
        self._eviction_queue: list[str] = []

        # Shutdown flag
        self._shutdown = False

        # Background monitor task
        self._monitor_task: asyncio.Task | None = None

        self._initialized = True
        logger.info(
            f"[MemoryCoordinator] Initialized with {self.config.available_for_models_gb:.1f}GB available"
        )

    def _cleanup_async_primitives(self):
        """Clean up async primitives from dead event loops."""
        with self._async_lock_storage:
            dead_loops = []
            for loop_id, loop_ref in self._loop_refs.items():
                if loop_ref() is None:
                    dead_loops.append(loop_id)
            for loop_id in dead_loops:
                self._async_locks.pop(loop_id, None)
                self._async_semaphores.pop(loop_id, None)
                self._loop_refs.pop(loop_id, None)

    def _detect_memory_config(self) -> MemoryBudgetConfig:
        """Detect system memory and create config."""
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
        except Exception:
            total_gb = 16.0  # Default for M4

        # Scale config based on available memory
        if total_gb >= 32:
            return MemoryBudgetConfig(
                total_gb=total_gb,
                os_reserved_gb=4.0,
                max_model_memory_gb=20.0,
                inference_buffer_gb=5.0,
                cache_gb=3.0,
                headroom_gb=1.0,
            )
        elif total_gb >= 16:
            return MemoryBudgetConfig(
                total_gb=total_gb,
                os_reserved_gb=2.5,
                max_model_memory_gb=10.0,
                inference_buffer_gb=2.5,
                cache_gb=1.5,
                headroom_gb=0.5,
            )
        else:
            # 8GB or less - very conservative
            return MemoryBudgetConfig(
                total_gb=total_gb,
                os_reserved_gb=2.0,
                max_model_memory_gb=4.0,
                inference_buffer_gb=1.0,
                cache_gb=0.5,
                headroom_gb=0.5,
            )

    def _get_async_lock(self) -> asyncio.Lock:
        """Get async lock for current event loop - thread-safe.

        FIXED C2: Properly creates locks within the running event loop context
        and tracks loop references for cleanup.
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            # No running loop - this shouldn't happen in async context
            raise RuntimeError(
                "Cannot get async lock without running event loop. "
                "Use acquire_memory_sync() for synchronous contexts."
            )

        with self._async_lock_storage:
            # Clean up dead loops periodically
            if len(self._loop_refs) > 10:
                self._cleanup_async_primitives()

            if loop_id not in self._async_locks:
                self._async_locks[loop_id] = asyncio.Lock()
                self._loop_refs[loop_id] = weakref.ref(loop)
            return self._async_locks[loop_id]

    def _get_async_semaphore(self, max_concurrent: int = 2) -> asyncio.Semaphore:
        """Get async semaphore for current event loop (max concurrent model loads).

        FIXED C2: Thread-safe semaphore creation with event loop tracking.
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            raise RuntimeError("Cannot get async semaphore without running event loop.")

        with self._async_lock_storage:
            if loop_id not in self._async_semaphores:
                self._async_semaphores[loop_id] = asyncio.Semaphore(max_concurrent)
                if loop_id not in self._loop_refs:
                    self._loop_refs[loop_id] = weakref.ref(loop)
            return self._async_semaphores[loop_id]

    # =========================================================================
    # Memory Pressure Monitoring
    # =========================================================================

    def get_memory_pressure(self) -> MemoryPressure:
        """Get current memory pressure level."""
        try:
            mem = psutil.virtual_memory()
            used_ratio = mem.percent / 100.0
        except Exception:
            # Fallback to our internal tracking
            used_ratio = (
                self._allocated_gb + self._pending_gb
            ) / self.config.available_for_models_gb

        if used_ratio >= self.config.emergency_threshold:
            return MemoryPressure.EMERGENCY
        elif used_ratio >= self.config.critical_threshold:
            return MemoryPressure.CRITICAL
        elif used_ratio >= self.config.warning_threshold:
            return MemoryPressure.WARNING
        return MemoryPressure.NORMAL

    def get_available_memory_gb(self) -> float:
        """Get memory available for new model loads."""
        with self._lock:
            used = self._allocated_gb + self._pending_gb
            return max(0, self.config.max_model_memory_gb - used)

    def get_system_memory_gb(self) -> tuple[float, float]:
        """Get (used, total) system memory in GB."""
        try:
            mem = psutil.virtual_memory()
            return (mem.used / (1024**3), mem.total / (1024**3))
        except Exception:
            return (self._allocated_gb, self.config.total_gb)

    # =========================================================================
    # Model Memory Acquisition
    # =========================================================================

    @asynccontextmanager
    async def acquire_memory(
        self,
        model_name: str,
        memory_gb: float,
        priority: int = 0,
        timeout: float = 60.0,
    ):
        """
        Async context manager to acquire memory for model loading.

        Usage:
            async with coordinator.acquire_memory("llm", 4.5):
                model = load_model()  # Actually load
                coordinator.register_model("llm", model, unload_fn)

        Raises:
            MemoryError: If memory cannot be acquired after timeout
        """
        if self._shutdown:
            raise RuntimeError("MemoryCoordinator is shutting down")

        async_lock = self._get_async_lock()
        load_semaphore = self._get_async_semaphore(max_concurrent=2)

        # First acquire semaphore to limit concurrent loads
        async with load_semaphore:
            acquired = False
            start_time = time.time()

            try:
                while not acquired:
                    # Check timeout
                    if time.time() - start_time > timeout:
                        raise MemoryError(
                            f"Timeout acquiring {memory_gb:.1f}GB for {model_name}. "
                            f"Available: {self.get_available_memory_gb():.1f}GB"
                        )

                    async with async_lock:
                        available = self.get_available_memory_gb()

                        if memory_gb <= available:
                            # Reserve memory
                            with self._lock:
                                self._pending_gb += memory_gb
                                self._models[model_name] = ModelRegistration(
                                    name=model_name,
                                    memory_gb=memory_gb,
                                    model_ref=weakref.ref(lambda: None),  # Placeholder
                                    unload_fn=None,
                                    state=ModelState.LOADING,
                                    priority=priority,
                                )
                            acquired = True
                            logger.info(
                                f"[MemoryCoordinator] Reserved {memory_gb:.1f}GB for {model_name}"
                            )
                        else:
                            # Need to evict something
                            pressure = self.get_memory_pressure()
                            if pressure in (
                                MemoryPressure.CRITICAL,
                                MemoryPressure.EMERGENCY,
                            ):
                                await self._evict_for_memory(memory_gb - available)
                            else:
                                # Wait and retry
                                await asyncio.sleep(0.5)

                yield  # Allow model loading

            except Exception:
                # Release reserved memory on failure
                with self._lock:
                    self._pending_gb = max(0, self._pending_gb - memory_gb)
                    if model_name in self._models:
                        self._models[model_name].state = ModelState.UNLOADED
                raise

    @contextmanager
    def acquire_memory_sync(
        self,
        model_name: str,
        memory_gb: float,
        priority: int = 0,
    ):
        """
        Synchronous version of acquire_memory for non-async contexts.

        WARNING: This may block. Prefer async version when possible.
        """
        if self._shutdown:
            raise RuntimeError("MemoryCoordinator is shutting down")

        with self._lock:
            available = self.get_available_memory_gb()

            if memory_gb > available:
                # Try eviction synchronously
                self._evict_for_memory_sync(memory_gb - available)
                available = self.get_available_memory_gb()

            if memory_gb > available:
                raise MemoryError(
                    f"Insufficient memory for {model_name}: need {memory_gb:.1f}GB, "
                    f"available {available:.1f}GB"
                )

            self._pending_gb += memory_gb
            self._models[model_name] = ModelRegistration(
                name=model_name,
                memory_gb=memory_gb,
                model_ref=weakref.ref(lambda: None),
                unload_fn=None,
                state=ModelState.LOADING,
                priority=priority,
            )
            logger.info(
                f"[MemoryCoordinator] Reserved {memory_gb:.1f}GB for {model_name}"
            )

        try:
            yield
        except Exception:
            with self._lock:
                self._pending_gb = max(0, self._pending_gb - memory_gb)
                if model_name in self._models:
                    self._models[model_name].state = ModelState.UNLOADED
            raise

    # =========================================================================
    # Model Registration
    # =========================================================================

    def register_model(
        self,
        model_name: str,
        model: Any,
        unload_fn: Callable[[], None] | None = None,
    ):
        """
        Register a loaded model after successful loading.

        Call this after model is fully loaded and warmed up.
        """
        with self._lock:
            if model_name not in self._models:
                raise ValueError(
                    f"Model {model_name} was not acquired via acquire_memory()"
                )

            reg = self._models[model_name]

            # Move from pending to allocated
            self._pending_gb = max(0, self._pending_gb - reg.memory_gb)
            self._allocated_gb += reg.memory_gb

            # Update registration
            reg.model_ref = weakref.ref(model)
            reg.unload_fn = unload_fn
            reg.state = ModelState.READY
            reg.load_time = time.time()
            reg.last_used = time.time()

            logger.info(
                f"[MemoryCoordinator] Registered {model_name} "
                f"({reg.memory_gb:.1f}GB, total: {self._allocated_gb:.1f}GB)"
            )

    def touch_model(self, model_name: str):
        """Mark model as recently used (call on each inference)."""
        with self._lock:
            if model_name in self._models:
                self._models[model_name].touch()

    def get_model(self, model_name: str) -> Any | None:
        """Get model reference and touch it."""
        with self._lock:
            if model_name not in self._models:
                return None

            reg = self._models[model_name]
            if reg.state != ModelState.READY:
                return None

            model = reg.model_ref()
            if model is None:
                # Model was garbage collected
                reg.state = ModelState.UNLOADED
                return None

            reg.touch()
            return model

    def is_loaded(self, model_name: str) -> bool:
        """Check if model is loaded and ready."""
        with self._lock:
            if model_name not in self._models:
                return False
            return self._models[model_name].state == ModelState.READY

    # =========================================================================
    # Eviction
    # =========================================================================

    def _get_eviction_candidates(self) -> list[tuple[str, ModelRegistration]]:
        """Get models sorted by eviction priority (most evictable first)."""
        with self._lock:
            candidates = []
            for name, reg in self._models.items():
                if reg.state != ModelState.READY:
                    continue
                if reg.age_seconds < self.config.min_model_age_for_eviction_s:
                    continue
                candidates.append((name, reg))

            # Sort by: priority ASC, idle_seconds DESC, use_count ASC
            candidates.sort(
                key=lambda x: (x[1].priority, -x[1].idle_seconds, x[1].use_count)
            )
            return candidates

    async def _evict_for_memory(self, needed_gb: float):
        """Evict models to free up memory (async)."""
        candidates = self._get_eviction_candidates()
        freed = 0.0

        for name, reg in candidates:
            if freed >= needed_gb:
                break

            logger.info(
                f"[MemoryCoordinator] Evicting {name} to free {reg.memory_gb:.1f}GB"
            )
            await self._unload_model_async(name)
            freed += reg.memory_gb

        if freed < needed_gb:
            logger.warning(
                f"[MemoryCoordinator] Could only free {freed:.1f}GB of {needed_gb:.1f}GB needed"
            )

        # Force garbage collection
        gc.collect()

    def _evict_for_memory_sync(self, needed_gb: float):
        """Evict models to free up memory (sync)."""
        candidates = self._get_eviction_candidates()
        freed = 0.0

        for name, reg in candidates:
            if freed >= needed_gb:
                break

            logger.info(
                f"[MemoryCoordinator] Evicting {name} to free {reg.memory_gb:.1f}GB"
            )
            self._unload_model_sync(name)
            freed += reg.memory_gb

        gc.collect()

    async def _unload_model_async(self, model_name: str):
        """Unload a model asynchronously."""
        with self._lock:
            if model_name not in self._models:
                return

            reg = self._models[model_name]
            if reg.state != ModelState.READY:
                return

            reg.state = ModelState.EVICTING

        # Call unload function outside lock
        if reg.unload_fn:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, reg.unload_fn)
            except Exception as e:
                logger.error(f"Error unloading {model_name}: {e}")

        with self._lock:
            self._allocated_gb = max(0, self._allocated_gb - reg.memory_gb)
            reg.state = ModelState.UNLOADED
            reg.model_ref = weakref.ref(lambda: None)

    def _unload_model_sync(self, model_name: str):
        """Unload a model synchronously."""
        with self._lock:
            if model_name not in self._models:
                return

            reg = self._models[model_name]
            if reg.state != ModelState.READY:
                return

            reg.state = ModelState.EVICTING

        if reg.unload_fn:
            try:
                reg.unload_fn()
            except Exception as e:
                logger.error(f"Error unloading {model_name}: {e}")

        with self._lock:
            self._allocated_gb = max(0, self._allocated_gb - reg.memory_gb)
            reg.state = ModelState.UNLOADED
            reg.model_ref = weakref.ref(lambda: None)

    # =========================================================================
    # Pressure Callbacks
    # =========================================================================

    def register_pressure_callback(self, callback: Callable[[MemoryPressure], None]):
        """Register callback for memory pressure changes."""
        self._pressure_callbacks.append(callback)

    def _notify_pressure_change(self, pressure: MemoryPressure):
        """Notify all registered callbacks of pressure change."""
        for callback in self._pressure_callbacks:
            try:
                callback(pressure)
            except Exception as e:
                logger.error(f"Pressure callback error: {e}")

    # =========================================================================
    # Background Monitor
    # =========================================================================

    async def start_monitor(self, interval: float = 5.0):
        """Start background memory monitor."""
        if self._monitor_task is not None:
            return

        async def monitor_loop():
            last_pressure = MemoryPressure.NORMAL
            while not self._shutdown:
                try:
                    pressure = self.get_memory_pressure()

                    if pressure != last_pressure:
                        logger.info(f"[MemoryCoordinator] Pressure: {pressure.value}")
                        self._notify_pressure_change(pressure)
                        last_pressure = pressure

                    if pressure == MemoryPressure.EMERGENCY:
                        # Emergency eviction
                        await self._evict_for_memory(1.0)  # Try to free 1GB
                    elif pressure == MemoryPressure.CRITICAL:
                        # Cleanup garbage
                        gc.collect()

                except Exception as e:
                    logger.error(f"Monitor error: {e}")

                await asyncio.sleep(interval)

        self._monitor_task = asyncio.create_task(monitor_loop())
        logger.info("[MemoryCoordinator] Background monitor started")

    # =========================================================================
    # Shutdown
    # =========================================================================

    async def shutdown(self):
        """Gracefully shutdown coordinator and unload all models."""
        logger.info("[MemoryCoordinator] Shutting down...")
        self._shutdown = True

        # Cancel monitor
        if self._monitor_task:
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task

        # Unload all models
        model_names = list(self._models.keys())
        for name in model_names:
            await self._unload_model_async(name)

        # Clear async primitives
        self._async_locks.clear()
        self._async_semaphores.clear()

        logger.info("[MemoryCoordinator] Shutdown complete")

    def shutdown_sync(self):
        """Synchronous shutdown."""
        self._shutdown = True

        model_names = list(self._models.keys())
        for name in model_names:
            self._unload_model_sync(name)

        gc.collect()

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get coordinator status for monitoring."""
        with self._lock:
            used_mem, total_mem = self.get_system_memory_gb()

            return {
                "pressure": self.get_memory_pressure().value,
                "system_memory_gb": {
                    "used": round(used_mem, 2),
                    "total": round(total_mem, 2),
                    "percent": round((used_mem / total_mem) * 100, 1)
                    if total_mem > 0
                    else 0,
                },
                "model_memory_gb": {
                    "allocated": round(self._allocated_gb, 2),
                    "pending": round(self._pending_gb, 2),
                    "available": round(self.get_available_memory_gb(), 2),
                    "limit": round(self.config.max_model_memory_gb, 2),
                },
                "models": {
                    name: {
                        "state": reg.state.value,
                        "memory_gb": round(reg.memory_gb, 2),
                        "priority": reg.priority,
                        "idle_seconds": round(reg.idle_seconds, 1),
                        "use_count": reg.use_count,
                    }
                    for name, reg in self._models.items()
                },
                "config": {
                    "total_gb": self.config.total_gb,
                    "max_model_memory_gb": self.config.max_model_memory_gb,
                    "warning_threshold": self.config.warning_threshold,
                    "critical_threshold": self.config.critical_threshold,
                },
            }

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics (simplified version of get_status)."""
        with self._lock:
            return {
                "used_memory_gb": round(self._allocated_gb + self._pending_gb, 2),
                "available_memory_gb": round(self.get_available_memory_gb(), 2),
                "loaded_models": {
                    name: round(reg.memory_gb, 2)
                    for name, reg in self._models.items()
                    if reg.state == ModelState.READY
                },
            }

    @property
    def available_memory_gb(self) -> float:
        """Property for available memory."""
        return self.get_available_memory_gb()

    @property
    def _loaded_models(self) -> dict[str, ModelRegistration]:
        """Property for loaded models (for shutdown logging)."""
        with self._lock:
            return {
                name: reg
                for name, reg in self._models.items()
                if reg.state == ModelState.READY
            }

    # =========================================================================
    # Simplified Acquire/Release API for Singletons
    # =========================================================================

    # Model name -> budget mapping (must match what's in rag.py and other services)
    MODEL_BUDGETS = {
        "llm": 4.5,
        "embedder": 2.0,
        "reranker": 1.5,
        "translation": 2.0,
        "tts": 0.5,
        "stt": 2.0,
        "ocr": 1.5,
        "validator": 1.5,
    }

    async def acquire(self, model_name: str, timeout: float = 30.0) -> bool:
        """
        Simplified async acquire for singleton model loaders.

        This reserves memory for a model before loading it.
        Call release() if loading fails, or register the model if successful.

        Args:
            model_name: Name of model (must be in MODEL_BUDGETS)
            timeout: Maximum time to wait for memory

        Returns:
            True if acquired, False if timeout or error
        """
        memory_gb = self.MODEL_BUDGETS.get(model_name, 1.0)

        if self._shutdown:
            return False

        async_lock = self._get_async_lock()
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout acquiring memory for {model_name}")
                return False

            async with async_lock:
                available = self.get_available_memory_gb()

                if memory_gb <= available:
                    with self._lock:
                        self._pending_gb += memory_gb
                        self._models[model_name] = ModelRegistration(
                            name=model_name,
                            memory_gb=memory_gb,
                            model_ref=weakref.ref(lambda: None),
                            unload_fn=None,
                            state=ModelState.LOADING,
                            priority=1,
                        )
                    logger.info(
                        f"[MemoryCoordinator] Acquired {memory_gb:.1f}GB for {model_name}"
                    )
                    return True
                else:
                    # Try to evict for more space
                    await self._evict_for_memory(memory_gb - available)

            await asyncio.sleep(0.1)

    def try_acquire_sync(self, model_name: str) -> bool:
        """
        Synchronous non-blocking acquire for model loading.

        Tries to acquire memory immediately. If not available,
        attempts eviction and retries once.

        Returns:
            True if acquired, False otherwise
        """
        memory_gb = self.MODEL_BUDGETS.get(model_name, 1.0)

        if self._shutdown:
            return False

        with self._lock:
            available = self.get_available_memory_gb()

            if memory_gb <= available:
                self._pending_gb += memory_gb
                self._models[model_name] = ModelRegistration(
                    name=model_name,
                    memory_gb=memory_gb,
                    model_ref=weakref.ref(lambda: None),
                    unload_fn=None,
                    state=ModelState.LOADING,
                    priority=1,
                )
                logger.info(
                    f"[MemoryCoordinator] Acquired {memory_gb:.1f}GB for {model_name} (sync)"
                )
                return True

        # Try eviction
        self._evict_for_memory_sync(memory_gb - available)

        with self._lock:
            available = self.get_available_memory_gb()
            if memory_gb <= available:
                self._pending_gb += memory_gb
                self._models[model_name] = ModelRegistration(
                    name=model_name,
                    memory_gb=memory_gb,
                    model_ref=weakref.ref(lambda: None),
                    unload_fn=None,
                    state=ModelState.LOADING,
                    priority=1,
                )
                logger.info(
                    f"[MemoryCoordinator] Acquired {memory_gb:.1f}GB for {model_name} after eviction"
                )
                return True

        logger.warning(f"Could not acquire {memory_gb:.1f}GB for {model_name}")
        return False

    def release(self, model_name: str):
        """
        Release memory for a model (either after failure or unload).

        This is the simplified release for singleton patterns.
        """
        with self._lock:
            if model_name not in self._models:
                return

            reg = self._models[model_name]

            # Update allocated/pending based on state
            if reg.state == ModelState.LOADING:
                self._pending_gb = max(0, self._pending_gb - reg.memory_gb)
            elif reg.state == ModelState.READY:
                self._allocated_gb = max(0, self._allocated_gb - reg.memory_gb)

            reg.state = ModelState.UNLOADED
            del self._models[model_name]

            logger.info(
                f"[MemoryCoordinator] Released {reg.memory_gb:.1f}GB from {model_name}"
            )

    def mark_ready(
        self, model_name: str, model: Any = None, unload_fn: Callable | None = None
    ):
        """
        Mark a model as ready after successful loading.

        Moves memory from pending to allocated.
        """
        with self._lock:
            if model_name not in self._models:
                logger.warning(f"Cannot mark_ready: {model_name} not in models")
                return

            reg = self._models[model_name]

            # Move from pending to allocated
            if reg.state == ModelState.LOADING:
                self._pending_gb = max(0, self._pending_gb - reg.memory_gb)
                self._allocated_gb += reg.memory_gb

            reg.state = ModelState.READY
            if model is not None:
                reg.model_ref = weakref.ref(model)
            if unload_fn is not None:
                reg.unload_fn = unload_fn
            reg.load_time = time.time()
            reg.last_used = time.time()

            logger.info(
                f"[MemoryCoordinator] {model_name} ready ({reg.memory_gb:.1f}GB)"
            )

    def force_cleanup(self):
        """Force garbage collection and memory cleanup."""
        gc.collect()

        # Try to clean up MPS cache if available
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        logger.info("[MemoryCoordinator] Force cleanup completed")


# ============================================================================
# Global Instance
# ============================================================================

_coordinator: GlobalMemoryCoordinator | None = None
_coordinator_lock = threading.Lock()


def get_memory_coordinator() -> GlobalMemoryCoordinator:
    """Get global memory coordinator singleton."""
    global _coordinator
    if _coordinator is None:
        with _coordinator_lock:
            if _coordinator is None:
                _coordinator = GlobalMemoryCoordinator()
    return _coordinator


# ============================================================================
# Convenience Decorator
# ============================================================================


def managed_model(model_name: str, memory_gb: float, priority: int = 0):
    """
    Decorator for model loader functions to auto-register with coordinator.

    Usage:
        @managed_model("embedding_model", memory_gb=2.5, priority=1)
        def load_embedding_model():
            return SentenceTransformer("BAAI/bge-m3")
    """

    def decorator(loader_fn):
        def wrapper(*args, **kwargs):
            coordinator = get_memory_coordinator()

            # Check if already loaded
            if coordinator.is_loaded(model_name):
                return coordinator.get_model(model_name)

            with coordinator.acquire_memory_sync(model_name, memory_gb, priority):
                model = loader_fn(*args, **kwargs)

                # Create unload function
                def unload():
                    nonlocal model
                    del model
                    gc.collect()

                coordinator.register_model(model_name, model, unload)
                return model

        return wrapper

    return decorator


async def managed_model_async(
    model_name: str,
    memory_gb: float,
    loader_fn,
    priority: int = 0,
    unload_fn: Callable | None = None,
):
    """
    Async helper to load a model with memory coordination.

    Usage:
        model = await managed_model_async(
            "llm",
            memory_gb=4.5,
            loader_fn=lambda: load_llm_model(),
            priority=1
        )
    """
    coordinator = get_memory_coordinator()

    if coordinator.is_loaded(model_name):
        return coordinator.get_model(model_name)

    async with coordinator.acquire_memory(model_name, memory_gb, priority):
        loop = asyncio.get_running_loop()
        model = await loop.run_in_executor(None, loader_fn)

        if unload_fn is None:

            def default_unload():
                gc.collect()

            unload_fn = default_unload

        coordinator.register_model(model_name, model, unload_fn)
        return model


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "GlobalMemoryCoordinator",
    "MemoryBudgetConfig",
    "MemoryPressure",
    "ModelRegistration",
    "ModelState",
    "get_memory_coordinator",
    "managed_model",
    "managed_model_async",
]
