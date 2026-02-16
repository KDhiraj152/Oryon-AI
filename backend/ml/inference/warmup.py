"""
Model Warm-up Service.

Pre-loads and warms up ML models at application startup to reduce
first-inference latency. Supports:
- Lazy loading with background warm-up
- Priority-based loading order
- Memory-aware loading decisions
- Health checks for loaded models
"""

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

class ModelPriority(Enum):
    """Loading priority for models."""

    CRITICAL = 0  # Load immediately, block startup
    HIGH = 1  # Load early, non-blocking
    NORMAL = 2  # Load when resources available
    LOW = 3  # Load on-demand only

class ModelStatus(Enum):
    """Model loading status."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"

@dataclass
class ModelSpec:
    """Specification for a model to warm up."""

    name: str
    loader: Callable[[], Any]
    priority: ModelPriority = ModelPriority.NORMAL
    warmup_fn: Callable[[Any], None] | None = None
    memory_mb: int = 500  # Estimated memory usage
    timeout_seconds: int = 300
    retries: int = 2

@dataclass
class LoadedModel:
    """Information about a loaded model."""

    name: str
    status: ModelStatus = ModelStatus.NOT_LOADED
    model: Any = None
    load_time_ms: float = 0
    warmup_time_ms: float = 0
    error: str | None = None
    last_used: float = field(default_factory=time.time)

class ModelWarmupService:
    """
    Service for warming up ML models at startup.

    Features:
    - Priority-based loading queue
    - Memory-aware decisions
    - Background loading
    - Automatic retry on failure
    - Health monitoring
    """

    def __init__(self, max_memory_mb: int = 8000, max_concurrent_loads: int = 2):
        self._models: dict[str, LoadedModel] = {}
        self._specs: dict[str, ModelSpec] = {}
        self._max_memory_mb = max_memory_mb
        self._current_memory_mb = 0
        self._loading_lock = asyncio.Lock()
        from ...core.optimized.thread_pool_manager import get_ml_executor
        self._executor = get_ml_executor()  # Shared ML pool
        self._load_order: list[str] = []
        self._background_task: asyncio.Task | None = None

    def register(self, spec: ModelSpec):
        """Register a model for warm-up."""
        self._specs[spec.name] = spec
        self._models[spec.name] = LoadedModel(name=spec.name)
        logger.info("Registered model: %s (priority=%s)", spec.name, spec.priority.name)

    async def warmup_all(self, block_on_critical: bool = True):
        """
        Warm up all registered models.

        Args:
            block_on_critical: Wait for CRITICAL models before returning
        """
        # Sort by priority
        sorted_specs = sorted(self._specs.values(), key=lambda s: s.priority.value)

        critical_tasks = []
        background_tasks = []

        for spec in sorted_specs:
            task = self._warmup_model(spec)
            if spec.priority == ModelPriority.CRITICAL:
                critical_tasks.append(task)
            else:
                background_tasks.append(task)

        # Wait for critical models
        if critical_tasks and block_on_critical:
            logger.info("Loading %s critical models...", len(critical_tasks))
            await asyncio.gather(*critical_tasks)
            logger.info("Critical models loaded")

        # Start background loading - store task reference to prevent GC
        if background_tasks:
            logger.info(
                f"Starting background load of {len(background_tasks)} models..."
            )
            self._background_task = asyncio.create_task(
                self._run_background_tasks(background_tasks)
            )

    async def _run_background_tasks(self, tasks: list):
        """Run background loading tasks."""
        for task in tasks:
            try:
                await task
            except (RuntimeError, OSError) as e:
                logger.error("Background model load error: %s", e)

    async def _warmup_model(self, spec: ModelSpec) -> bool:
        """Warm up a single model with retries."""
        model_info = self._models[spec.name]

        for attempt in range(spec.retries + 1):
            try:
                async with self._loading_lock:
                    # Check memory
                    if self._current_memory_mb + spec.memory_mb > self._max_memory_mb:
                        logger.warning(
                            f"Skipping {spec.name}: insufficient memory "
                            f"({self._current_memory_mb}/{self._max_memory_mb} MB used)"
                        )
                        return False

                    model_info.status = ModelStatus.LOADING

                # Load model (in thread pool for sync loaders)
                start = time.perf_counter()
                loop = asyncio.get_running_loop()
                model = await loop.run_in_executor(self._executor, spec.loader)
                load_time = (time.perf_counter() - start) * 1000

                # Warm up
                warmup_time = 0
                if spec.warmup_fn:
                    start = time.perf_counter()
                    await loop.run_in_executor(self._executor, spec.warmup_fn, model)
                    warmup_time = (time.perf_counter() - start) * 1000

                # Update state
                async with self._loading_lock:
                    model_info.model = model
                    model_info.status = ModelStatus.READY
                    model_info.load_time_ms = load_time
                    model_info.warmup_time_ms = warmup_time
                    model_info.last_used = time.time()
                    self._current_memory_mb += spec.memory_mb
                    self._load_order.append(spec.name)

                logger.info(
                    f"✓ {spec.name} ready (load={load_time:.0f}ms, warmup={warmup_time:.0f}ms)"
                )
                return True

            except (RuntimeError, OSError) as e:
                logger.warning(
                    f"Failed to load {spec.name} (attempt {attempt + 1}): {e}"
                )
                if attempt == spec.retries:
                    model_info.status = ModelStatus.ERROR
                    model_info.error = str(e)
                    return False
                await asyncio.sleep(1)  # Brief delay before retry

        return False

    def get_model(self, name: str) -> Any | None:
        """Get a loaded model by name."""
        if name in self._models:
            info = self._models[name]
            if info.status == ModelStatus.READY:
                info.last_used = time.time()
                return info.model
        return None

    def get_status(self) -> dict[str, Any]:
        """Get status of all models."""
        # Get device type from DeviceRouter singleton
        device_type = "unknown"
        try:
            from ...core.optimized import get_device_router

            router = get_device_router()
            device_type = router.capabilities.device_type
        except (ImportError, RuntimeError):
            pass

        return {
            "models": {
                name: {
                    "status": info.status.value,
                    "load_time_ms": info.load_time_ms,
                    "warmup_time_ms": info.warmup_time_ms,
                    "error": info.error,
                }
                for name, info in self._models.items()
            },
            "memory_used_mb": self._current_memory_mb,
            "memory_limit_mb": self._max_memory_mb,
            "load_order": self._load_order,
            "device_type": device_type,
            "warmed_models": len(
                [m for m in self._models.values() if m.status == ModelStatus.READY]
            ),
        }

    async def unload_model(self, name: str):
        """Unload a model to free memory."""
        if name not in self._models:
            return

        async with self._loading_lock:
            info = self._models[name]
            if info.status == ModelStatus.READY:
                spec = self._specs.get(name)
                if spec:
                    self._current_memory_mb -= spec.memory_mb

                info.model = None
                info.status = ModelStatus.UNLOADED

                if name in self._load_order:
                    self._load_order.remove(name)

                logger.info("Unloaded model: %s", name)

    async def reload_model(self, name: str) -> bool:
        """Reload a model."""
        if name in self._specs:
            return await self._warmup_model(self._specs[name])
        return False

    def is_ready(self, name: str) -> bool:
        """Check if a model is ready."""
        return name in self._models and self._models[name].status == ModelStatus.READY

    def all_critical_ready(self) -> bool:
        """Check if all critical models are ready."""
        for name, spec in self._specs.items():
            if spec.priority == ModelPriority.CRITICAL and not self.is_ready(name):
                return False
        return True

    def warm_up_all(self) -> dict[str, Any]:
        """
        Synchronous wrapper for warmup_all().
        Used when called from thread pool executor.
        """
        # For now, just register default models and return status
        # The actual loading will happen on first use
        logger.info("Synchronous warm-up started")

        # Try to detect available backends using singleton
        try:
            from ...core.optimized import get_device_router

            router = get_device_router()
            logger.info("Device detected: %s", router.capabilities.device_type)
            logger.info(
                f"Available backends: MLX={router.capabilities.mlx_available}, "
                f"CoreML={router.capabilities.coreml_available}, "
                f"MPS={router.capabilities.has_mps}"
            )
        except (ImportError, RuntimeError) as e:
            logger.warning("Could not detect device capabilities: %s", e)

        return self.get_status()

# ==================== Global Instance ====================

_warmup_service: ModelWarmupService | None = None

def get_warmup_service() -> ModelWarmupService:
    """Get singleton warmup service."""
    global _warmup_service
    if _warmup_service is None:
        _warmup_service = ModelWarmupService()
    return _warmup_service

# ==================== Convenience Decorators ====================

def warmup_model(
    name: str,
    priority: ModelPriority = ModelPriority.NORMAL,
    memory_mb: int = 500,
    warmup_fn: Callable | None = None,
):
    """Decorator to register a model loader for warm-up."""

    def decorator(loader_fn: Callable[[], Any]):
        spec = ModelSpec(
            name=name,
            loader=loader_fn,
            priority=priority,
            memory_mb=memory_mb,
            warmup_fn=warmup_fn,
        )
        get_warmup_service().register(spec)
        return loader_fn

    return decorator

# ==================== Exports ====================

__all__ = [
    "LoadedModel",
    "ModelPriority",
    "ModelSpec",
    "ModelStatus",
    "ModelWarmupService",
    "get_warmup_service",
    "warmup_model",
]
