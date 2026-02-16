"""
Inference Runtime — Central Execution Coordinator
=====================================================

The runtime is the single entry point for all model inference.
It coordinates:
  - Backend selection (based on hardware layer)
  - Worker pool management
  - Dynamic batching
  - Streaming orchestration
  - Cancellation handling
  - Metrics collection

All inference goes through: Runtime → Worker Pool → Backend
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any

from backend.infra.runtime.backends import (
    InferenceBackend,
    InferenceRequest,
    InferenceResult,
)
from backend.infra.runtime.backends.unified_bridge import UnifiedBridgeBackend
from backend.infra.runtime.batcher import DynamicBatcher, get_batcher
from backend.infra.runtime.streamer import StreamConfig, TokenStreamer
from backend.infra.runtime.worker_pool import WorkerPool, get_worker_pool
from backend.infra.telemetry import get_logger, get_metrics, get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)
metrics = get_metrics()

class InferenceRuntime:
    """
    Central inference coordinator.

    Manages the lifecycle of inference requests through:
      Backend selection → Worker pool → Batcher → Backend execution

    Usage:
        runtime = get_runtime()
        await runtime.initialize()

        # Single request
        result = await runtime.execute(task_type="chat", payload={"prompt": "Hi"})

        # Batch request
        results = await runtime.batch_execute(task_type="embed", payloads=[...])

        # Streaming
        async for token in runtime.stream(task_type="chat", payload={...}):
            yield token
    """

    def __init__(self) -> None:
        self._backends: dict[str, InferenceBackend] = {}
        self._default_backend: InferenceBackend | None = None
        self._worker_pool: WorkerPool | None = None
        self._batcher: DynamicBatcher | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(
        self,
        *,
        backend: InferenceBackend | None = None,
        worker_pool: WorkerPool | None = None,
        batcher: DynamicBatcher | None = None,
    ) -> None:
        """
        Initialize the runtime.

        Auto-detects hardware and creates appropriate backend if none provided.
        """
        async with self._lock:
            if self._initialized:
                return

            # Detect device
            try:
                from backend.core.hal import get_device
                device = get_device()
            except ImportError:
                device = "cpu"

            # Set up backend
            self._default_backend = backend or UnifiedBridgeBackend(device=device)
            self._backends["default"] = self._default_backend

            # Set up worker pool
            self._worker_pool = worker_pool or get_worker_pool()
            await self._worker_pool.start()

            # Set up batcher
            self._batcher = batcher or get_batcher()
            self._batcher.set_backend(self._default_backend)
            await self._batcher.start()

            self._initialized = True
            logger.info(
                "runtime_initialized",
                device=device,
                backend=type(self._default_backend).__name__,
            )

    async def execute(
        self,
        *,
        task_type: str,
        payload: dict[str, Any],
        model: str | None = None,
        device: str | None = None,
        timeout_s: float = 30.0,
    ) -> Any:
        """
        Execute a single inference request.

        Routes through worker pool for concurrency control.
        Batch-eligible tasks go through the batcher.
        """
        self._ensure_initialized()

        request = InferenceRequest(
            task=self._normalize_task(task_type),
            inputs=payload,
            max_tokens=payload.get("max_tokens", 2048),
            temperature=payload.get("temperature", 0.7),
            top_p=payload.get("top_p", 0.9),
            timeout_s=timeout_s,
        )

        with tracer.span(
            f"runtime.execute.{task_type}",
            attributes={"task": task_type, "model": model or "default"},
        ) as span:
            backend = self._select_backend(model)

            # Batch-eligible tasks go through batcher
            if self._batcher and self._batcher.is_task_enabled(request.task):
                result = await self._worker_pool.submit(  # type: ignore[union-attr]
                    lambda: self._batcher.submit(request),  # type: ignore[union-attr]
                    timeout_s=timeout_s,
                )
            else:
                result = await self._worker_pool.submit(  # type: ignore[union-attr]
                    lambda: backend.infer(request),
                    timeout_s=timeout_s,
                )

            # Record metrics
            metrics.record_inference(
                model=result.model or task_type,
                task=task_type,
                device=result.device or device or "unknown",
                latency_s=result.latency_ms / 1000,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

            span.set_attribute("latency_ms", round(result.latency_ms, 2))
            span.set_attribute("device", result.device)

            return result.output

    async def batch_execute(
        self,
        *,
        task_type: str,
        payloads: list[dict[str, Any]],
        timeout_s: float = 120.0,
    ) -> list[Any]:
        """Execute a batch of inference requests."""
        self._ensure_initialized()

        with tracer.span(
            "runtime.batch_execute",
            attributes={"task": task_type, "batch_size": len(payloads)},
        ):
            backend = self._select_backend(None)
            requests = [
                InferenceRequest(
                    task=self._normalize_task(task_type),
                    inputs=p,
                    max_tokens=p.get("max_tokens", 2048),
                    timeout_s=timeout_s,
                )
                for p in payloads
            ]

            results = await self._worker_pool.submit(  # type: ignore[union-attr]
                lambda: backend.infer_batch(requests),
                timeout_s=timeout_s,
            )

            return [r.output for r in results]

    async def stream(
        self,
        *,
        task_type: str,
        payload: dict[str, Any],
        model: str | None = None,
        device: str | None = None,
        timeout_s: float = 120.0,
    ) -> AsyncGenerator[str, None]:
        """Stream inference results token by token."""
        self._ensure_initialized()

        request = InferenceRequest(
            task=self._normalize_task(task_type),
            inputs=payload,
            max_tokens=payload.get("max_tokens", 4096),
            temperature=payload.get("temperature", 0.7),
            stream=True,
            timeout_s=timeout_s,
        )

        backend = self._select_backend(model)

        async for token in backend.infer_stream(request):
            yield token

    def register_backend(self, name: str, backend: InferenceBackend) -> None:
        """Register a named backend."""
        self._backends[name] = backend
        logger.info("backend_registered", name=name, type=type(backend).__name__)

    def _select_backend(self, model: str | None) -> InferenceBackend:
        """Select the appropriate backend."""
        if model and model in self._backends:
            return self._backends[model]
        return self._default_backend  # type: ignore[return-value]

    def _normalize_task(self, task_type: str) -> str:
        """Normalize task type names."""
        mapping = {
            "chat": "generate",
            "reasoning": "generate",
            "code": "generate",
            "summarization": "generate",
            "embedding": "embed",
            "reranking": "rerank",
        }
        return mapping.get(task_type, task_type)

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "InferenceRuntime not initialized. Call await runtime.initialize() first."
            )

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._batcher:
            await self._batcher.stop()
        if self._worker_pool:
            await self._worker_pool.shutdown(drain=True)
        for backend in self._backends.values():
            await backend.shutdown()
        self._initialized = False
        logger.info("runtime_shutdown")

    def get_stats(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "backends": list(self._backends.keys()),
            "worker_pool": self._worker_pool.get_stats() if self._worker_pool else None,
            "batcher": self._batcher.get_stats() if self._batcher else None,
        }

# ── Singleton ──────────────────────────────────────────────────────

_runtime: InferenceRuntime | None = None

def get_runtime() -> InferenceRuntime:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _runtime
    if _runtime is not None:
        return _runtime
    _runtime = InferenceRuntime()
    return _runtime