"""
Dynamic Batcher — Automatic Request Batching
================================================

Accumulates inference requests and dispatches them as batches
for hardware-efficient execution.

Design:
  - Configurable batch window (time-based + count-based triggers)
  - Per-model batch size limits (from hardware layer)
  - Automatic grouping by task type and model
  - Individual result routing back to callers
  - Backpressure via bounded queue
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, cast

from backend.infra.runtime.backends import InferenceRequest, InferenceResult
from backend.infra.telemetry import get_logger, get_metrics

logger = get_logger(__name__)
metrics = get_metrics()

@dataclass
class BatchConfig:
    """Batching configuration."""

    max_batch_size: int = 32
    max_wait_ms: float = 50.0  # Max time to wait for batch to fill
    min_batch_size: int = 1
    enabled_tasks: set[str] = field(
        default_factory=lambda: {"embed", "rerank", "translate"}
    )

@dataclass
class PendingRequest:
    """Request waiting in the batcher."""

    request: InferenceRequest
    future: asyncio.Future[InferenceResult]
    enqueued_at: float = field(default_factory=time.monotonic)

    @property
    def wait_ms(self) -> float:
        return (time.monotonic() - self.enqueued_at) * 1000

class DynamicBatcher:
    """
    Accumulates requests and dispatches batches to backends.

    Usage:
        batcher = DynamicBatcher(config=BatchConfig(max_batch_size=64))
        batcher.set_backend(backend)
        await batcher.start()

        # Individual requests are transparently batched:
        result = await batcher.submit(request)
    """

    def __init__(self, config: BatchConfig | None = None) -> None:
        self._config = config or BatchConfig()
        self._pending: dict[str, list[PendingRequest]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._backend: Any = None
        self._running = False
        self._flush_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks

        # Stats
        self._total_submitted = 0
        self._total_batches = 0
        self._total_items = 0

    def _spawn_batch_task(self, key: str, batch: list[PendingRequest]) -> None:
        """Spawn a tracked batch execution task with proper error logging."""
        task = asyncio.create_task(self._execute_batch(key, batch))
        self._background_tasks.add(task)
        task.add_done_callback(self._on_batch_task_done)

    def _on_batch_task_done(self, task: asyncio.Task) -> None:
        """Callback to log unhandled exceptions from batch tasks."""
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error("batch_task_unhandled_error", error=str(exc), exc_type=type(exc).__name__)

    def is_task_enabled(self, task: str) -> bool:
        """Check whether a task type is eligible for batching."""
        return task in self._config.enabled_tasks

    def set_backend(self, backend: Any) -> None:
        """Set the inference backend for batch execution."""
        self._backend = backend

    async def start(self) -> None:
        """Start the background flush loop."""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._flush_task.add_done_callback(self._on_flush_task_done)
        logger.info("batcher_started", max_batch=self._config.max_batch_size)

    def _on_flush_task_done(self, task: asyncio.Task) -> None:
        """Log unhandled exceptions from the flush loop."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error("flush_loop_crashed", error=str(exc), exc_type=type(exc).__name__)

    async def stop(self) -> None:
        """Stop the batcher, flushing remaining requests."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task

        # Flush remaining
        await self._flush_all()

    async def submit(self, request: InferenceRequest) -> InferenceResult:
        """
        Submit a request for batched execution.

        Blocks until the batch containing this request is processed.
        Non-batchable tasks are executed immediately.
        """
        # Non-batchable tasks bypass batching
        if request.task not in self._config.enabled_tasks:
            if self._backend:
                return cast(InferenceResult, await self._backend.infer(request))
            raise RuntimeError("No backend configured")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResult] = loop.create_future()
        pending = PendingRequest(request=request, future=future)

        async with self._lock:
            key = f"{request.task}"
            self._pending[key].append(pending)
            self._total_submitted += 1

            # Immediate flush if batch is full
            if len(self._pending[key]) >= self._config.max_batch_size:
                batch = self._pending[key][: self._config.max_batch_size]
                self._pending[key] = self._pending[key][self._config.max_batch_size:]
                self._spawn_batch_task(key, batch)

        return await future

    async def _flush_loop(self) -> None:
        """Background loop that flushes aged batches."""
        while self._running:
            await asyncio.sleep(self._config.max_wait_ms / 1000)
            await self._flush_aged()

    async def _flush_aged(self) -> None:
        """Flush batches that have waited long enough."""
        async with self._lock:
            for key in list(self._pending.keys()):
                batch = self._pending[key]
                if not batch:
                    continue

                # Check if oldest request has waited long enough
                oldest_wait = batch[0].wait_ms
                if oldest_wait >= self._config.max_wait_ms or len(batch) >= self._config.min_batch_size:
                    to_flush = batch[: self._config.max_batch_size]
                    self._pending[key] = batch[self._config.max_batch_size:]
                    self._spawn_batch_task(key, to_flush)

    async def _flush_all(self) -> None:
        """Flush all pending batches."""
        async with self._lock:
            for key in list(self._pending.keys()):
                batch = self._pending[key]
                if batch:
                    self._pending[key] = []
                    await self._execute_batch(key, batch)

    async def _execute_batch(self, key: str, batch: list[PendingRequest]) -> None:
        """Execute a batch and route results to individual futures."""
        if not batch or not self._backend:
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(RuntimeError("No backend"))
            return

        self._total_batches += 1
        self._total_items += len(batch)

        try:
            requests = [p.request for p in batch]
            results = await self._backend.infer_batch(requests)

            for pending, result in zip(batch, results, strict=False):
                if not pending.future.done():
                    pending.future.set_result(result)

            metrics.record_inference(
                model=key,
                task=key,
                device=self._backend.capabilities().device,
                latency_s=max(r.latency_ms for r in results) / 1000,
                batch_size=len(batch),
            )

        except (RuntimeError, OSError) as exc:
            logger.error("batch_execution_failed", exc=exc, key=key, batch_size=len(batch))
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(exc)

    def get_stats(self) -> dict[str, Any]:
        return {
            "pending": {k: len(v) for k, v in self._pending.items()},
            "total_submitted": self._total_submitted,
            "total_batches": self._total_batches,
            "total_items": self._total_items,
            "avg_batch_size": round(
                self._total_items / max(self._total_batches, 1), 1
            ),
            "config": {
                "max_batch_size": self._config.max_batch_size,
                "max_wait_ms": self._config.max_wait_ms,
                "enabled_tasks": list(self._config.enabled_tasks),
            },
        }

# ── Singleton ──────────────────────────────────────────────────────

_batcher: DynamicBatcher | None = None

def get_batcher() -> DynamicBatcher:
    """Get or create the global batcher (lock-free; races are benign)."""
    global _batcher
    if _batcher is None:
        # Benign race: worst case two instances created, one is discarded.
        # Avoids threading.Lock which blocks the event loop in async context.
        _batcher = DynamicBatcher()
    return _batcher
