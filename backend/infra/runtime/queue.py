"""
Priority Queue Manager — Bounded Priority Queue with Load Shedding
=====================================================================

Manages request queuing with:
  - Priority-based ordering (CRITICAL > HIGH > NORMAL > LOW > BACKGROUND)
  - Bounded capacity with configurable limits per priority
  - Load shedding for low-priority requests under pressure
  - Queue depth monitoring and alerting
  - Fair scheduling across priority levels

Design:
  - Uses asyncio.PriorityQueue for non-blocking operations
  - Separate capacity tracking per priority level
  - Background drain loop for worker dispatch
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from backend.infra.runtime.pipeline import PipelineContext, RequestPriority
from backend.infra.telemetry import get_logger, get_metrics

logger = get_logger(__name__)
metrics = get_metrics()

T = TypeVar("T")

@dataclass(order=True)
class QueueItem:
    """Priority queue entry. Lower priority value = higher priority."""

    priority: int
    timestamp: float = field(compare=False)
    context: PipelineContext = field(compare=False)

    @property
    def wait_time_ms(self) -> float:
        return (time.monotonic() - self.timestamp) * 1000

@dataclass
class QueueConfig:
    """Queue capacity configuration."""

    max_total: int = 500
    max_per_priority: dict[RequestPriority, int] = field(default_factory=lambda: {
        RequestPriority.CRITICAL: 50,
        RequestPriority.HIGH: 100,
        RequestPriority.NORMAL: 200,
        RequestPriority.LOW: 100,
        RequestPriority.BACKGROUND: 50,
    })
    shed_threshold: float = 0.8  # Start shedding at 80% capacity
    stale_timeout_s: float = 60.0  # Drop stale requests

class PriorityQueueManager:
    """
    Bounded priority queue with load shedding and fair scheduling.

    Provides async enqueue/dequeue with backpressure signaling.
    Monitors queue depth and sheds low-priority requests under load.
    """

    def __init__(self, config: QueueConfig | None = None) -> None:
        self._config = config or QueueConfig()
        self._queue: asyncio.PriorityQueue[QueueItem] = asyncio.PriorityQueue(
            maxsize=self._config.max_total
        )
        self._counts: dict[RequestPriority, int] = dict.fromkeys(RequestPriority, 0)
        self._lock = asyncio.Lock()
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_shed = 0
        self._running = False
        self._drain_task: asyncio.Task | None = None

    async def enqueue(self, ctx: PipelineContext) -> bool:
        """
        Enqueue a request. Returns False if request was shed.

        Load shedding rules:
          1. CRITICAL requests are never shed
          2. HIGH requests shed only at 95% capacity
          3. NORMAL requests shed at shed_threshold
          4. LOW/BACKGROUND shed at 70% capacity
        """
        async with self._lock:
            total = sum(self._counts.values())
            capacity_ratio = total / self._config.max_total if self._config.max_total > 0 else 0

            # Check priority-specific capacity
            priority_limit = self._config.max_per_priority.get(
                ctx.priority, self._config.max_total
            )
            if self._counts[ctx.priority] >= priority_limit:
                logger.warning(
                    "queue_priority_full",
                    priority=ctx.priority.name,
                    current=self._counts[ctx.priority],
                    limit=priority_limit,
                    request_id=ctx.request_id,
                )
                self._total_shed += 1
                metrics.record_queue_depth(ctx.priority.name, self._counts[ctx.priority])
                return False

            # Load shedding by priority
            if ctx.priority == RequestPriority.CRITICAL:
                pass  # Never shed
            elif (ctx.priority == RequestPriority.HIGH and capacity_ratio > 0.95) or (ctx.priority == RequestPriority.NORMAL and capacity_ratio > self._config.shed_threshold) or (ctx.priority in (RequestPriority.LOW, RequestPriority.BACKGROUND) and capacity_ratio > 0.7):
                self._total_shed += 1
                return False

            # Enqueue
            item = QueueItem(
                priority=ctx.priority.value,
                timestamp=time.monotonic(),
                context=ctx,
            )

            try:
                self._queue.put_nowait(item)
                self._counts[ctx.priority] += 1
                self._total_enqueued += 1
                metrics.record_queue_depth(ctx.priority.name, self._counts[ctx.priority])
                return True
            except asyncio.QueueFull:
                self._total_shed += 1
                logger.warning(
                    "queue_full",
                    total=total,
                    request_id=ctx.request_id,
                )
                return False

    async def dequeue(self, timeout: float | None = None) -> PipelineContext | None:
        """Dequeue the highest-priority request."""
        try:
            if timeout is not None:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                item = await self._queue.get()

            async with self._lock:
                self._counts[RequestPriority(item.priority)] -= 1
                self._total_dequeued += 1

            # Check staleness
            if item.wait_time_ms > self._config.stale_timeout_s * 1000:
                logger.warning(
                    "stale_request_dropped",
                    request_id=item.context.request_id,
                    wait_ms=round(item.wait_time_ms, 1),
                )
                return None

            return item.context

        except (TimeoutError, asyncio.CancelledError):
            return None

    async def start_drain(
        self,
        handler: Callable[[PipelineContext], Awaitable[None]],
        *,
        concurrency: int = 4,
    ) -> None:
        """Start background drain loop dispatching to handler."""
        self._running = True
        sem = asyncio.Semaphore(concurrency)

        async def process_one() -> None:
            while self._running:
                ctx = await self.dequeue(timeout=1.0)
                if ctx is None:
                    continue
                async with sem:
                    try:
                        await handler(ctx)
                    except (RuntimeError, OSError) as exc:
                        logger.error(
                            "drain_handler_error",
                            exc=exc,
                            request_id=ctx.request_id,
                        )

        self._drain_task = asyncio.create_task(process_one())

    async def stop(self) -> None:
        """Stop the drain loop."""
        self._running = False
        if self._drain_task and not self._drain_task.done():
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    def get_stats(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "counts_by_priority": {p.name: c for p, c in self._counts.items()},
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_shed": self._total_shed,
            "capacity": self._config.max_total,
            "utilization": round(self.depth / max(self._config.max_total, 1), 3),
        }

# ── Singleton ──────────────────────────────────────────────────────

_manager: PriorityQueueManager | None = None

def get_queue_manager() -> PriorityQueueManager:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _manager
    if _manager is not None:
        return _manager
    _manager = PriorityQueueManager()
    return _manager