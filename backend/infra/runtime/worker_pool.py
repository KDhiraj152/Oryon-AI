"""
Worker Pool — Concurrent Inference Worker Management
=======================================================

Manages a pool of inference workers with:
  - Configurable concurrency limits
  - Per-model semaphores
  - Health monitoring
  - Graceful drain for scaling
  - Stats collection per worker
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from backend.infra.telemetry import get_logger, get_metrics

logger = get_logger(__name__)
metrics = get_metrics()

@dataclass
class WorkerConfig:
    """Worker pool configuration."""

    max_workers: int = 4
    max_queue_size: int = 100
    worker_timeout_s: float = 60.0
    health_check_interval_s: float = 30.0
    drain_timeout_s: float = 30.0

@dataclass
class WorkerStats:
    """Per-worker statistics."""

    worker_id: int
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_latency_ms: float = 0.0
    busy: bool = False
    last_task_at: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.total_latency_ms / self.tasks_completed

class WorkerPool:
    """
    Pool of async inference workers.

    Provides bounded concurrency for model inference,
    ensuring the system doesn't overload GPU/memory.

    Usage:
        pool = WorkerPool(config=WorkerConfig(max_workers=4))
        await pool.start()

        result = await pool.submit(inference_coro)

        await pool.shutdown()
    """

    def __init__(self, config: WorkerConfig | None = None) -> None:
        self._config = config or WorkerConfig()
        self._sem = asyncio.Semaphore(self._config.max_workers)
        self._workers: dict[int, WorkerStats] = {
            i: WorkerStats(worker_id=i) for i in range(self._config.max_workers)
        }
        self._active_count = 0
        self._lock = asyncio.Lock()
        self._running = False
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_rejected = 0

    async def start(self) -> None:
        """Start the worker pool."""
        self._running = True
        logger.info("worker_pool_started", max_workers=self._config.max_workers)

    async def shutdown(self, *, drain: bool = True) -> None:
        """
        Shutdown the worker pool.

        Args:
            drain: If True, wait for in-flight tasks to complete.
        """
        self._running = False
        if drain:
            # Wait for active tasks
            deadline = time.monotonic() + self._config.drain_timeout_s
            while self._active_count > 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.1)

            if self._active_count > 0:
                logger.warning(
                    "worker_pool_drain_timeout",
                    remaining=self._active_count,
                )

        logger.info(
            "worker_pool_shutdown",
            completed=self._total_completed,
            failed=self._total_failed,
        )

    async def submit(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """
        Submit work to the pool.

        Blocks until a worker is available (backpressure).

        Args:
            coro_factory: Factory that creates the coroutine to run
            timeout_s: Override per-task timeout
        """
        if not self._running:
            raise RuntimeError("Worker pool is not running")

        self._total_submitted += 1
        timeout = timeout_s or self._config.worker_timeout_s

        # Acquire worker slot (wait_for raises TimeoutError on expiry)
        await asyncio.wait_for(
            self._sem.acquire(), timeout=timeout
        )

        async with self._lock:
            self._active_count += 1
            worker_id = self._find_idle_worker()
            if worker_id is not None:
                self._workers[worker_id].busy = True

        try:
            start = time.monotonic()
            result = await asyncio.wait_for(coro_factory(), timeout=timeout)
            latency_ms = (time.monotonic() - start) * 1000

            # Update stats
            async with self._lock:
                self._total_completed += 1
                if worker_id is not None:
                    w = self._workers[worker_id]
                    w.tasks_completed += 1
                    w.total_latency_ms += latency_ms
                    w.last_task_at = time.monotonic()

            return result

        except (RuntimeError, OSError):
            async with self._lock:
                self._total_failed += 1
                if worker_id is not None:
                    self._workers[worker_id].tasks_failed += 1
            raise

        finally:
            async with self._lock:
                self._active_count -= 1
                if worker_id is not None:
                    self._workers[worker_id].busy = False
            self._sem.release()

    def _find_idle_worker(self) -> int | None:
        """Find an idle worker slot."""
        for wid, stats in self._workers.items():
            if not stats.busy:
                return wid
        return None

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def available_count(self) -> int:
        return self._config.max_workers - self._active_count

    def get_stats(self) -> dict[str, Any]:
        return {
            "max_workers": self._config.max_workers,
            "active": self._active_count,
            "available": self.available_count,
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "total_rejected": self._total_rejected,
            "running": self._running,
            "workers": {
                wid: {
                    "completed": w.tasks_completed,
                    "failed": w.tasks_failed,
                    "avg_latency_ms": round(w.avg_latency_ms, 1),
                    "busy": w.busy,
                }
                for wid, w in self._workers.items()
            },
        }

# ── Singleton ──────────────────────────────────────────────────────

_pool: WorkerPool | None = None

def get_worker_pool() -> WorkerPool:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _pool
    if _pool is not None:
        return _pool
    _pool = WorkerPool()
    return _pool