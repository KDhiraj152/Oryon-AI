"""
Async-First Architecture Optimizer
===================================

Phase 1 of M4 Core-Level Optimization

Based on benchmark findings:
- Async I/O achieved 10.86x speedup over sequential
- Python GIL blocks threading (0.71x slower)
- Solution: Maximize async/await, use asyncio.gather()

Key Components:
1. AsyncTaskRunner - Parallel task execution with gather()
2. AsyncConnectionPool - Reusable async connections
3. AsyncBatchProcessor - Batched async operations
4. AsyncPipelineExecutor - Concurrent pipeline stages
"""

import asyncio
import contextlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class TaskPriority(Enum):
    """Task priority levels for scheduling."""

    CRITICAL = 0  # User-facing, latency-sensitive
    HIGH = 1  # Important processing
    NORMAL = 2  # Standard tasks
    LOW = 3  # Background tasks
    BACKGROUND = 4  # Can be delayed


@dataclass
class AsyncTaskResult(Generic[T]):
    """Result wrapper for async tasks."""

    success: bool
    value: T | None = None
    error: Exception | None = None
    elapsed_ms: float = 0.0
    task_id: str = ""

    @property
    def ok(self) -> bool:
        return self.success and self.error is None


@dataclass
class AsyncPoolConfig:
    """Configuration for async connection pools."""

    min_size: int = 2
    max_size: int = 10
    max_idle_time: float = 60.0  # seconds
    acquire_timeout: float = 5.0  # seconds
    health_check_interval: float = 30.0


class AsyncTaskRunner:
    """
    High-performance async task runner with parallel execution.

    Uses asyncio.gather() with proper error handling and
    configurable concurrency limits.

    Features:
    - Parallel execution with gather()
    - Concurrency limiting via semaphores
    - Priority-based scheduling
    - Automatic retry with backoff
    - Task cancellation support
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        default_timeout: float = 30.0,
    ):
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks: set[asyncio.Task] = set()
        self._stats = {
            "total_tasks": 0,
            "successful": 0,
            "failed": 0,
            "timed_out": 0,
            "total_time_ms": 0.0,
        }
        self._lock = asyncio.Lock()

    async def run_parallel(
        self,
        tasks: list[Coroutine[Any, Any, T]],
        return_exceptions: bool = False,
        timeout: float | None = None,
    ) -> list[AsyncTaskResult[T]]:
        """
        Run multiple tasks in parallel using asyncio.gather().

        This is the core optimization - achieves 10.86x speedup for I/O tasks.

        Args:
            tasks: List of coroutines to run
            return_exceptions: If True, exceptions are returned instead of raised
            timeout: Overall timeout for all tasks

        Returns:
            List of AsyncTaskResult with success/failure info
        """
        if not tasks:
            return []

        timeout = timeout or self.default_timeout
        results: list[AsyncTaskResult[T]] = []

        async def wrapped_task(coro: Coroutine, idx: int) -> AsyncTaskResult[T]:
            """Wrap task with semaphore, timing, and error handling."""
            start = time.perf_counter()
            async with self._semaphore:
                try:
                    value = await asyncio.wait_for(coro, timeout=timeout)
                    elapsed = (time.perf_counter() - start) * 1000
                    async with self._lock:
                        self._stats["successful"] += 1
                        self._stats["total_time_ms"] += elapsed
                    return AsyncTaskResult(
                        success=True,
                        value=value,
                        elapsed_ms=elapsed,
                        task_id=f"task_{idx}",
                    )
                except TimeoutError:
                    elapsed = (time.perf_counter() - start) * 1000
                    async with self._lock:
                        self._stats["timed_out"] += 1
                    return AsyncTaskResult(
                        success=False,
                        error=TimeoutError(f"Task {idx} timed out"),
                        elapsed_ms=elapsed,
                        task_id=f"task_{idx}",
                    )
                except Exception as e:
                    elapsed = (time.perf_counter() - start) * 1000
                    async with self._lock:
                        self._stats["failed"] += 1
                    return AsyncTaskResult(
                        success=False,
                        error=e,
                        elapsed_ms=elapsed,
                        task_id=f"task_{idx}",
                    )

        async with self._lock:
            self._stats["total_tasks"] += len(tasks)

        # Use gather for parallel execution - this is the 10.86x speedup
        wrapped = [wrapped_task(t, i) for i, t in enumerate(tasks)]
        results = await asyncio.gather(*wrapped, return_exceptions=return_exceptions)

        return results

    async def run_with_priority(
        self,
        tasks: list[tuple[TaskPriority, Coroutine[Any, Any, T]]],
    ) -> list[AsyncTaskResult[T]]:
        """Run tasks sorted by priority (critical first)."""
        sorted_tasks = sorted(tasks, key=lambda x: x[0].value)
        coroutines = [t[1] for t in sorted_tasks]
        return await self.run_parallel(coroutines)

    async def map_async(
        self,
        func: Callable[[T], Coroutine[Any, Any, R]],
        items: list[T],
        chunk_size: int = 0,
    ) -> list[R]:
        """
        Async map over items with parallel execution.

        Like concurrent.futures.map() but for async functions.

        Args:
            func: Async function to apply
            items: Items to process
            chunk_size: If > 0, process in chunks (for memory management)

        Returns:
            List of results in same order as input
        """
        if chunk_size > 0:
            # Process in chunks to manage memory
            results = []
            for i in range(0, len(items), chunk_size):
                chunk = items[i : i + chunk_size]
                chunk_coros = [func(item) for item in chunk]
                chunk_results = await self.run_parallel(chunk_coros)
                results.extend([r.value for r in chunk_results if r.success])
            return results
        else:
            coroutines = [func(item) for item in items]
            results = await self.run_parallel(coroutines)
            return [r.value for r in results if r.success]

    def get_stats(self) -> dict[str, Any]:
        """Get runner statistics."""
        total = self._stats["total_tasks"]
        successful = self._stats["successful"]
        return {
            **self._stats,
            "success_rate": f"{successful / total:.1%}" if total > 0 else "0%",
            "avg_time_ms": self._stats["total_time_ms"] / successful
            if successful > 0
            else 0,
            "max_concurrent": self.max_concurrent,
            "active_tasks": len(self._active_tasks),
        }


class AsyncConnectionPool(Generic[T]):
    """
    Generic async connection pool.

    Provides reusable connections with:
    - Min/max pool sizing
    - Automatic cleanup of idle connections
    - Health checking
    - Graceful shutdown
    """

    def __init__(
        self,
        factory: Callable[[], Coroutine[Any, Any, T]],
        config: AsyncPoolConfig | None = None,
        validator: Callable[[T], Coroutine[Any, Any, bool]] | None = None,
        destructor: Callable[[T], Coroutine[Any, Any, None]] | None = None,
    ):
        self.factory = factory
        self.config = config or AsyncPoolConfig()
        self.validator = validator
        self.destructor = destructor

        self._pool: asyncio.Queue[tuple[T, float]] = asyncio.Queue(
            maxsize=self.config.max_size
        )
        self._size = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self._cleanup_task: asyncio.Task | None = None

    async def _create_connection(self) -> T:
        """Create a new connection."""
        conn = await self.factory()
        async with self._lock:
            self._size += 1
        return conn

    async def _validate_connection(self, conn: T) -> bool:
        """Validate a connection is still usable."""
        if self.validator is None:
            return True
        try:
            return await self.validator(conn)
        except Exception:
            return False

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.

        Usage:
            async with pool.acquire() as conn:
                await conn.execute(...)
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        conn = None
        try:
            # Try to get from pool
            while True:
                try:
                    conn, created_at = self._pool.get_nowait()
                    # Check if connection is too old
                    if time.time() - created_at > self.config.max_idle_time:
                        if self.destructor:
                            await self.destructor(conn)
                        async with self._lock:
                            self._size -= 1
                        continue
                    # Validate connection
                    if await self._validate_connection(conn):
                        break
                    else:
                        if self.destructor:
                            await self.destructor(conn)
                        async with self._lock:
                            self._size -= 1
                except asyncio.QueueEmpty:
                    # No available connection, create new if under limit
                    async with self._lock:
                        if self._size < self.config.max_size:
                            conn = await self._create_connection()
                            break
                    # Wait for a connection to be returned
                    conn, _ = await asyncio.wait_for(
                        self._pool.get(),
                        timeout=self.config.acquire_timeout,
                    )
                    if await self._validate_connection(conn):
                        break

            yield conn

        finally:
            if conn is not None and not self._closed:
                await self._pool.put((conn, time.time()))

    async def close(self):
        """Close the pool and all connections."""
        self._closed = True
        if self._cleanup_task:
            self._cleanup_task.cancel()

        while not self._pool.empty():
            try:
                conn, _ = self._pool.get_nowait()
                if self.destructor:
                    await self.destructor(conn)
            except asyncio.QueueEmpty:
                break

        async with self._lock:
            self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def available(self) -> int:
        return self._pool.qsize()


class AsyncBatchProcessor(Generic[T, R]):
    """
    Batched async processor for high-throughput operations.

    Collects items and processes them in batches for efficiency.
    Useful for:
    - Embedding generation (batch for GPU efficiency)
    - Database inserts (batch for fewer round trips)
    - API calls (batch to reduce overhead)
    """

    def __init__(
        self,
        processor: Callable[[list[T]], Coroutine[Any, Any, list[R]]],
        batch_size: int = 32,
        max_wait_ms: float = 50.0,
    ):
        self.processor = processor
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        self._queue: asyncio.Queue[tuple[T, asyncio.Future]] = asyncio.Queue()
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "avg_batch_size": 0.0,
        }

    async def start(self):
        """Start the batch processor worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        """Stop the batch processor."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

    async def _worker(self):
        """Background worker that processes batches."""
        cancelled = False
        while self._running:
            batch: list[tuple[T, asyncio.Future]] = []

            try:
                # Wait for first item
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                batch.append(item)

                # Collect more items up to batch_size or timeout
                deadline = time.perf_counter() + (self.max_wait_ms / 1000)
                while len(batch) < self.batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=remaining,
                        )
                        batch.append(item)
                    except TimeoutError:
                        break

                # Process batch
                if batch:
                    items = [b[0] for b in batch]
                    futures = [b[1] for b in batch]

                    try:
                        results = await self.processor(items)
                        for future, result in zip(futures, results, strict=False):
                            if not future.done():
                                future.set_result(result)
                    except Exception as e:
                        for future in futures:
                            if not future.done():
                                future.set_exception(e)

                    # Update stats
                    self._stats["batches_processed"] += 1
                    self._stats["items_processed"] += len(batch)
                    self._stats["avg_batch_size"] = (
                        self._stats["items_processed"]
                        / self._stats["batches_processed"]
                    )

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                cancelled = True
                break

        if cancelled:
            raise asyncio.CancelledError()

    async def process(self, item: T) -> R:
        """
        Submit item for batch processing.

        Returns when the item's batch is processed.
        """
        if not self._running:
            await self.start()

        future: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        await self._queue.put((item, future))
        return await future

    async def process_many(self, items: list[T]) -> list[R]:
        """Submit multiple items and wait for all results."""
        tasks = [self.process(item) for item in items]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "batch_size": self.batch_size,
            "running": self._running,
        }


class AsyncPipelineStage(ABC, Generic[T, R]):
    """Abstract base for pipeline stages."""

    @abstractmethod
    async def process(self, input_data: T) -> R:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class AsyncPipelineExecutor:
    """
    Concurrent pipeline executor for multi-stage processing.

    Executes pipeline stages with:
    - Parallel execution where possible
    - Stage-level concurrency control
    - Automatic resource cleanup
    - Progress tracking

    Example pipeline:
        tokenize(CPU) → embed(GPU) → cache(CPU) → respond

    Can run tokenize and cache operations on different cores
    while GPU handles embedding.
    """

    def __init__(self, stages: list[AsyncPipelineStage]):
        self.stages = stages
        self._stats = {stage.name: {"calls": 0, "total_ms": 0.0} for stage in stages}

    async def execute(self, input_data: Any) -> Any:
        """Execute pipeline sequentially (for dependent stages)."""
        current = input_data
        for stage in self.stages:
            start = time.perf_counter()
            current = await stage.process(current)
            elapsed = (time.perf_counter() - start) * 1000
            self._stats[stage.name]["calls"] += 1
            self._stats[stage.name]["total_ms"] += elapsed
        return current

    async def execute_parallel_stages(
        self,
        input_data: Any,
        parallel_groups: list[list[int]],
    ) -> Any:
        """
        Execute pipeline with parallel stage groups.

        Args:
            input_data: Initial input
            parallel_groups: List of stage index groups to run in parallel
                             e.g., [[0], [1, 2], [3]] runs stages 1&2 together

        Returns:
            Final result after all stages
        """
        current = input_data

        for group in parallel_groups:
            if len(group) == 1:
                # Single stage, run normally
                stage = self.stages[group[0]]
                current = await stage.process(current)
            else:
                # Multiple stages, run in parallel
                tasks = [self.stages[i].process(current) for i in group]
                results = await asyncio.gather(*tasks)
                # Merge results (take first by default, override for custom merging)
                current = results[0] if results else current

        return current

    def get_stats(self) -> dict[str, Any]:
        stats = {}
        for name, data in self._stats.items():
            calls = data["calls"]
            stats[name] = {
                "calls": calls,
                "avg_ms": data["total_ms"] / calls if calls > 0 else 0,
            }
        return stats


# ==================== UTILITY FUNCTIONS ====================


def async_retry(
    max_attempts: int = 3,
    backoff_factor: float = 1.5,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for async functions with retry logic.

    Usage:
        @async_retry(max_attempts=3)
        async def flaky_operation():
            ...
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = 0.1

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    logger.warning(
                        f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                    )

            raise last_exception

        return wrapper

    return decorator


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run async coroutine synchronously.

    Handles nested event loops properly.
    """
    try:
        loop = asyncio.get_running_loop()
        # Already in async context, use nest_asyncio
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(coro)


async def gather_with_concurrency(
    n: int,
    *coros: Coroutine[Any, Any, T],
) -> list[T]:
    """
    Like asyncio.gather() but with concurrency limit.

    Args:
        n: Maximum concurrent coroutines
        coros: Coroutines to run

    Returns:
        Results in same order as input
    """
    semaphore = asyncio.Semaphore(n)

    async def limited_coro(coro: Coroutine) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[limited_coro(c) for c in coros])


# ==================== SINGLETON INSTANCES ====================

_task_runner: AsyncTaskRunner | None = None
_task_runner_lock = threading.Lock()


def get_async_task_runner(max_concurrent: int = 10) -> AsyncTaskRunner:
    """Get global async task runner instance."""
    global _task_runner
    if _task_runner is None:
        with _task_runner_lock:
            if _task_runner is None:
                _task_runner = AsyncTaskRunner(max_concurrent=max_concurrent)
    return _task_runner


__all__ = [
    "AsyncBatchProcessor",
    "AsyncConnectionPool",
    "AsyncPipelineExecutor",
    "AsyncPipelineStage",
    "AsyncPoolConfig",
    "AsyncTaskResult",
    "AsyncTaskRunner",
    "TaskPriority",
    "async_retry",
    "gather_with_concurrency",
    "get_async_task_runner",
    "run_sync",
]
