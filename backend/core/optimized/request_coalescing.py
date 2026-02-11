"""
Request Coalescing for High Throughput
=======================================

Coalesces similar requests to reduce per-request overhead.
Instead of processing N identical/similar requests separately,
we process them once and return the result to all waiters.

Benefits:
- Reduces GPU contention
- Improves cache utilization
- Handles thundering herd problem
- Reduces memory pressure

Example: 10 users requesting the same text simplification
→ Only 1 GPU call, result shared to all 10 requests

M4 Optimizations:
- xxhash for fast request fingerprinting
- Lock-free hot path for cache hits
- Configurable coalesce windows per task type
"""

import asyncio
import hashlib
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

# Try xxhash for faster hashing (10-20x faster than hashlib)
try:
    import xxhash

    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False

T = TypeVar("T")


# ============================================================================
# COALESCE CONFIGURATION
# ============================================================================


class CoalesceTaskType(str, Enum):
    """Task types with different coalescing strategies."""

    EMBEDDING = "embedding"  # High coalesce - identical texts
    SIMPLIFY = "simplify"  # Medium coalesce - same text + grade
    TRANSLATE = "translate"  # Medium coalesce - same text + lang pair
    TTS = "tts"  # Low coalesce - same text + voice
    LLM = "llm"  # Very low - prompts rarely identical
    RERANK = "rerank"  # No coalesce - query-dependent


# Default coalesce windows (seconds)
COALESCE_WINDOWS: dict[CoalesceTaskType, float] = {
    CoalesceTaskType.EMBEDDING: 0.1,  # 100ms - high similarity
    CoalesceTaskType.SIMPLIFY: 0.05,  # 50ms - medium
    CoalesceTaskType.TRANSLATE: 0.05,  # 50ms - medium
    CoalesceTaskType.TTS: 0.02,  # 20ms - low
    CoalesceTaskType.LLM: 0.01,  # 10ms - very low
    CoalesceTaskType.RERANK: 0.0,  # No coalescing
}


# ============================================================================
# REQUEST FINGERPRINT
# ============================================================================


def compute_fingerprint(
    task_type: str,
    key_parts: tuple,
) -> str:
    """
    Compute fast fingerprint for request coalescing.

    Uses xxhash if available (10-20x faster than SHA256).
    """
    key_str = f"{task_type}:{':'.join(str(p) for p in key_parts)}"

    if _HAS_XXHASH:
        return xxhash.xxh64_hexdigest(key_str)
    else:
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


# ============================================================================
# COALESCING ENTRY
# ============================================================================


@dataclass
class CoalesceEntry(Generic[T]):
    """
    Entry for a coalesced request.

    Multiple requests can wait on the same entry.
    When the first request completes, all waiters get the result.
    """

    fingerprint: str
    task_type: CoalesceTaskType
    created_at: float = field(default_factory=time.perf_counter)
    completed: bool = False
    result: T | None = None
    error: Exception | None = None
    waiters: set[asyncio.Future] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def is_expired(self, window: float) -> bool:
        """Check if entry has exceeded coalesce window."""
        return (time.perf_counter() - self.created_at) > window

    def add_waiter(self, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        """Add a waiter and return future to await."""
        future = loop.create_future()
        self.waiters.add(future)
        return future

    def complete(self, result: T) -> None:
        """Complete all waiters with result."""
        self.completed = True
        self.result = result
        for future in self.waiters:
            if not future.done():
                future.set_result(result)

    def fail(self, error: Exception) -> None:
        """Fail all waiters with error."""
        self.completed = True
        self.error = error
        for future in self.waiters:
            if not future.done():
                future.set_exception(error)


# ============================================================================
# REQUEST COALESCER
# ============================================================================


class RequestCoalescer:
    """
    Coalesces similar requests to reduce redundant processing.

    Thread-safe and async-compatible.

    Usage:
        coalescer = RequestCoalescer()

        async def process_with_coalescing(text: str):
            fingerprint = compute_fingerprint("simplify", (text, complexity_level))

            result = await coalescer.coalesce_or_execute(
                fingerprint=fingerprint,
                task_type=CoalesceTaskType.SIMPLIFY,
                executor=lambda: simplify_text(text, complexity_level)
            )
            return result
    """

    def __init__(
        self,
        windows: dict[CoalesceTaskType, float] | None = None,
        max_entries: int = 10000,
        cleanup_interval: float = 10.0,
    ):
        self._windows = windows or COALESCE_WINDOWS
        self._max_entries = max_entries
        self._cleanup_interval = cleanup_interval

        # Per-task-type entries for better isolation
        self._entries: dict[CoalesceTaskType, dict[str, CoalesceEntry]] = defaultdict(
            dict
        )
        self._lock = asyncio.Lock()

        # Stats
        self._stats = {
            "total_requests": 0,
            "coalesced_requests": 0,
            "unique_executions": 0,
            "coalesce_ratio": 0.0,
        }

        # Cleanup task reference
        self._cleanup_task: asyncio.Task | None = None

    async def coalesce_or_execute(
        self,
        fingerprint: str,
        task_type: CoalesceTaskType,
        executor: Callable[[], Awaitable[T]],
    ) -> T:
        """
        Execute with coalescing.

        If an identical request is in-flight within the coalesce window,
        wait for it instead of executing again.

        Args:
            fingerprint: Request fingerprint for deduplication
            task_type: Task type (determines coalesce window)
            executor: Async function to execute if no coalesce hit

        Returns:
            Result from executor (or coalesced result)
        """
        window = self._windows.get(task_type, 0.0)

        # No coalescing for this task type
        if window <= 0:
            return await executor()

        self._stats["total_requests"] += 1

        loop = asyncio.get_running_loop()

        async with self._lock:
            entries = self._entries[task_type]

            # Check for existing entry
            if fingerprint in entries:
                entry = entries[fingerprint]

                # If not expired and not completed, coalesce
                if not entry.is_expired(window) and not entry.completed:
                    self._stats["coalesced_requests"] += 1
                    self._update_ratio()

                    # Add ourselves as waiter
                    future = entry.add_waiter(loop)

                    # Release lock while waiting
                    self._lock.release()
                    try:
                        return await future
                    finally:
                        await self._lock.acquire()

            # Create new entry
            entry = CoalesceEntry(
                fingerprint=fingerprint,
                task_type=task_type,
            )
            entries[fingerprint] = entry
            self._stats["unique_executions"] += 1
            self._update_ratio()

        # Execute outside lock
        try:
            result = await executor()

            async with self._lock:
                entry.complete(result)

            return result

        except Exception as e:
            async with self._lock:
                entry.fail(e)
            raise

        finally:
            # Schedule cleanup if not running
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def _update_ratio(self) -> None:
        """Update coalesce ratio stat."""
        total = self._stats["total_requests"]
        if total > 0:
            self._stats["coalesce_ratio"] = self._stats["coalesced_requests"] / total

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        await asyncio.sleep(self._cleanup_interval)

        async with self._lock:
            for task_type, entries in self._entries.items():
                window = self._windows.get(task_type, 0.0)
                expired = [
                    fp
                    for fp, entry in entries.items()
                    if entry.completed or entry.is_expired(window * 2)
                ]
                for fp in expired:
                    del entries[fp]

            # Trim if too many entries
            total_entries = sum(len(e) for e in self._entries.values())
            if total_entries > self._max_entries:
                # Remove oldest entries
                for _task_type, entries in self._entries.items():
                    if len(entries) > self._max_entries // len(CoalesceTaskType):
                        sorted_entries = sorted(
                            entries.items(), key=lambda x: x[1].created_at
                        )
                        for fp, _ in sorted_entries[: len(entries) // 2]:
                            del entries[fp]

    def get_stats(self) -> dict[str, Any]:
        """Get coalescing statistics."""
        return {
            **self._stats,
            "entries_by_type": {t.value: len(e) for t, e in self._entries.items()},
        }

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._entries.clear()


# ============================================================================
# GLOBAL COALESCER
# ============================================================================

_coalescer: RequestCoalescer | None = None
_coalescer_lock = threading.Lock()


def get_request_coalescer() -> RequestCoalescer:
    """Get global request coalescer singleton."""
    global _coalescer

    if _coalescer is None:
        with _coalescer_lock:
            if _coalescer is None:
                _coalescer = RequestCoalescer()

    return _coalescer


# ============================================================================
# DECORATOR FOR EASY COALESCING
# ============================================================================


def coalesce(
    task_type: CoalesceTaskType,
    key_extractor: Callable[..., tuple],
):
    """
    Decorator to add request coalescing to async functions.

    Usage:
        @coalesce(
            task_type=CoalesceTaskType.SIMPLIFY,
            key_extractor=lambda text, complexity_level: (text, complexity_level)
        )
        async def simplify_text(text: str, complexity_level: int) -> str:
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            coalescer = get_request_coalescer()

            # Extract key parts from arguments
            key_parts = key_extractor(*args, **kwargs)
            fingerprint = compute_fingerprint(task_type.value, key_parts)

            return await coalescer.coalesce_or_execute(
                fingerprint=fingerprint,
                task_type=task_type,
                executor=lambda: func(*args, **kwargs),
            )

        return wrapper

    return decorator


# ============================================================================
# EMBEDDING COALESCER (SPECIALIZED)
# ============================================================================


class EmbeddingCoalescer:
    """
    Specialized coalescer for embedding requests.

    Features:
    - Batches similar texts together
    - Deduplicates identical texts within batch
    - Returns proper indices for each caller

    Example:
        3 requests for texts ["a", "b", "a"]
        → 1 batch call for ["a", "b"]
        → Returns [emb_a, emb_b, emb_a]
    """

    def __init__(
        self,
        batch_size: int = 64,
        max_wait_ms: float = 50.0,
    ):
        self._batch_size = batch_size
        self._max_wait_ms = max_wait_ms

        # Pending texts and their waiters
        self._pending: dict[str, asyncio.Future] = {}
        self._pending_order: list = []
        self._lock = asyncio.Lock()

        # Batch execution task
        self._batch_task: asyncio.Task | None = None
        self._last_add_time: float = 0

        # Stats
        self._stats = {
            "texts_received": 0,
            "unique_texts": 0,
            "batches_executed": 0,
        }

    async def embed(
        self,
        text: str,
        embedder: Callable[[list], Any],
    ) -> Any:
        """
        Get embedding for text, with deduplication.

        Args:
            text: Text to embed
            embedder: Function that takes list of texts, returns embeddings

        Returns:
            Embedding for the text
        """
        self._stats["texts_received"] += 1
        loop = asyncio.get_running_loop()

        async with self._lock:
            # Check if already pending
            if text in self._pending:
                future = self._pending[text]
            else:
                # Add new pending text
                future = loop.create_future()
                self._pending[text] = future
                self._pending_order.append(text)
                self._stats["unique_texts"] += 1

            self._last_add_time = time.perf_counter()

            # Start batch task if needed
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._batch_execute(embedder))

        return await future

    async def _batch_execute(self, embedder: Callable[[list], Any]) -> None:
        """Execute batch when ready."""
        # Wait for batch to fill or timeout
        while True:
            await asyncio.sleep(self._max_wait_ms / 1000)

            async with self._lock:
                # Check if we have enough or timed out
                time_since_last = (time.perf_counter() - self._last_add_time) * 1000

                if (
                    len(self._pending_order) >= self._batch_size
                    or time_since_last >= self._max_wait_ms
                ):
                    # Execute batch
                    if not self._pending_order:
                        return

                    texts = self._pending_order[: self._batch_size]
                    futures = [self._pending[t] for t in texts]

                    # Clear processed
                    for t in texts:
                        del self._pending[t]
                    self._pending_order = self._pending_order[self._batch_size :]

                    break

        # Execute outside lock
        try:
            embeddings = await asyncio.get_running_loop().run_in_executor(
                None, embedder, texts
            )

            self._stats["batches_executed"] += 1

            # Distribute results
            for future, embedding in zip(futures, embeddings, strict=False):
                if not future.done():
                    future.set_result(embedding)

        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics."""
        return {
            **self._stats,
            "dedup_ratio": (
                1
                - (self._stats["unique_texts"] / max(1, self._stats["texts_received"]))
            ),
        }


# Singleton
_embedding_coalescer: EmbeddingCoalescer | None = None


def get_embedding_coalescer() -> EmbeddingCoalescer:
    """Get global embedding coalescer."""
    global _embedding_coalescer

    if _embedding_coalescer is None:
        _embedding_coalescer = EmbeddingCoalescer()

    return _embedding_coalescer
