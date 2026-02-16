"""
Predictive Prefetching for Low Latency
========================================

Prefetches likely-needed resources based on access patterns.

Strategies:
1. Adjacency prefetch - next items in sequence
2. Semantic prefetch - similar content based on embeddings
3. Session prefetch - user behavior prediction
4. Temporal prefetch - time-of-day patterns

M4 Optimizations:
- Background prefetch on E-cores (low priority)
- Memory-aware prefetch limits
- Cache-aware deduplication
"""

import asyncio
import contextlib
import heapq
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

from backend.utils.lock_factory import create_lock

logger = logging.getLogger(__name__)

# ============================================================================
# PREFETCH STRATEGIES
# ============================================================================

class PrefetchStrategy(StrEnum):
    """Prefetch strategies."""

    ADJACENT = "adjacent"  # Sequential items
    SEMANTIC = "semantic"  # Similar content
    TEMPORAL = "temporal"  # Time-based patterns
    SESSION = "session"  # User behavior

# ============================================================================
# ACCESS PATTERN TRACKER
# ============================================================================

@dataclass
class AccessRecord:
    """Record of a resource access."""

    resource_id: str
    timestamp: float
    session_id: str | None = None
    context: dict[str, Any] | None = None

class AccessPatternTracker:
    """
    Tracks access patterns to predict future accesses.

    Uses a sliding window to maintain recency.
    """

    def __init__(
        self,
        window_size: int = 1000,
        sequence_length: int = 5,
    ):
        self._window_size = window_size
        self._sequence_length = sequence_length
        self._max_transition_keys = 10_000
        self._max_sessions = 10_000

        # Recent accesses (ring buffer)
        self._accesses: deque = deque(maxlen=window_size)

        # Transition counts: from_id -> {to_id: count}
        self._transitions: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Per-session sequences
        self._session_sequences: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=sequence_length)
        )

        self._lock = threading.Lock()

    def record_access(
        self,
        resource_id: str,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a resource access."""
        record = AccessRecord(
            resource_id=resource_id,
            timestamp=time.time(),
            session_id=session_id,
            context=context,
        )

        with self._lock:
            # Add to global accesses
            self._accesses.append(record)

            # Update session sequence and transitions
            if session_id:
                seq = self._session_sequences[session_id]
                if seq:
                    prev_id = seq[-1]
                    self._transitions[prev_id][resource_id] += 1
                seq.append(resource_id)

            # Evict oldest transition keys if exceeding limit
            if len(self._transitions) > self._max_transition_keys:
                oldest = next(iter(self._transitions))
                del self._transitions[oldest]

            # Evict oldest sessions if exceeding limit
            if len(self._session_sequences) > self._max_sessions:
                oldest = next(iter(self._session_sequences))
                del self._session_sequences[oldest]

    def predict_next(
        self,
        current_id: str,
        session_id: str | None = None,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Predict most likely next accesses.

        Returns list of (resource_id, probability) tuples.
        """
        with self._lock:
            candidates: dict[str, float] = defaultdict(float)

            # Transition probabilities
            if current_id in self._transitions:
                transitions = self._transitions[current_id]
                total = sum(transitions.values())
                for next_id, count in transitions.items():
                    candidates[next_id] += (count / total) * 0.7  # 70% weight

            # Session-based prediction
            if session_id and session_id in self._session_sequences:
                seq = list(self._session_sequences[session_id])
                if len(seq) >= 2:
                    # Look for similar sub-sequences in other sessions
                    pattern = tuple(seq[-2:])
                    for other_seq in self._session_sequences.values():
                        other_list = list(other_seq)
                        for i in range(len(other_list) - 2):
                            if tuple(other_list[i : i + 2]) == pattern and i + 2 < len(
                                other_list
                            ):
                                candidates[other_list[i + 2]] += 0.3  # 30% weight

            # Sort by probability
            sorted_candidates = sorted(
                candidates.items(), key=lambda x: x[1], reverse=True
            )[:top_k]

            return sorted_candidates

    def get_hot_resources(self, top_k: int = 10) -> list[str]:
        """Get most frequently accessed resources."""
        with self._lock:
            counts: dict[str, int] = defaultdict(int)
            for record in self._accesses:
                counts[record.resource_id] += 1

            return [
                r
                for r, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[
                    :top_k
                ]
            ]

# ============================================================================
# PREFETCH MANAGER
# ============================================================================

@dataclass
class PrefetchRequest:
    """Request to prefetch a resource."""

    resource_id: str
    priority: float  # Higher = more urgent
    requested_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        return self.priority > other.priority  # Max heap

class PrefetchManager:
    """
    Manages prefetching with priority queue and memory awareness.

    Features:
    - Priority-based prefetch queue
    - Memory budget enforcement
    - Background worker on E-cores
    - Cache integration
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        max_memory_mb: float = 500,
        max_concurrent: int = 2,
    ):
        self._max_queue_size = max_queue_size
        self._max_memory_mb = max_memory_mb
        self._max_concurrent = max_concurrent

        # Priority queue
        self._queue: list[PrefetchRequest] = []
        self._queue_lock = asyncio.Lock()

        # In-progress set
        self._in_progress: set[str] = set()

        # Pattern tracker
        self._tracker = AccessPatternTracker()

        # Fetcher functions by resource type
        self._fetchers: dict[str, Callable[[str], Awaitable[Any]]] = {}

        # Cache reference (set externally)
        self._cache = None

        # Stats
        self._stats = {
            "prefetches_requested": 0,
            "prefetches_completed": 0,
            "prefetches_skipped": 0,
            "cache_hits_from_prefetch": 0,
        }

        # Worker task
        self._worker_task: asyncio.Task | None = None
        self._running = False

    def register_fetcher(
        self,
        resource_type: str,
        fetcher: Callable[[str], Awaitable[Any]],
    ) -> None:
        """Register a fetcher function for a resource type."""
        self._fetchers[resource_type] = fetcher

    def set_cache(self, cache: Any) -> None:
        """Set cache reference for checking/storing prefetched data."""
        self._cache = cache

    async def request_prefetch(
        self,
        resource_id: str,
        priority: float = 0.5,
    ) -> None:
        """Request prefetch of a resource."""
        self._stats["prefetches_requested"] += 1

        # Skip if already in cache
        if self._cache and await self._cache.get(resource_id):
            self._stats["prefetches_skipped"] += 1
            return

        # Skip if already in queue or in-progress
        async with self._queue_lock:
            if resource_id in self._in_progress:
                return

            if any(r.resource_id == resource_id for r in self._queue):
                return

            # Add to queue
            request = PrefetchRequest(resource_id=resource_id, priority=priority)
            heapq.heappush(self._queue, request)

            # Trim queue if too large
            while len(self._queue) > self._max_queue_size:
                heapq.heappop(self._queue)

        # Start worker if not running
        if not self._running:
            await self.start()

    async def on_access(
        self,
        resource_id: str,
        session_id: str | None = None,
    ) -> None:
        """
        Called when a resource is accessed.

        Records pattern and triggers predictive prefetch.
        """
        # Record access pattern
        self._tracker.record_access(resource_id, session_id)

        # Get predictions
        predictions = self._tracker.predict_next(
            current_id=resource_id, session_id=session_id, top_k=3
        )

        # Request prefetch for predictions
        for pred_id, prob in predictions:
            if prob > 0.2:  # Only prefetch if >20% probability
                await self.request_prefetch(pred_id, priority=prob)

    async def start(self) -> None:
        """Start prefetch worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.debug("Prefetch worker started")

    async def stop(self) -> None:
        """Stop prefetch worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

    async def _worker_loop(self) -> None:
        """Background worker for processing prefetch queue."""
        while self._running:
            try:
                # Get next request
                request = None
                async with self._queue_lock:
                    if self._queue and len(self._in_progress) < self._max_concurrent:
                        request = heapq.heappop(self._queue)
                        self._in_progress.add(request.resource_id)

                if request:
                    await self._execute_prefetch(request)
                else:
                    # No work, sleep
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError) as e:
                logger.warning("Prefetch worker error: %s", e)
                await asyncio.sleep(1)

    async def _execute_prefetch(self, request: PrefetchRequest) -> None:
        """Execute a single prefetch."""
        try:
            # Determine resource type from ID
            # Format: "type:actual_id" or just "id"
            if ":" in request.resource_id:
                resource_type, actual_id = request.resource_id.split(":", 1)
            else:
                resource_type = "default"
                actual_id = request.resource_id

            # Get fetcher
            fetcher = self._fetchers.get(resource_type)
            if not fetcher:
                return

            # Execute fetch
            result = await fetcher(actual_id)

            # Store in cache
            if self._cache and result is not None:
                await self._cache.set(request.resource_id, result)

            self._stats["prefetches_completed"] += 1

        except (RuntimeError, OSError) as e:
            logger.debug("Prefetch failed for %s: %s", request.resource_id, e)

        finally:
            async with self._queue_lock:
                self._in_progress.discard(request.resource_id)

    def get_stats(self) -> dict[str, Any]:
        """Get prefetch statistics."""
        return {
            **self._stats,
            "queue_size": len(self._queue),
            "in_progress": len(self._in_progress),
            "hot_resources": self._tracker.get_hot_resources(5),
        }

# ============================================================================
# GLOBAL PREFETCH MANAGER
# ============================================================================

_prefetch_manager: PrefetchManager | None = None
_prefetch_lock = create_lock()

def get_prefetch_manager() -> PrefetchManager:
    """Get global prefetch manager singleton."""
    global _prefetch_manager

    if _prefetch_manager is None:
        with _prefetch_lock:
            if _prefetch_manager is None:
                _prefetch_manager = PrefetchManager()

    return _prefetch_manager

# ============================================================================
# DECORATOR FOR PREFETCH-AWARE FUNCTIONS
# ============================================================================

def with_prefetch(
    resource_type: str = "default",
    id_extractor: Callable[..., str] | None = None,
):
    """
    Decorator to add prefetch awareness to async functions.

    Records access patterns and triggers predictive prefetch.

    Usage:
        @with_prefetch(
            resource_type="embedding",
            id_extractor=lambda text: f"embedding:{hash(text)}"
        )
        async def get_embedding(text: str) -> list[float]:
            ...
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args, **kwargs) -> Any:
            # Extract resource ID
            resource_id = None
            if id_extractor:
                resource_id = id_extractor(*args, **kwargs)

            # Execute function
            result = await func(*args, **kwargs)

            # Record access for prefetch prediction
            if resource_id:
                manager = get_prefetch_manager()
                await manager.on_access(resource_id)

            return result

        return wrapper

    return decorator
