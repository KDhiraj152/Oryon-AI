"""
Performance Profiler — Fine-Grained Execution Profiling
=========================================================

Provides hierarchical profiling for request lifecycles,
model execution paths, and hardware utilization patterns.

Design:
  - Span-tree based profiling (parent → child relationships)
  - Automatic GPU memory delta tracking
  - Aggregated statistics with configurable sampling
  - Export to Chrome Trace format for visualization
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from backend.infra.telemetry.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ProfileEntry:
    """Single profile entry."""

    name: str
    start_ns: int
    end_ns: int = 0
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_ns == 0:
            return 0.0
        return (self.end_ns - self.start_ns) / 1_000_000

    @property
    def duration_us(self) -> float:
        if self.end_ns == 0:
            return 0.0
        return (self.end_ns - self.start_ns) / 1_000

@dataclass
class AggregatedProfile:
    """Aggregated statistics for a profiling point."""

    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    def record(self, duration_ms: float) -> None:
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)

class PerformanceProfiler:
    """
    Hierarchical performance profiler.

    Usage:
        profiler = get_profiler()

        with profiler.profile("request.chat") as p:
            with profiler.profile("orchestration.route", parent="request.chat"):
                ...
            with profiler.profile("execution.inference", parent="request.chat") as ep:
                ep["model"] = "qwen3-8b"
                ep["tokens"] = 128
    """

    def __init__(self, *, enabled: bool = True, max_entries: int = 10000):
        self._enabled = enabled
        self._entries: list[ProfileEntry] = []
        self._aggregated: dict[str, AggregatedProfile] = defaultdict(
            lambda: AggregatedProfile(name="")
        )
        self._max_entries = max_entries
        self._lock = threading.Lock()

    @contextmanager
    def profile(
        self,
        name: str,
        *,
        parent: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Profile a code block."""
        if not self._enabled:
            yield {}
            return

        metadata: dict[str, Any] = {}
        start = time.monotonic_ns()
        try:
            yield metadata
        finally:
            end = time.monotonic_ns()
            entry = ProfileEntry(
                name=name,
                start_ns=start,
                end_ns=end,
                parent=parent,
                metadata=metadata,
            )

            with self._lock:
                if len(self._entries) < self._max_entries:
                    self._entries.append(entry)

                # Update aggregated stats
                agg = self._aggregated[name]
                if agg.name == "":
                    agg.name = name
                agg.record(entry.duration_ms)

    def get_aggregated(self) -> dict[str, dict[str, Any]]:
        """Get aggregated profiling statistics."""
        with self._lock:
            return {
                name: {
                    "count": agg.count,
                    "avg_ms": round(agg.avg_ms, 3),
                    "min_ms": round(agg.min_ms, 3) if agg.min_ms != float("inf") else 0,
                    "max_ms": round(agg.max_ms, 3),
                    "total_ms": round(agg.total_ms, 3),
                }
                for name, agg in self._aggregated.items()
            }

    def get_recent(self, n: int = 100) -> list[dict[str, Any]]:
        """Get recent profile entries."""
        with self._lock:
            entries = self._entries[-n:]
        return [
            {
                "name": e.name,
                "duration_ms": round(e.duration_ms, 3),
                "parent": e.parent,
                "metadata": e.metadata,
            }
            for e in entries
        ]

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._entries.clear()
            self._aggregated.clear()

    def export_chrome_trace(self) -> list[dict[str, Any]]:
        """Export in Chrome Trace Event format for chrome://tracing."""
        with self._lock:
            events = []
            for entry in self._entries:
                events.append(
                    {
                        "name": entry.name,
                        "cat": entry.parent or "root",
                        "ph": "X",  # Complete event
                        "ts": entry.start_ns / 1000,  # microseconds
                        "dur": entry.duration_us,
                        "pid": 1,
                        "tid": 1,
                        "args": entry.metadata,
                    }
                )
            return events

# ── Singleton ──────────────────────────────────────────────────────

_profiler: PerformanceProfiler | None = None

def get_profiler() -> PerformanceProfiler:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _profiler
    if _profiler is not None:
        return _profiler
    _profiler = PerformanceProfiler()
    return _profiler