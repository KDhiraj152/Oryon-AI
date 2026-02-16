"""
Latency Controller — Phase 4
==============================

Early-exit strategies, confidence-based stopping, and parallel inference.

Design:
- LatencyBudget tracks remaining time per-request and triggers early exit
- Confidence gate: if intermediate output exceeds confidence threshold, stop early
- Parallel inference: fan-out to multiple backends, take first valid result
- Adaptive timeout: adjusts per-stage timeouts based on historical latency
- All state per-request — no global mutable state
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# ── Enums ────────────────────────────────────────────────────────────────────

class ExitReason(StrEnum):
    COMPLETED = "completed"
    EARLY_EXIT_CONFIDENCE = "early_exit_confidence"
    EARLY_EXIT_TIMEOUT = "early_exit_timeout"
    EARLY_EXIT_BUDGET = "early_exit_budget"
    PARALLEL_RACE_WIN = "parallel_race_win"
    FAILED = "failed"

# ── Data Contracts ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LatencyBudget:
    """Per-request latency budget.

    Created at the start of a request, consumed by stages.
    Immutable — stages check remaining budget rather than mutating it.
    """

    total_ms: float
    created_at: float = field(default_factory=lambda: time.perf_counter())
    stage_budgets: dict[str, float] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.created_at) * 1000

    @property
    def remaining_ms(self) -> float:
        return max(0, self.total_ms - self.elapsed_ms)

    @property
    def is_expired(self) -> bool:
        return self.remaining_ms <= 0

    def budget_for_stage(self, stage_name: str, default_ms: float = 5000.0) -> float:
        """Get the budget for a specific stage, clamped to remaining."""
        stage_budget = self.stage_budgets.get(stage_name, default_ms)
        return min(stage_budget, self.remaining_ms)

    def with_stage_budgets(self, budgets: dict[str, float]) -> LatencyBudget:
        """Create a new budget with updated stage allocations."""
        return LatencyBudget(
            total_ms=self.total_ms,
            created_at=self.created_at,
            stage_budgets={**self.stage_budgets, **budgets},
        )

@dataclass(frozen=True, slots=True)
class LatencyResult:
    """Result from a latency-controlled operation."""

    value: Any
    latency_ms: float
    exit_reason: ExitReason
    confidence: float = 0.0
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

# ── Confidence Gate ──────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    """Configuration for confidence-based stopping."""

    threshold: float = 0.85       # Stop if confidence ≥ this
    min_latency_ms: float = 50.0  # Don't stop before this time
    max_latency_ms: float = 0.0   # 0 = no max (use budget)
    fallback_on_low: bool = True  # Try fallback model if confidence low

class ConfidenceGate:
    """Evaluates whether an intermediate result is good enough to return early."""

    def __init__(self, config: ConfidenceConfig | None = None):
        self._config = config or ConfidenceConfig()

    def should_early_exit(
        self,
        confidence: float,
        latency_ms: float,
        budget: LatencyBudget,
    ) -> bool:
        """Decide whether to stop early based on confidence and timing."""
        # Don't exit too early
        if latency_ms < self._config.min_latency_ms:
            return False

        # High confidence → exit
        if confidence >= self._config.threshold:
            return True

        # Budget pressure → accept lower confidence
        budget_used_ratio = 1.0 - (budget.remaining_ms / max(1, budget.total_ms))
        adjusted_threshold = self._config.threshold * (1.0 - 0.2 * budget_used_ratio)
        return bool(confidence >= adjusted_threshold and budget_used_ratio > 0.7)

    def should_try_fallback(
        self,
        confidence: float,
        budget: LatencyBudget,
    ) -> bool:
        """Check if we should attempt a fallback model."""
        if not self._config.fallback_on_low:
            return False
        return confidence < self._config.threshold * 0.6 and budget.remaining_ms > 2000

# ── Parallel Inference ───────────────────────────────────────────────────────

InferenceFunc = Callable[..., Awaitable[tuple[Any, float]]]
# Returns (result, confidence)

async def parallel_inference(
    funcs: list[InferenceFunc],
    budget: LatencyBudget,
    min_confidence: float = 0.5,
) -> LatencyResult:
    """
    Fan out to multiple inference backends; return first valid result.

    Strategy:
    - Launch all functions concurrently
    - First result meeting min_confidence wins
    - Cancel remaining tasks
    - If none meet threshold, return best result
    """
    if not funcs:
        return LatencyResult(
            value=None,
            latency_ms=0.0,
            exit_reason=ExitReason.FAILED,
        )

    t0 = time.perf_counter()
    timeout = budget.remaining_ms / 1000.0

    tasks: list[asyncio.Task[tuple[Any, float]]] = [
        asyncio.create_task(fn(), name=f"inference-{i}")  # type: ignore[arg-type]
        for i, fn in enumerate(funcs)
    ]

    best_result: tuple[Any, float] | None = None
    best_confidence = 0.0

    try:
        done: set[asyncio.Task] = set()
        pending = set(tasks)

        while pending:
            completed, pending = await asyncio.wait(
                pending,
                timeout=max(0.01, timeout - (time.perf_counter() - t0)),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in completed:
                done.add(task)
                try:
                    result, confidence = task.result()
                    if confidence >= min_confidence:
                        # Winner — cancel the rest
                        for p in pending:
                            p.cancel()
                        latency = (time.perf_counter() - t0) * 1000
                        return LatencyResult(
                            value=result,
                            latency_ms=round(latency, 2),
                            exit_reason=ExitReason.PARALLEL_RACE_WIN,
                            confidence=confidence,
                            attempts=len(done),
                        )
                    # Track best so far
                    if confidence > best_confidence:
                        best_result = (result, confidence)
                        best_confidence = confidence
                except (RuntimeError, ValueError, OSError) as e:
                    logger.debug("Parallel inference task failed: %s", e)

            if not pending:
                break

            # Check overall timeout
            if (time.perf_counter() - t0) * 1000 >= budget.remaining_ms:
                for p in pending:
                    p.cancel()
                break

    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Parallel inference error: %s", e)

    latency = (time.perf_counter() - t0) * 1000

    if best_result:
        return LatencyResult(
            value=best_result[0],
            latency_ms=round(latency, 2),
            exit_reason=ExitReason.EARLY_EXIT_TIMEOUT,
            confidence=best_result[1],
            attempts=len(done),
        )

    return LatencyResult(
        value=None,
        latency_ms=round(latency, 2),
        exit_reason=ExitReason.FAILED,
        attempts=len(done) if 'done' in dir() else 0,
    )

# ── Adaptive Timeout ─────────────────────────────────────────────────────────

class AdaptiveTimeout:
    """
    Tracks historical latencies per stage and adjusts timeouts.

    Formula: timeout = P95 * multiplier (clamped to [min, max])
    """

    def __init__(
        self,
        window_size: int = 200,
        multiplier: float = 1.5,
        min_timeout_ms: float = 100.0,
        max_timeout_ms: float = 30_000.0,
    ):
        self._window_size = window_size
        self._multiplier = multiplier
        self._min = min_timeout_ms
        self._max = max_timeout_ms
        self._history: dict[str, deque[float]] = {}

    def record(self, stage_name: str, latency_ms: float) -> None:
        """Record a latency observation."""
        if stage_name not in self._history:
            self._history[stage_name] = deque(maxlen=self._window_size)
        self._history[stage_name].append(latency_ms)

    def get_timeout_ms(self, stage_name: str, default_ms: float = 5000.0) -> float:
        """Get adaptive timeout for a stage."""
        history = self._history.get(stage_name)
        if not history or len(history) < 10:
            return default_ms

        sorted_latencies = sorted(history)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]

        timeout = p95 * self._multiplier
        return max(self._min, min(self._max, timeout))

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get latency summary per stage."""
        result = {}
        for name, history in self._history.items():
            if not history:
                continue
            sorted_h = sorted(history)
            n = len(sorted_h)
            result[name] = {
                "count": n,
                "p50_ms": round(sorted_h[int(n * 0.5)], 2),
                "p95_ms": round(sorted_h[min(int(n * 0.95), n - 1)], 2),
                "p99_ms": round(sorted_h[min(int(n * 0.99), n - 1)], 2),
                "timeout_ms": round(self.get_timeout_ms(name), 2),
            }
        return result

# ── Latency Controller ───────────────────────────────────────────────────────

class LatencyController:
    """
    Unified latency control facade.

    Combines:
    - Budget allocation per request
    - Confidence-based early exit
    - Adaptive timeouts learned from history
    - Parallel inference orchestration
    """

    # Default per-stage budget allocation (ms)
    DEFAULT_STAGE_BUDGETS: ClassVar[dict[str, float]] = {
        "classify": 200,
        "rag_retrieve": 3000,
        "generate": 20000,
        "translate": 5000,
        "tts": 5000,
        "stt": 8000,
        "ocr": 6000,
        "embed": 2000,
        "rerank": 1000,
        "format": 1000,
        "simplify": 10000,
    }

    def __init__(
        self,
        confidence_config: ConfidenceConfig | None = None,
        default_budget_ms: float = 30_000.0,
    ):
        self.confidence_gate = ConfidenceGate(confidence_config)
        self.adaptive_timeout = AdaptiveTimeout()
        self._default_budget = default_budget_ms
        self._total_requests = 0
        self._early_exits = 0
        self._parallel_wins = 0

    # ── Public API ───────────────────────────────────────────────────

    def create_budget(
        self,
        total_ms: float | None = None,
        stage_overrides: dict[str, float] | None = None,
    ) -> LatencyBudget:
        """Create a latency budget for a new request."""
        budgets = dict(self.DEFAULT_STAGE_BUDGETS)

        # Override adaptive timeouts from history
        for stage_name in budgets:
            adaptive = self.adaptive_timeout.get_timeout_ms(
                stage_name, budgets[stage_name]
            )
            budgets[stage_name] = adaptive

        # Apply explicit overrides
        if stage_overrides:
            budgets.update(stage_overrides)

        self._total_requests += 1

        return LatencyBudget(
            total_ms=total_ms or self._default_budget,
            stage_budgets=budgets,
        )

    def check_early_exit(
        self,
        confidence: float,
        latency_ms: float,
        budget: LatencyBudget,
    ) -> bool:
        """Check if we should stop early."""
        should_exit = self.confidence_gate.should_early_exit(
            confidence, latency_ms, budget
        )
        if should_exit:
            self._early_exits += 1
        return should_exit

    def record_latency(self, stage_name: str, latency_ms: float) -> None:
        """Record a stage latency for adaptive timeout learning."""
        self.adaptive_timeout.record(stage_name, latency_ms)

    async def race_inference(
        self,
        funcs: list[InferenceFunc],
        budget: LatencyBudget,
        min_confidence: float = 0.5,
    ) -> LatencyResult:
        """Run parallel inference race."""
        result = await parallel_inference(funcs, budget, min_confidence)
        if result.exit_reason == ExitReason.PARALLEL_RACE_WIN:
            self._parallel_wins += 1
        return result

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_requests": self._total_requests,
            "early_exits": self._early_exits,
            "parallel_wins": self._parallel_wins,
            "early_exit_rate": round(
                self._early_exits / max(1, self._total_requests), 4
            ),
            "adaptive_timeouts": self.adaptive_timeout.get_stats(),
        }
