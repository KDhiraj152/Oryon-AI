"""
Self-Evaluation Loop — Phase 5
================================

Measures output quality, latency, and routing accuracy.
Logs routing decisions and adjusts heuristics over time.

Design:
- EvaluationReport captures per-request telemetry
- Routing ledger records (decision, outcome) pairs for heuristic learning
- Quality estimator scores outputs without requiring ground truth
- Heuristic adjuster tunes routing thresholds based on observed outcomes
- All adjustments are bounded and reversible
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# ── Data Contracts ───────────────────────────────────────────────────────────

class QualitySignal(StrEnum):
    """Observable quality signals (no ground truth required)."""

    GOOD = "good"                # High confidence, fast, no errors
    ACCEPTABLE = "acceptable"    # Met SLA, moderate confidence
    DEGRADED = "degraded"        # SLA miss, low confidence
    FAILED = "failed"            # Error or timeout

@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Per-request evaluation telemetry."""

    request_id: str
    intent: str
    model_target: str
    model_used: str
    complexity_score: float

    # Timing
    total_latency_ms: float
    stage_latencies: dict[str, float] = field(default_factory=dict)

    # Quality
    confidence: float = 0.0
    quality_signal: QualitySignal = QualitySignal.ACCEPTABLE
    output_tokens: int = 0
    was_early_exit: bool = False
    used_cache: bool = False
    used_rag: bool = False

    # Routing
    routing_was_correct: bool | None = None  # Set by feedback
    could_have_used_smaller: bool = False

    timestamp: float = field(default_factory=time.time)

@dataclass
class RoutingOutcome:
    """Records a routing decision and its outcome for learning."""

    intent: str
    complexity_score: float
    model_target: str
    quality_signal: QualitySignal
    latency_ms: float
    confidence: float
    timestamp: float = field(default_factory=time.time)

# ── Quality Estimator ────────────────────────────────────────────────────────

class QualityEstimator:
    """
    Estimates output quality without ground truth.

    Uses proxy signals:
    - Confidence from the model
    - Latency relative to SLA
    - Presence of errors / empty output
    - Token utilization (too few tokens → might be truncated)
    """

    # SLA targets per task type (ms)
    SLA_TARGETS: ClassVar[dict[str, float]] = {
        "chat": 5000,
        "question": 5000,
        "translation": 3000,
        "simplification": 5000,
        "summarization": 5000,
        "code": 8000,
        "calculation": 1000,
        "audio": 4000,
        "quiz": 5000,
        "creative": 8000,
        "conversation": 2000,
        "embedding": 200,
        "reranking": 500,
        "stt": 8000,
        "ocr": 6000,
    }

    def estimate(
        self,
        intent: str,
        confidence: float,
        latency_ms: float,
        output_tokens: int,
        has_error: bool = False,
    ) -> QualitySignal:
        """Estimate quality from observable signals."""
        if has_error:
            return QualitySignal.FAILED

        sla = self.SLA_TARGETS.get(intent, 5000)

        # Score components
        confidence_score = confidence  # 0-1

        latency_ratio = latency_ms / sla
        latency_score = max(0, 1.0 - max(0, latency_ratio - 0.5))

        # Token utilization (rough: empty or very short output is suspicious)
        token_score = min(1.0, output_tokens / 50) if output_tokens > 0 else 0.0

        # Weighted composite
        composite = (
            confidence_score * 0.45
            + latency_score * 0.35
            + token_score * 0.20
        )

        if composite >= 0.75:
            return QualitySignal.GOOD
        elif composite >= 0.50:
            return QualitySignal.ACCEPTABLE
        else:
            return QualitySignal.DEGRADED

# ── Routing Ledger ───────────────────────────────────────────────────────────

class RoutingLedger:
    """
    Records routing decisions and outcomes for learning.

    Maintains per-(intent, model_target) statistics to detect
    over-provisioning (strong model used for simple tasks) and
    under-provisioning (lightweight model producing poor quality).
    """

    def __init__(self, max_entries: int = 5000, max_stats_keys: int = 500):
        self._history: deque[RoutingOutcome] = deque(maxlen=max_entries)
        self._stats: dict[str, _RouteStats] = defaultdict(_RouteStats)
        self._max_stats_keys = max_stats_keys

    def record(self, outcome: RoutingOutcome) -> None:
        """Record a routing outcome."""
        self._history.append(outcome)
        key = f"{outcome.intent}:{outcome.model_target}"
        stats = self._stats[key]
        stats.total += 1
        stats.total_latency_ms += outcome.latency_ms
        stats.total_confidence += outcome.confidence

        if outcome.quality_signal == QualitySignal.GOOD:
            stats.good += 1
        elif outcome.quality_signal == QualitySignal.FAILED:
            stats.failed += 1
        elif outcome.quality_signal == QualitySignal.DEGRADED:
            stats.degraded += 1

        # Prune least-used keys when exceeding max
        if len(self._stats) > self._max_stats_keys:
            # Remove keys with fewest total observations
            least_key = min(self._stats, key=lambda k: self._stats[k].total)
            del self._stats[least_key]

    def get_quality_rate(self, intent: str, model_target: str) -> float:
        """Get the good-quality rate for a (intent, model) pair."""
        key = f"{intent}:{model_target}"
        stats = self._stats.get(key)
        if not stats or stats.total < 5:
            return 0.5  # Not enough data
        return stats.good / stats.total

    def get_failure_rate(self, intent: str, model_target: str) -> float:
        key = f"{intent}:{model_target}"
        stats = self._stats.get(key)
        if not stats or stats.total < 5:
            return 0.0
        return stats.failed / stats.total

    def get_avg_latency(self, intent: str, model_target: str) -> float:
        key = f"{intent}:{model_target}"
        stats = self._stats.get(key)
        if not stats or stats.total == 0:
            return 0.0
        return stats.total_latency_ms / stats.total

    def detect_over_provisioning(self) -> list[dict[str, Any]]:
        """Detect routes where a smaller model would suffice.

        Signal: high quality rate with a strong model → could downgrade.
        """
        issues = []
        for key, stats in self._stats.items():
            if stats.total < 20:
                continue
            intent, model_target = key.split(":", 1)
            quality_rate = stats.good / stats.total
            if model_target in ("strong",) and quality_rate > 0.90:
                issues.append({
                    "intent": intent,
                    "model_target": model_target,
                    "quality_rate": round(quality_rate, 3),
                    "suggestion": "Downgrade to standard — consistently high quality",
                    "sample_size": stats.total,
                })
        return issues

    def detect_under_provisioning(self) -> list[dict[str, Any]]:
        """Detect routes where a larger model is needed.

        Signal: high degraded/failed rate with a lightweight model → upgrade.
        """
        issues = []
        for key, stats in self._stats.items():
            if stats.total < 20:
                continue
            intent, model_target = key.split(":", 1)
            bad_rate = (stats.degraded + stats.failed) / stats.total
            if model_target in ("lightweight",) and bad_rate > 0.25:
                issues.append({
                    "intent": intent,
                    "model_target": model_target,
                    "bad_rate": round(bad_rate, 3),
                    "suggestion": "Upgrade to standard — too many failures",
                    "sample_size": stats.total,
                })
        return issues

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_outcomes": len(self._history),
            "routes": {
                key: {
                    "total": s.total,
                    "good_rate": round(s.good / max(1, s.total), 3),
                    "fail_rate": round(s.failed / max(1, s.total), 3),
                    "avg_latency_ms": round(s.total_latency_ms / max(1, s.total), 1),
                }
                for key, s in self._stats.items()
            },
            "over_provisioned": self.detect_over_provisioning(),
            "under_provisioned": self.detect_under_provisioning(),
        }

@dataclass
class _RouteStats:
    total: int = 0
    good: int = 0
    degraded: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0
    total_confidence: float = 0.0

# ── Heuristic Adjuster ──────────────────────────────────────────────────────

class HeuristicAdjuster:
    """
    Adjusts routing heuristics based on observed outcomes.

    Maintains a set of tunable parameters and applies bounded
    adjustments based on the routing ledger's analysis.

    All adjustments are bounded (min/max) and logged for auditability.
    """

    def __init__(self):
        # Tunable thresholds
        self._complexity_thresholds: dict[str, float] = {
            # intent → complexity_score above which we upgrade model tier
            "question": 0.30,
            "code": 0.15,
            "creative": 0.35,
            "conversation": 0.60,
            "summarization": 0.40,
            "simplification": 0.40,
        }

        # Bounds
        self._min_threshold = 0.10
        self._max_threshold = 0.80
        self._adjustment_step = 0.05

        # Adjustment log
        self._adjustments: deque[dict[str, Any]] = deque(maxlen=500)
        self._adjustment_count = 0

    def get_threshold(self, intent: str) -> float:
        return self._complexity_thresholds.get(intent, 0.35)

    def adjust_from_ledger(self, ledger: RoutingLedger) -> list[dict[str, Any]]:
        """
        Analyze the routing ledger and make bounded adjustments.

        Returns list of adjustments made.
        """
        changes = []

        # Over-provisioning → lower complexity threshold (route to smaller model earlier)
        for issue in ledger.detect_over_provisioning():
            intent = issue["intent"]
            old = self.get_threshold(intent)
            new = max(self._min_threshold, old - self._adjustment_step)
            if new != old:
                self._complexity_thresholds[intent] = new
                change = {
                    "type": "lower_threshold",
                    "intent": intent,
                    "old": round(old, 3),
                    "new": round(new, 3),
                    "reason": issue["suggestion"],
                    "timestamp": time.time(),
                }
                changes.append(change)
                self._adjustments.append(change)
                self._adjustment_count += 1

        # Under-provisioning → raise complexity threshold (route to larger model later)
        for issue in ledger.detect_under_provisioning():
            intent = issue["intent"]
            old = self.get_threshold(intent)
            new = min(self._max_threshold, old + self._adjustment_step)
            if new != old:
                self._complexity_thresholds[intent] = new
                change = {
                    "type": "raise_threshold",
                    "intent": intent,
                    "old": round(old, 3),
                    "new": round(new, 3),
                    "reason": issue["suggestion"],
                    "timestamp": time.time(),
                }
                changes.append(change)
                self._adjustments.append(change)
                self._adjustment_count += 1

        if changes:
            logger.info("Heuristic adjuster made %s adjustments: %s", len(changes), changes)

        return changes

    def get_stats(self) -> dict[str, Any]:
        return {
            "thresholds": dict(self._complexity_thresholds),
            "total_adjustments": self._adjustment_count,
            "recent_adjustments": list(self._adjustments)[-10:],
        }

# ── Self-Evaluator ───────────────────────────────────────────────────────────

class SelfEvaluator:
    """
    Unified self-evaluation loop.

    Workflow:
    1. After each request → record evaluation report
    2. Quality estimator scores the output
    3. Routing ledger records the (decision, outcome) pair
    4. Periodically → heuristic adjuster tunes routing thresholds
    5. All telemetry exposed for observability

    This class is the single integration point for all Phase 5 components.
    """

    def __init__(
        self,
        adjustment_interval: int = 100,
        max_reports: int = 5000,
    ):
        self.quality_estimator = QualityEstimator()
        self.routing_ledger = RoutingLedger(max_entries=max_reports)
        self.heuristic_adjuster = HeuristicAdjuster()

        self._adjustment_interval = adjustment_interval
        self._reports: deque[EvaluationReport] = deque(maxlen=max_reports)
        self._requests_since_adjustment = 0
        self._total_adjustments_applied = 0

    # ── Public API ───────────────────────────────────────────────────

    def evaluate(
        self,
        request_id: str,
        intent: str,
        model_target: str,
        model_used: str,
        complexity_score: float,
        total_latency_ms: float,
        confidence: float = 0.0,
        output_tokens: int = 0,
        has_error: bool = False,
        was_early_exit: bool = False,
        used_cache: bool = False,
        used_rag: bool = False,
        stage_latencies: dict[str, float] | None = None,
    ) -> EvaluationReport:
        """
        Evaluate a completed request and record telemetry.

        Called after every request completes.
        """
        # Estimate quality
        quality = self.quality_estimator.estimate(
            intent=intent,
            confidence=confidence,
            latency_ms=total_latency_ms,
            output_tokens=output_tokens,
            has_error=has_error,
        )

        # Check if we over-provisioned
        could_downgrade = (
            quality == QualitySignal.GOOD
            and model_target in ("strong",)
            and confidence > 0.85
        )

        report = EvaluationReport(
            request_id=request_id,
            intent=intent,
            model_target=model_target,
            model_used=model_used,
            complexity_score=complexity_score,
            total_latency_ms=total_latency_ms,
            stage_latencies=stage_latencies or {},
            confidence=confidence,
            quality_signal=quality,
            output_tokens=output_tokens,
            was_early_exit=was_early_exit,
            used_cache=used_cache,
            used_rag=used_rag,
            could_have_used_smaller=could_downgrade,
        )

        self._reports.append(report)

        # Record in routing ledger
        self.routing_ledger.record(RoutingOutcome(
            intent=intent,
            complexity_score=complexity_score,
            model_target=model_target,
            quality_signal=quality,
            latency_ms=total_latency_ms,
            confidence=confidence,
        ))

        # Periodically adjust heuristics
        self._requests_since_adjustment += 1
        if self._requests_since_adjustment >= self._adjustment_interval:
            adjustments = self.heuristic_adjuster.adjust_from_ledger(
                self.routing_ledger
            )
            self._total_adjustments_applied += len(adjustments)
            self._requests_since_adjustment = 0

        return report

    def get_complexity_threshold(self, intent: str) -> float:
        """Get the current complexity threshold for routing decisions."""
        return self.heuristic_adjuster.get_threshold(intent)

    def force_adjustment(self) -> list[dict[str, Any]]:
        """Force an immediate heuristic adjustment cycle."""
        adjustments = self.heuristic_adjuster.adjust_from_ledger(
            self.routing_ledger
        )
        self._total_adjustments_applied += len(adjustments)
        self._requests_since_adjustment = 0
        return adjustments

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive self-evaluation statistics."""
        recent = list(self._reports)[-100:]
        quality_dist: dict[str, int] = defaultdict(int)
        for r in recent:
            quality_dist[r.quality_signal.value] += 1

        total_recent = len(recent) or 1
        avg_latency = sum(r.total_latency_ms for r in recent) / total_recent
        avg_confidence = sum(r.confidence for r in recent) / total_recent
        early_exit_rate = sum(1 for r in recent if r.was_early_exit) / total_recent
        cache_hit_rate = sum(1 for r in recent if r.used_cache) / total_recent

        return {
            "total_evaluations": len(self._reports),
            "quality_distribution": dict(quality_dist),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_confidence": round(avg_confidence, 3),
            "early_exit_rate": round(early_exit_rate, 3),
            "cache_hit_rate": round(cache_hit_rate, 3),
            "total_adjustments": self._total_adjustments_applied,
            "routing_summary": self.routing_ledger.get_summary(),
            "heuristic_thresholds": self.heuristic_adjuster.get_stats(),
        }
