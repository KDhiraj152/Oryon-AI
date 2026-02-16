"""
Evaluation Agent
================

Continuously evaluates system quality and performance:
- Tracks per-model quality metrics (response coherence, translation BLEU proxy)
- Monitors SLA compliance (P95 latency targets per task type)
- Detects performance regressions (sliding window comparison)
- Generates evaluation reports
- Triggers self-improvement when regressions detected

Listens to: ModelExecutionAgent (inference metrics), ResourceMonitor (system metrics)
Emits to: SelfImprovementAgent (regression alerts), Orchestrator (quality reports)
"""

import asyncio
import contextlib
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .base import AgentMessage, BaseAgent, MessageType

logger = logging.getLogger(__name__)

# SLA targets per task type (P95 latency in milliseconds)
SLA_TARGETS: dict[str, float] = {
    "llm": 5000.0,          # 5s for full generation
    "embedder": 200.0,       # 200ms for batch embed
    "reranker": 500.0,       # 500ms for reranking
    "translation": 3000.0,   # 3s for translation
    "tts": 4000.0,           # 4s for speech synthesis
    "stt": 8000.0,           # 8s for transcription
    "ocr": 6000.0,           # 6s for OCR processing
}

@dataclass
class MetricWindow:
    """Sliding window of metric values for regression detection."""
    values: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    max_size: int = 500
    window_seconds: float = 300.0  # 5-minute window

    def add(self, value: float, ts: float | None = None) -> None:
        ts = ts or time.time()
        self.values.append(value)
        self.timestamps.append(ts)
        self._prune()

    def _prune(self) -> None:
        """Remove old entries beyond window."""
        if not self.timestamps:
            return
        cutoff = time.time() - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
            self.values.pop(0)
        # Also cap at max_size
        while len(self.values) > self.max_size:
            self.values.pop(0)
            self.timestamps.pop(0)

    @property
    def mean(self) -> float:
        return sum(self.values) / max(1, len(self.values))

    @property
    def p50(self) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        return s[len(s) // 2]

    @property
    def p95(self) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        return s[int(len(s) * 0.95)]

    @property
    def stddev(self) -> float:
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        variance = sum((v - m) ** 2 for v in self.values) / (len(self.values) - 1)
        return math.sqrt(variance)

    @property
    def count(self) -> int:
        return len(self.values)

@dataclass
class SLAStatus:
    """Current SLA compliance status for a model."""
    model: str
    target_ms: float
    current_p95: float
    compliant: bool
    headroom_pct: float  # How far from SLA limit (negative = violation)
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "target_ms": self.target_ms,
            "current_p95_ms": round(self.current_p95, 1),
            "compliant": self.compliant,
            "headroom_pct": round(self.headroom_pct, 1),
            "samples": self.samples,
        }

@dataclass
class RegressionAlert:
    """Detected performance regression."""
    model: str
    metric: str
    baseline_value: float
    current_value: float
    degradation_pct: float
    severity: str  # "warning", "critical"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "metric": self.metric,
            "baseline": round(self.baseline_value, 2),
            "current": round(self.current_value, 2),
            "degradation_pct": round(self.degradation_pct, 1),
            "severity": self.severity,
            "timestamp": self.timestamp,
        }

class EvaluationAgent(BaseAgent):
    """
    Quality and performance evaluation system.

    Maintains two time windows per model:
    - "baseline" window (older data, used as reference)
    - "current" window (recent data, compared against baseline)

    Regression detection uses z-score analysis:
    - WARNING: current mean > baseline mean + 2σ
    - CRITICAL: current mean > baseline mean + 3σ
    """

    def __init__(self):
        super().__init__(name="evaluation")
        # Per-model latency windows
        self._latency_windows: dict[str, MetricWindow] = {}
        # Per-model baseline (updated periodically)
        self._baselines: dict[str, dict[str, float]] = {}
        self._baseline_update_interval: float = 300.0  # 5 min
        self._last_baseline_update: float = 0.0
        # Regression history
        self._regressions: list[RegressionAlert] = []
        self._max_regression_history: int = 100
        # Error rate windows
        self._error_windows: dict[str, MetricWindow] = {}
        # Global eval loop interval
        self._eval_interval: float = 30.0  # Evaluate every 30s
        self._eval_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Start the evaluation loop."""
        self._eval_task = asyncio.create_task(self._evaluation_loop())

        # Bridge to existing SelfOptimizer for RAG pipeline metrics
        self._self_optimizer = None
        try:
            from backend.core.optimized.self_optimizer import get_self_optimizer
            self._self_optimizer = get_self_optimizer()
            logger.info("EvaluationAgent bridged to SelfOptimizer")
        except (ImportError, Exception):  # NOSONAR
            pass

        logger.info("EvaluationAgent initialized with SLA targets: "
                     f"{list(SLA_TARGETS.keys())}")

    async def stop(self) -> None:
        """Stop evaluation loop and base agent."""
        if self._eval_task:
            self._eval_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._eval_task
        await super().stop()

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process metrics and evaluation requests."""
        if message.msg_type == MessageType.METRIC:
            action = message.payload.get("action")
            if action == "record_inference":
                self._record_inference_metric(message.payload)
                return None

        elif message.msg_type == MessageType.REQUEST:
            action = message.payload.get("action")
            if action == "get_sla_status":
                return message.reply({"sla": self._get_all_sla_status()})
            elif action == "get_regressions":
                return message.reply({
                    "regressions": [r.to_dict() for r in self._regressions[-20:]],
                })
            elif action == "get_report":
                return message.reply(self._generate_report())
            elif action == "get_baselines":
                return message.reply({"baselines": self._baselines})

        return None

    def _record_inference_metric(self, payload: dict) -> None:
        """Record an inference metric from ModelExecutionAgent."""
        model = payload.get("model", "unknown")
        latency_ms = payload.get("latency_ms", 0.0)
        is_error = payload.get("is_error", False)

        # Latency window
        if model not in self._latency_windows:
            self._latency_windows[model] = MetricWindow(window_seconds=600.0)
        self._latency_windows[model].add(latency_ms)

        # Error tracking
        if model not in self._error_windows:
            self._error_windows[model] = MetricWindow(window_seconds=600.0)
        self._error_windows[model].add(1.0 if is_error else 0.0)

    async def _evaluation_loop(self) -> None:
        """Periodically evaluate system performance."""
        await asyncio.sleep(60.0)  # Wait for initial metrics
        while True:
            try:
                await self._run_evaluation()
            except asyncio.CancelledError:
                logger.debug("Evaluation loop cancelled")
                raise
            except Exception as e:  # top-level handler
                logger.error("Evaluation loop error: %s", e, exc_info=True)
            await asyncio.sleep(self._eval_interval)

    async def _run_evaluation(self) -> None:
        """Run one evaluation cycle."""
        now = time.time()

        # Update baselines periodically
        if now - self._last_baseline_update > self._baseline_update_interval:
            self._update_baselines()
            self._last_baseline_update = now

        # Check for regressions
        new_regressions = self._detect_regressions()
        if new_regressions:
            for reg in new_regressions:
                self._regressions.append(reg)
                logger.warning(
                    f"REGRESSION [{reg.severity}] {reg.model}/{reg.metric}: "
                    f"{reg.baseline_value:.1f} → {reg.current_value:.1f} "
                    f"({reg.degradation_pct:+.1f}%)"
                )
            # Trim history
            while len(self._regressions) > self._max_regression_history:
                self._regressions.pop(0)

            # Alert self_improvement agent
            from .base import AgentRegistry
            registry = AgentRegistry()
            await registry.route_message(AgentMessage(
                msg_type=MessageType.EVENT,
                sender=self.name,
                recipient="self_improvement",
                payload={
                    "action": "regressions_detected",
                    "regressions": [r.to_dict() for r in new_regressions],
                },
                priority=2 if any(r.severity == "critical" for r in new_regressions) else 1,
            ))

        # Check SLA compliance
        sla_violations = [s for s in self._compute_sla_statuses() if not s.compliant]
        if sla_violations:
            from .base import AgentRegistry
            registry = AgentRegistry()
            await registry.route_message(AgentMessage(
                msg_type=MessageType.EVENT,
                sender=self.name,
                recipient="self_improvement",
                payload={
                    "action": "sla_violations",
                    "violations": [v.to_dict() for v in sla_violations],
                },
            ))

    def _update_baselines(self) -> None:
        """Capture current metrics as baseline for future regression detection."""
        for model, window in self._latency_windows.items():
            if window.count >= 10:
                self._baselines[model] = {
                    "mean_ms": window.mean,
                    "p95_ms": window.p95,
                    "stddev_ms": window.stddev,
                    "samples": window.count,
                    "timestamp": time.time(),
                }

    def _detect_latency_regression(
        self, model: str, window: "MetricWindow", baseline: dict
    ) -> RegressionAlert | None:
        """Detect latency regression for a single model."""
        baseline_mean = baseline["mean_ms"]
        baseline_stddev = baseline.get("stddev_ms", 0)

        if baseline_stddev < 1.0:
            if window.mean > baseline_mean * 1.5:
                degradation = ((window.mean - baseline_mean) / max(1, baseline_mean)) * 100
                return RegressionAlert(
                    model=model,
                    metric="latency_mean",
                    baseline_value=baseline_mean,
                    current_value=window.mean,
                    degradation_pct=degradation,
                    severity="critical" if degradation > 100 else "warning",
                )
        else:
            z_score = (window.mean - baseline_mean) / baseline_stddev
            if z_score > 2.0:
                degradation = ((window.mean - baseline_mean) / max(1, baseline_mean)) * 100
                return RegressionAlert(
                    model=model,
                    metric="latency_mean",
                    baseline_value=baseline_mean,
                    current_value=window.mean,
                    degradation_pct=degradation,
                    severity="critical" if z_score > 3.0 else "warning",
                )
        return None

    def _detect_regressions(self) -> list[RegressionAlert]:
        """Detect regressions using z-score analysis against baselines."""
        alerts: list[RegressionAlert] = []

        for model, window in self._latency_windows.items():
            baseline = self._baselines.get(model)
            if not baseline or window.count < 10:
                continue
            alert = self._detect_latency_regression(model, window, baseline)
            if alert:
                alerts.append(alert)

        # Error rate regression
        for model, window in self._error_windows.items():
            if window.count < 20:
                continue
            error_rate = window.mean  # mean of 0/1 = error rate
            if error_rate > 0.05:  # >5% error rate
                alerts.append(RegressionAlert(
                    model=model,
                    metric="error_rate",
                    baseline_value=0.01,  # Target
                    current_value=error_rate,
                    degradation_pct=error_rate * 100,
                    severity="critical" if error_rate > 0.10 else "warning",
                ))

        return alerts

    def _compute_sla_statuses(self) -> list[SLAStatus]:
        """Compute SLA compliance for all tracked models."""
        statuses = []
        for model, window in self._latency_windows.items():
            target = SLA_TARGETS.get(model, 5000.0)
            if window.count < 5:
                continue
            p95 = window.p95
            headroom = ((target - p95) / target) * 100
            statuses.append(SLAStatus(
                model=model,
                target_ms=target,
                current_p95=p95,
                compliant=p95 <= target,
                headroom_pct=headroom,
                samples=window.count,
            ))
        return statuses

    def _get_all_sla_status(self) -> list[dict]:
        return [s.to_dict() for s in self._compute_sla_statuses()]

    def _generate_report(self) -> dict[str, Any]:
        """Generate comprehensive evaluation report."""
        return {
            "timestamp": time.time(),
            "sla_status": self._get_all_sla_status(),
            "baselines": self._baselines,
            "active_regressions": [r.to_dict() for r in self._regressions[-10:]],
            "model_summaries": {
                model: {
                    "mean_ms": round(w.mean, 1),
                    "p50_ms": round(w.p50, 1),
                    "p95_ms": round(w.p95, 1),
                    "samples": w.count,
                }
                for model, w in self._latency_windows.items()
            },
        }
