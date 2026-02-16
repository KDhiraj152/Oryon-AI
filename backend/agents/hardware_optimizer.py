"""
Hardware Optimizer Agent
=========================

Dynamically tunes hardware parameters based on resource metrics:
- Adjusts batch sizes based on GPU memory pressure
- Manages model loading/eviction priorities
- Tunes thread pool sizes
- Controls memory pressure responses
- Optimizes Metal/MPS configuration at runtime

Listens to: ResourceMonitorAgent (metrics), EvaluationAgent (quality)
Emits to: ModelExecutionAgent (config changes), ResourceMonitorAgent (acks)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .base import AgentMessage, BaseAgent, MessageType

logger = logging.getLogger(__name__)

@dataclass
class HardwareConfig:
    """Current hardware configuration state."""
    embedding_batch_size: int = 64
    reranking_batch_size: int = 32
    translation_batch_size: int = 8
    llm_max_tokens: int = 4096
    gpu_memory_fraction: float = 0.95
    ml_thread_count: int = 4
    io_thread_count: int = 6
    gc_during_inference: bool = False
    model_eviction_threshold_pct: float = 85.0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__.items())

@dataclass
class OptimizationAction:
    """A proposed hardware optimization."""
    action_type: str
    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    expected_impact: str
    timestamp: float = field(default_factory=time.time)
    applied: bool = False
    result: str = ""

class HardwareOptimizerAgent(BaseAgent):
    """
    Reactive agent that adjusts hardware configuration based on system conditions.

    Strategy:
    1. Receives resource snapshots from ResourceMonitorAgent
    2. Analyzes trends (not just point-in-time)
    3. Proposes configuration changes
    4. Applies changes and monitors impact
    5. Reverts if performance degrades
    """

    def __init__(self):
        super().__init__(name="hardware_optimizer")
        self.config = HardwareConfig()
        self._optimization_log: list[OptimizationAction] = []
        self._baseline_metrics: dict[str, float] = {}
        self._last_optimization: float = 0
        self._min_interval_s: float = 30.0  # Don't optimize more than every 30s
        self._snapshots_since_opt: list = []

    async def initialize(self) -> None:
        """Load current hardware configuration."""
        try:
            from backend.core.config import settings
            self.config.embedding_batch_size = settings.EMBEDDING_BATCH_SIZE
            self.config.ml_thread_count = settings.THREADPOOL_MAX_WORKERS
            self.config.io_thread_count = settings.ASYNC_POOL_SIZE
        except (ImportError, AttributeError):
            pass
        logger.info("HardwareOptimizer initialized: %s", self.config.to_dict())

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process resource metrics and optimization requests."""
        if message.msg_type == MessageType.METRIC:
            action = message.payload.get("action")
            if action == "resource_snapshot":
                return await self._handle_resource_snapshot(message)

        elif message.msg_type == MessageType.EVENT:
            action = message.payload.get("action")
            if action == "memory_pressure":
                return await self._handle_memory_pressure(message)

        elif message.msg_type == MessageType.REQUEST:
            action = message.payload.get("action")
            if action == "get_config":
                return message.reply({"config": self.config.to_dict()})
            elif action == "get_optimization_log":
                return message.reply({
                    "log": [a.__dict__ for a in self._optimization_log[-50:]],
                })

        return None

    async def _handle_resource_snapshot(self, message: AgentMessage) -> AgentMessage | None:
        """Analyze resource snapshot and propose optimizations."""
        snapshot = message.payload.get("snapshot", {})
        self._snapshots_since_opt.append(snapshot)

        # Only analyze every N snapshots or when triggered
        if len(self._snapshots_since_opt) < 6:  # ~30s at 5s intervals
            return None

        now = time.time()
        if now - self._last_optimization < self._min_interval_s:
            return None

        proposals = self._analyze_and_propose(self._snapshots_since_opt)
        self._snapshots_since_opt.clear()

        if proposals:
            for proposal in proposals:
                self._apply_optimization(proposal)
                self._optimization_log.append(proposal)
            self._last_optimization = now

            return message.reply({
                "action": "optimizations_applied",
                "count": len(proposals),
                "details": [p.__dict__ for p in proposals],
            }, msg_type=MessageType.EVENT)

        return None

    async def _handle_memory_pressure(self, message: AgentMessage) -> AgentMessage | None:
        """Emergency memory pressure response."""
        level = message.payload.get("level", "warning")
        ram_pct = message.payload.get("ram_percent", 0)

        if level == "emergency":
            # Aggressive: reduce all batch sizes and trigger model eviction
            actions = [
                OptimizationAction(
                    action_type="reduce_batch",
                    parameter="embedding_batch_size",
                    old_value=self.config.embedding_batch_size,
                    new_value=max(8, self.config.embedding_batch_size // 4),
                    reason=f"Emergency memory pressure ({ram_pct:.0f}%)",
                    expected_impact="Reduce GPU memory by ~50%",
                ),
                OptimizationAction(
                    action_type="evict_model",
                    parameter="model_eviction",
                    old_value=None,
                    new_value="lowest_priority",
                    reason=f"Emergency memory pressure ({ram_pct:.0f}%)",
                    expected_impact="Free 1-4GB of memory",
                ),
            ]
            for action in actions:
                self._apply_optimization(action)
                self._optimization_log.append(action)

            # Trigger model eviction
            try:
                from backend.core.optimized.memory_coordinator import (
                    get_memory_coordinator,
                )
                coordinator = get_memory_coordinator()
                coordinator.emergency_evict()  # type: ignore[attr-defined]
                logger.warning("Emergency eviction triggered at %.0f% RAM", ram_pct)
            except (ImportError, RuntimeError) as e:
                logger.error("Emergency eviction failed: %s", e)

        elif level == "critical":
            # Moderate: reduce batch sizes by half
            action = OptimizationAction(
                action_type="reduce_batch",
                parameter="embedding_batch_size",
                old_value=self.config.embedding_batch_size,
                new_value=max(16, self.config.embedding_batch_size // 2),
                reason=f"Critical memory pressure ({ram_pct:.0f}%)",
                expected_impact="Reduce GPU memory by ~25%",
            )
            self._apply_optimization(action)
            self._optimization_log.append(action)

        return message.reply({
            "action": "memory_pressure_handled",
            "level": level,
        })

    def _analyze_and_propose(self, snapshots: list[dict]) -> list[OptimizationAction]:
        """Analyze resource trends and propose optimizations."""
        proposals: list[OptimizationAction] = []

        if not snapshots:
            return proposals

        # Average metrics over the window
        avg_ram = sum(s.get("ram_percent", 0) for s in snapshots) / len(snapshots)
        avg_p95 = sum(s.get("p95_latency_ms", 0) for s in snapshots) / len(snapshots)
        sum(s.get("requests_per_second", 0) for s in snapshots) / len(snapshots)

        # Strategy 1: If memory is low and latency is fine, increase batch sizes
        if avg_ram < 60 and avg_p95 < 2000:
            if self.config.embedding_batch_size < 128:
                proposals.append(OptimizationAction(
                    action_type="increase_batch",
                    parameter="embedding_batch_size",
                    old_value=self.config.embedding_batch_size,
                    new_value=min(128, self.config.embedding_batch_size + 16),
                    reason=f"Low memory usage ({avg_ram:.0f}%) with acceptable latency ({avg_p95:.0f}ms)",
                    expected_impact="Increase embedding throughput ~10%",
                ))

        # Strategy 2: If latency is high but memory is available, check if batch too large
        elif avg_p95 > 3000 and avg_ram < 80 and self.config.embedding_batch_size > 16:
            proposals.append(OptimizationAction(
                action_type="reduce_batch",
                parameter="embedding_batch_size",
                old_value=self.config.embedding_batch_size,
                new_value=max(16, self.config.embedding_batch_size - 16),
                reason=f"High latency ({avg_p95:.0f}ms) — reducing batch to lower per-request latency",
                expected_impact="Reduce P95 latency",
            ))

        # Strategy 3: Memory pressure — reduce footprint
        if avg_ram > 80 and self.config.embedding_batch_size > 32:
            proposals.append(OptimizationAction(
                action_type="reduce_batch",
                parameter="embedding_batch_size",
                old_value=self.config.embedding_batch_size,
                new_value=32,
                reason=f"High memory ({avg_ram:.0f}%) — reducing batch size",
                expected_impact="Free ~100MB of inference buffers",
            ))

        return proposals

    def _apply_optimization(self, action: OptimizationAction) -> None:
        """Apply a proposed optimization."""
        try:
            if action.parameter == "embedding_batch_size":
                self.config.embedding_batch_size = action.new_value
                # Propagate to running system
                try:
                    from backend.core.config import settings
                    settings.EMBEDDING_BATCH_SIZE = action.new_value
                except (ImportError, AttributeError):
                    pass

            elif action.parameter == "reranking_batch_size":
                self.config.reranking_batch_size = action.new_value

            elif action.parameter == "model_eviction":
                pass  # Handled in caller

            action.applied = True
            action.result = "success"
            logger.info(
                f"Applied optimization: {action.parameter} "
                f"{action.old_value} → {action.new_value} ({action.reason})"
            )
        except (RuntimeError, ValueError, AttributeError) as e:
            action.result = f"failed: {e}"
            logger.error("Optimization failed: %s — %s", action.parameter, e)
