"""
Resource Monitor Agent
=======================

Continuously monitors system resources and emits alerts:
- RAM usage and pressure levels
- GPU memory utilization
- CPU core utilization (P-core vs E-core)
- Latency percentile distributions
- Throughput metrics (tokens/sec, requests/sec)

Emits METRIC messages to all subscribers at configurable intervals.
"""

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import psutil

from .base import AgentMessage, AgentStatus, BaseAgent, MessageType

logger = logging.getLogger(__name__)

@dataclass
class ResourceSnapshot:
    """Point-in-time resource measurement."""
    timestamp: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    cpu_percent: float
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    active_models: int = 0
    request_queue_depth: int = 0
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    @property
    def memory_pressure(self) -> str:
        if self.ram_percent > 95:
            return "emergency"
        elif self.ram_percent > 85:
            return "critical"
        elif self.ram_percent > 70:
            return "warning"
        return "normal"

class ResourceMonitorAgent(BaseAgent):
    """
    Agent that continuously monitors system resources.

    Emits periodic METRIC messages with resource snapshots.
    Triggers ALERT events when thresholds are breached.
    Maintains a sliding window of historical metrics for trend analysis.
    """

    def __init__(
        self,
        monitor_interval_s: float = 5.0,
        history_window: int = 720,  # 1 hour at 5s intervals
        alert_cooldown_s: float = 60.0,
    ):
        super().__init__(name="resource_monitor")
        self.monitor_interval_s = monitor_interval_s
        self.history: deque[ResourceSnapshot] = deque(maxlen=history_window)
        self._latency_window: deque[float] = deque(maxlen=1000)
        self._request_timestamps: deque[float] = deque(maxlen=1000)
        self._token_counts: deque[tuple[float, int]] = deque(maxlen=1000)
        self._alert_cooldowns: dict[str, float] = {}
        self._alert_cooldown_s = alert_cooldown_s
        self._monitor_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize resource monitoring."""
        logger.info("ResourceMonitorAgent initializing...")

    async def start(self) -> None:
        """Start monitoring loop alongside message processing."""
        await super().start()
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="resource-monitor-loop"
        )

    async def stop(self) -> None:
        """Stop monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
        await super().stop()

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """Handle incoming messages (latency reports, token counts, etc.)."""
        if message.msg_type == MessageType.METRIC:
            action = message.payload.get("action")
            if action == "record_latency":
                self._latency_window.append(message.payload["latency_ms"])
            elif action == "record_request":
                self._request_timestamps.append(time.time())
            elif action == "record_tokens":
                self._token_counts.append(
                    (time.time(), message.payload["token_count"])
                )
            elif action == "get_snapshot":
                snapshot = self._take_snapshot()
                return message.reply({"snapshot": snapshot.__dict__})
        elif message.msg_type == MessageType.REQUEST:
            if message.payload.get("action") == "get_history":
                return message.reply({
                    "history": [s.__dict__ for s in self.history],
                    "count": len(self.history),
                })
        return None

    async def _monitor_loop(self) -> None:
        """Periodic resource monitoring."""
        while self.status == AgentStatus.RUNNING:
            try:
                snapshot = self._take_snapshot()
                self.history.append(snapshot)

                # Broadcast metric
                await self._route_reply(AgentMessage(
                    msg_type=MessageType.METRIC,
                    sender=self.name,
                    recipient="*",
                    payload={
                        "action": "resource_snapshot",
                        "snapshot": snapshot.__dict__,
                    },
                ))

                # Check thresholds and emit alerts
                await self._check_thresholds(snapshot)

                await asyncio.sleep(self.monitor_interval_s)

            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError) as e:
                logger.error("Monitor loop error: %s", e)
                await asyncio.sleep(self.monitor_interval_s)

    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a point-in-time resource snapshot."""
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)  # Non-blocking

        # GPU memory (Apple Silicon unified memory)
        gpu_used_mb = 0.0
        gpu_total_mb = 0.0
        gpu_util = 0.0
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # On Apple Silicon, GPU memory = unified memory
                gpu_total_mb = mem.total / (1024 * 1024)
                gpu_used_mb = mem.used / (1024 * 1024)
                gpu_util = mem.percent
            elif torch.cuda.is_available():
                gpu_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                gpu_util = (gpu_used_mb / gpu_total_mb * 100) if gpu_total_mb > 0 else 0
        except (ImportError, RuntimeError):
            pass

        # Active model count
        active_models = 0
        try:
            from backend.core.optimized.memory_coordinator import get_memory_coordinator
            coord = get_memory_coordinator()
            active_models = len(coord._loaded_models)
        except (ImportError, RuntimeError):
            pass

        # Latency percentiles
        latencies = sorted(self._latency_window)
        p50 = latencies[len(latencies) // 2] if latencies else 0
        p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else 0
        p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else 0

        # Throughput
        now = time.time()
        recent_requests = sum(1 for t in self._request_timestamps if now - t < 60)
        rps = recent_requests / 60.0

        recent_tokens = sum(count for t, count in self._token_counts if now - t < 60)
        tps = recent_tokens / 60.0

        return ResourceSnapshot(
            timestamp=now,
            ram_used_gb=mem.used / (1024**3),
            ram_total_gb=mem.total / (1024**3),
            ram_percent=mem.percent,
            cpu_percent=cpu,
            gpu_memory_used_mb=gpu_used_mb,
            gpu_memory_total_mb=gpu_total_mb,
            gpu_utilization_percent=gpu_util,
            active_models=active_models,
            tokens_per_second=tps,
            requests_per_second=rps,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
        )

    async def _check_thresholds(self, snapshot: ResourceSnapshot) -> None:
        """Check resource thresholds and emit alerts."""
        time.time()

        if (snapshot.memory_pressure in ("critical", "emergency")
                and self._should_alert("memory_pressure")):
            await self._route_reply(AgentMessage(
                msg_type=MessageType.EVENT,
                sender=self.name,
                recipient="hardware_optimizer",
                payload={
                    "action": "memory_pressure",
                    "level": snapshot.memory_pressure,
                    "ram_percent": snapshot.ram_percent,
                    "ram_used_gb": snapshot.ram_used_gb,
                },
                priority=10,
            ))

        if snapshot.p95_latency_ms > 5000 and self._should_alert("high_latency"):  # > 5s P95
            await self._route_reply(AgentMessage(
                msg_type=MessageType.EVENT,
                sender=self.name,
                recipient="self_improvement",
                payload={
                    "action": "high_latency",
                    "p95_ms": snapshot.p95_latency_ms,
                    "p99_ms": snapshot.p99_latency_ms,
                },
            ))

    def _should_alert(self, alert_type: str) -> bool:
        """Check if alert should fire (respects cooldown)."""
        now = time.time()
        last = self._alert_cooldowns.get(alert_type, 0)
        if now - last >= self._alert_cooldown_s:
            self._alert_cooldowns[alert_type] = now
            return True
        return False

    def record_latency(self, latency_ms: float) -> None:
        """Direct API for recording latency (avoids message overhead)."""
        self._latency_window.append(latency_ms)

    def record_request(self) -> None:
        """Direct API for recording a request."""
        self._request_timestamps.append(time.time())

    def record_tokens(self, count: int) -> None:
        """Direct API for recording generated tokens."""
        self._token_counts.append((time.time(), count))
