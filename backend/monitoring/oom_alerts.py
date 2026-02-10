"""
OOM Alerting System (Principle T)
==================================
Alert on memory threshold breach to prevent crashes.

Strategy:
- Monitor system RAM usage
- Monitor GPU memory usage
- Trigger alerts at configurable thresholds
- Automatic model eviction when critical

Reference: "Alert on RAM > 85%"
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts."""

    RAM_HIGH = "ram_high"
    RAM_CRITICAL = "ram_critical"
    GPU_HIGH = "gpu_high"
    GPU_CRITICAL = "gpu_critical"
    OOM_IMMINENT = "oom_imminent"
    MODEL_EVICTED = "model_evicted"
    QUEUE_BACKLOG = "queue_backlog"


@dataclass
class Alert:
    """An alert instance."""

    alert_type: AlertType
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OOMAlertConfig:
    """Configuration for OOM alerting (Principle T)."""

    # RAM thresholds (percentage)
    ram_warning_threshold: float = 75.0
    ram_critical_threshold: float = 85.0  # Principle T: Alert on RAM > 85%
    ram_emergency_threshold: float = 95.0

    # GPU thresholds (percentage)
    gpu_warning_threshold: float = 80.0
    gpu_critical_threshold: float = 90.0
    gpu_emergency_threshold: float = 95.0

    # Monitoring interval
    check_interval_seconds: float = 5.0

    # Alert cooldown (don't spam same alert)
    alert_cooldown_seconds: float = 60.0

    # Auto-eviction on critical
    auto_evict_on_critical: bool = True

    # Webhook for alerts
    alert_webhook_url: str | None = None

    # Slack webhook
    slack_webhook_url: str | None = None


class AlertHandler:
    """Base class for alert handlers."""

    async def handle(self, alert: Alert) -> bool:
        """Handle an alert. Returns True if handled successfully."""
        raise NotImplementedError


class LogAlertHandler(AlertHandler):
    """Log alerts to logger."""

    async def handle(self, alert: Alert) -> bool:
        level_map = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
            AlertLevel.EMERGENCY: logger.critical,
        }

        log_fn = level_map.get(alert.level, logger.warning)
        log_fn(
            f"[{alert.level.value.upper()}] {alert.alert_type.value}: "
            f"{alert.message} (value={alert.value:.1f}%, threshold={alert.threshold:.1f}%)"
        )
        return True


class WebhookAlertHandler(AlertHandler):
    """Send alerts to webhook."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def handle(self, alert: Alert) -> bool:
        try:
            import aiohttp

            payload = {
                "alert_type": alert.alert_type.value,
                "level": alert.level.value,
                "message": alert.message,
                "value": alert.value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp,
                "metadata": alert.metadata,
            }

            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    return resp.status == 200

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class SlackAlertHandler(AlertHandler):
    """Send alerts to Slack."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def handle(self, alert: Alert) -> bool:
        try:
            import aiohttp

            color_map = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ffcc00",
                AlertLevel.CRITICAL: "#ff6600",
                AlertLevel.EMERGENCY: "#ff0000",
            }

            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.level, "#808080"),
                        "title": f"🚨 {alert.alert_type.value.upper()}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Value",
                                "value": f"{alert.value:.1f}%",
                                "short": True,
                            },
                            {
                                "title": "Threshold",
                                "value": f"{alert.threshold:.1f}%",
                                "short": True,
                            },
                        ],
                        "footer": "Shiksha Setu Monitoring",
                        "ts": int(alert.timestamp),
                    }
                ]
            }

            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    return resp.status == 200

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class EvictionAlertHandler(AlertHandler):
    """Trigger model eviction on critical alerts."""

    def __init__(self, lifecycle_manager=None):
        self.lifecycle_manager = lifecycle_manager

    async def handle(self, alert: Alert) -> bool:
        if alert.level not in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            return True

        if alert.alert_type not in [
            AlertType.RAM_CRITICAL,
            AlertType.GPU_CRITICAL,
            AlertType.OOM_IMMINENT,
        ]:
            return True

        logger.warning(f"Triggering model eviction due to {alert.alert_type.value}")

        try:
            if self.lifecycle_manager:
                await self.lifecycle_manager.evict_idle_models()
            else:
                # Try to import and use optimized model manager
                from backend.core.optimized import get_model_manager

                manager = get_model_manager()
                await manager.unload_idle_models()

            return True
        except Exception as e:
            logger.error(f"Failed to evict models: {e}")
            return False


class OOMAlertManager:
    """
    Manages OOM monitoring and alerting (Principle T).

    Features:
    - Monitors system RAM and GPU memory
    - Triggers alerts at configurable thresholds
    - Supports multiple alert handlers (log, webhook, Slack)
    - Auto-evicts models when memory is critical
    """

    def __init__(self, config: OOMAlertConfig | None = None):
        self.config = config or OOMAlertConfig()
        self.handlers: list[AlertHandler] = []
        self._alert_history: dict[str, float] = {}  # alert_type -> last_alert_time
        self._running = False
        self._task: asyncio.Task | None = None

        # Always add log handler
        self.handlers.append(LogAlertHandler())

        # Add webhook handler if configured
        if self.config.alert_webhook_url:
            self.handlers.append(WebhookAlertHandler(self.config.alert_webhook_url))

        # Add Slack handler if configured
        if self.config.slack_webhook_url:
            self.handlers.append(SlackAlertHandler(self.config.slack_webhook_url))

        # Add eviction handler if auto-evict enabled
        if self.config.auto_evict_on_critical:
            self.handlers.append(EvictionAlertHandler())

    def add_handler(self, handler: AlertHandler):
        """Add an alert handler."""
        self.handlers.append(handler)

    async def start(self):
        """Start the monitoring loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("OOM alert manager started")

    async def stop(self):
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("OOM alert manager stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_memory()
            except Exception as e:
                logger.error(f"Error in OOM monitor: {e}")

            await asyncio.sleep(self.config.check_interval_seconds)

    async def _check_memory(self):
        """Check memory usage and trigger alerts."""
        # Check RAM
        ram_percent = self._get_ram_usage()
        await self._check_ram_thresholds(ram_percent)

        # Check GPU
        gpu_percent = self._get_gpu_usage()
        if gpu_percent is not None:
            await self._check_gpu_thresholds(gpu_percent)

    def _get_ram_usage(self) -> float:
        """Get current RAM usage percentage."""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

    def _get_gpu_usage(self) -> float | None:
        """Get current GPU memory usage percentage."""
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return (allocated / total) * 100.0
        except ImportError:
            pass
        return None

    async def _check_ram_thresholds(self, ram_percent: float):
        """Check RAM thresholds and trigger alerts."""
        if ram_percent >= self.config.ram_emergency_threshold:
            await self._trigger_alert(
                Alert(
                    alert_type=AlertType.OOM_IMMINENT,
                    level=AlertLevel.EMERGENCY,
                    message=f"RAM usage at {ram_percent:.1f}% - OOM imminent!",
                    value=ram_percent,
                    threshold=self.config.ram_emergency_threshold,
                )
            )
        elif ram_percent >= self.config.ram_critical_threshold:
            await self._trigger_alert(
                Alert(
                    alert_type=AlertType.RAM_CRITICAL,
                    level=AlertLevel.CRITICAL,
                    message=f"RAM usage at {ram_percent:.1f}% - above critical threshold",
                    value=ram_percent,
                    threshold=self.config.ram_critical_threshold,
                )
            )
        elif ram_percent >= self.config.ram_warning_threshold:
            await self._trigger_alert(
                Alert(
                    alert_type=AlertType.RAM_HIGH,
                    level=AlertLevel.WARNING,
                    message=f"RAM usage at {ram_percent:.1f}% - above warning threshold",
                    value=ram_percent,
                    threshold=self.config.ram_warning_threshold,
                )
            )

    async def _check_gpu_thresholds(self, gpu_percent: float):
        """Check GPU thresholds and trigger alerts."""
        if gpu_percent >= self.config.gpu_emergency_threshold:
            await self._trigger_alert(
                Alert(
                    alert_type=AlertType.OOM_IMMINENT,
                    level=AlertLevel.EMERGENCY,
                    message=f"GPU memory at {gpu_percent:.1f}% - OOM imminent!",
                    value=gpu_percent,
                    threshold=self.config.gpu_emergency_threshold,
                    metadata={"resource": "gpu"},
                )
            )
        elif gpu_percent >= self.config.gpu_critical_threshold:
            await self._trigger_alert(
                Alert(
                    alert_type=AlertType.GPU_CRITICAL,
                    level=AlertLevel.CRITICAL,
                    message=f"GPU memory at {gpu_percent:.1f}% - above critical threshold",
                    value=gpu_percent,
                    threshold=self.config.gpu_critical_threshold,
                )
            )
        elif gpu_percent >= self.config.gpu_warning_threshold:
            await self._trigger_alert(
                Alert(
                    alert_type=AlertType.GPU_HIGH,
                    level=AlertLevel.WARNING,
                    message=f"GPU memory at {gpu_percent:.1f}% - above warning threshold",
                    value=gpu_percent,
                    threshold=self.config.gpu_warning_threshold,
                )
            )

    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert through all handlers."""
        # Check cooldown
        last_alert = self._alert_history.get(alert.alert_type.value, 0)
        if time.time() - last_alert < self.config.alert_cooldown_seconds:
            return

        self._alert_history[alert.alert_type.value] = time.time()

        # Send to all handlers
        for handler in self.handlers:
            try:
                await handler.handle(alert)
            except Exception as e:
                logger.error(f"Alert handler {type(handler).__name__} failed: {e}")

    def get_current_status(self) -> dict[str, Any]:
        """Get current memory status."""
        ram = self._get_ram_usage()
        gpu = self._get_gpu_usage()

        return {
            "ram_percent": ram,
            "ram_status": self._get_status(
                ram,
                self.config.ram_warning_threshold,
                self.config.ram_critical_threshold,
            ),
            "gpu_percent": gpu,
            "gpu_status": self._get_status(
                gpu,
                self.config.gpu_warning_threshold,
                self.config.gpu_critical_threshold,
            )
            if gpu
            else "unavailable",
            "thresholds": {
                "ram_warning": self.config.ram_warning_threshold,
                "ram_critical": self.config.ram_critical_threshold,
                "gpu_warning": self.config.gpu_warning_threshold,
                "gpu_critical": self.config.gpu_critical_threshold,
            },
        }

    def _get_status(self, value: float | None, warning: float, critical: float) -> str:
        """Get status string based on thresholds."""
        if value is None:
            return "unknown"
        if value >= critical:
            return "critical"
        if value >= warning:
            return "warning"
        return "ok"


# Global alert manager instance
_alert_manager: OOMAlertManager | None = None


def get_alert_manager() -> OOMAlertManager:
    """Get or create global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = OOMAlertManager()
    return _alert_manager


async def start_oom_monitoring():
    """Start OOM monitoring."""
    manager = get_alert_manager()
    await manager.start()


async def stop_oom_monitoring():
    """Stop OOM monitoring."""
    manager = get_alert_manager()
    await manager.stop()


def get_memory_status() -> dict[str, Any]:
    """Get current memory status."""
    manager = get_alert_manager()
    return manager.get_current_status()
