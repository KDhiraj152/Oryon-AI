"""
Monitoring Module (Principles S, T)
====================================
Prometheus metrics and OOM alerting for system monitoring.
"""

import logging
import os
from datetime import UTC, datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


def check_system_health() -> dict[str, Any]:
    """
    Check overall system health including database, Redis, and services.

    Returns a detailed health status dictionary.
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "components": {},
    }

    # Check database
    try:
        from sqlalchemy import text

        from ..database import SessionLocal

        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        health["components"]["database"] = {"status": "connected"}
    except Exception as e:
        health["components"]["database"] = {"status": "disconnected", "error": str(e)}
        health["status"] = "degraded"

    # Check Redis
    try:
        from ..cache import get_redis

        redis_client = get_redis()
        if redis_client and redis_client.ping():
            health["components"]["redis"] = {"status": "connected"}
        else:
            health["components"]["redis"] = {"status": "disconnected"}
            health["status"] = "degraded"
    except Exception as e:
        health["components"]["redis"] = {"status": "disconnected", "error": str(e)}
        health["status"] = "degraded"

    # Check memory
    try:
        memory_status = get_memory_status()
        health["components"]["memory"] = memory_status
        if memory_status.get("percent_used", 0) > 90:
            health["status"] = "degraded"
    except Exception:
        health["components"]["memory"] = {"status": "unknown"}

    return health


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger."""
    return logging.getLogger(name)


def track_review_action(action: str, review_id: Any, user_id: Any) -> None:
    """
    Track a review action for monitoring/analytics.

    Args:
        action: The action type (e.g., "review_created", "comment_added")
        review_id: The review ID
        user_id: The user who performed the action
    """
    logger.info(f"Review action: {action} | review={review_id} | user={user_id}")


def track_websocket_connection(action: str, connection_id: str | None = None) -> None:
    """
    Track WebSocket connection events.

    Args:
        action: The action type (e.g., "connected", "disconnected", "error")
        connection_id: Optional connection identifier
    """
    logger.info(f"WebSocket: {action} | connection={connection_id or 'unknown'}")


def track_translation_latency(language: str, latency_ms: float) -> None:
    """
    Track translation latency for a specific language.

    Args:
        language: Target language of the translation
        latency_ms: Latency in milliseconds
    """
    logger.info(f"Translation latency: {language} | {latency_ms:.2f}ms")


from .metrics import (
    LatencyTracker,
    get_latency_stats,
    # Metrics endpoint
    get_metrics,
    get_metrics_content_type,
    # Latency tracking
    log_latency,
    record_cache_hit,
    record_cache_miss,
    record_error,
    record_gpu_memory,
    # Recording functions
    record_inference_latency,
    record_model_load_time,
    record_model_memory,
    record_queue_length,
    record_queue_wait,
    record_system_memory,
    record_tokens,
    setup_metrics_endpoint,
    # Context managers and decorators
    track_inference,
    track_latency,
)
from .oom_alerts import (
    # Alert types
    Alert,
    # Alert handlers
    AlertHandler,
    AlertLevel,
    AlertType,
    EvictionAlertHandler,
    LogAlertHandler,
    OOMAlertConfig,
    # Alert manager
    OOMAlertManager,
    SlackAlertHandler,
    WebhookAlertHandler,
    get_alert_manager,
    get_memory_status,
    # Convenience functions
    start_oom_monitoring,
    stop_oom_monitoring,
)

__all__ = [
    # OOM Alerts (Principle T)
    "Alert",
    "AlertHandler",
    "AlertLevel",
    "AlertType",
    "EvictionAlertHandler",
    "LatencyTracker",
    "LogAlertHandler",
    "OOMAlertConfig",
    "OOMAlertManager",
    "SlackAlertHandler",
    "WebhookAlertHandler",
    # System health
    "check_system_health",
    "get_alert_manager",
    "get_latency_stats",
    "get_logger",
    "get_memory_status",
    "get_metrics",
    "get_metrics_content_type",
    "log_latency",
    "record_cache_hit",
    "record_cache_miss",
    "record_error",
    "record_gpu_memory",
    # Metrics (Principle S)
    "record_inference_latency",
    "record_model_load_time",
    "record_model_memory",
    "record_queue_length",
    "record_queue_wait",
    "record_system_memory",
    "record_tokens",
    "setup_metrics_endpoint",
    "start_oom_monitoring",
    "stop_oom_monitoring",
    "track_inference",
    "track_latency",
    "track_review_action",
    "track_translation_latency",
    "track_websocket_connection",
]
