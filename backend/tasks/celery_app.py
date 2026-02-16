"""Celery application configuration for async task processing."""

import logging
import os
from datetime import UTC, datetime, timedelta, timezone

from celery import Celery, Task
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
    task_success,
)
from kombu import Queue

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("CELERY_REDIS_DB", 1)  # Use different DB than cache
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Build Redis URL
if REDIS_PASSWORD:
    BROKER_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    RESULT_BACKEND = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
else:
    BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    RESULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Define retriable exceptions - these indicate transient failures
RETRIABLE_EXCEPTIONS = (
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    TimeoutError,
    OSError,  # Covers socket errors
)

# Define non-retriable exceptions - these indicate permanent failures
NON_RETRIABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    PermissionError,
)

class SmartRetryTask(Task):
    """
    Custom Task base class with intelligent retry logic.

    Only retries on transient errors (network, timeout, connection).
    Rejects permanently on validation/logic errors (400-type errors).
    """

    autoretry_for = RETRIABLE_EXCEPTIONS
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    max_retries = 3

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure - distinguish between retriable and permanent failures."""
        if isinstance(exc, NON_RETRIABLE_EXCEPTIONS):
            logger.error(
                f"Task {self.name} [{task_id}] permanently failed (non-retriable): {exc}"
            )
        else:
            logger.error("Task %s [%s] failed: %s", self.name, task_id, exc)
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Log retry attempts."""
        logger.warning(
            f"Task {self.name} [{task_id}] retrying due to: {exc.__class__.__name__}: {exc}"
        )
        super().on_retry(exc, task_id, args, kwargs, einfo)

# Create Celery app with custom task base
celery_app = Celery(
    "oryon",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    task_cls=SmartRetryTask,
    include=[
        "backend.tasks.pipeline_tasks",
        "backend.tasks.qa_tasks",
        "backend.tasks.ocr_tasks",
        "backend.tasks.audio_tasks",
        "backend.tasks.translate_tasks",
        "backend.tasks.simplify_tasks",
        "backend.tasks.embedding_tasks",
        "backend.tasks.rag_tasks",
    ],
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "pipeline.*": {"queue": "pipeline"},
        "backend.tasks.pipeline_tasks.*": {"queue": "pipeline"},
        "backend.tasks.qa_tasks.*": {"queue": "pipeline"},
    },
    # Task queues
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("pipeline", routing_key="pipeline"),
        Queue("ocr", routing_key="ocr"),
        Queue("ml_gpu", routing_key="ml_gpu"),
        Queue("ml_cpu", routing_key="ml_cpu"),
        Queue("dead_letter", routing_key="dead_letter"),  # Dead letter queue
    ),
    # Result settings
    result_expires=3600,  # 1 hour default
    result_extended=True,
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    # Task execution settings
    task_acks_late=True,  # Acknowledge after task completion (prevents data loss)
    task_reject_on_worker_lost=True,  # Requeue tasks if worker dies
    task_track_started=True,  # Track when tasks start
    task_time_limit=1800,  # 30 minutes max
    task_soft_time_limit=1500,  # 25 minutes soft limit
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for GPU workers
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevent memory leaks)
    worker_disable_rate_limits=False,
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Retries - ONLY retry on transient/retriable errors, NOT all exceptions
    # This prevents infinite retries on 400 errors, validation failures, etc.
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    task_autoretry_for=(
        # Network/Connection errors - always retriable
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        TimeoutError,
        OSError,  # Covers socket errors
        # Database errors - often transient
        # Note: Import these at module level if using SQLAlchemy exceptions
    ),
    task_retry_backoff=True,  # Exponential backoff
    task_retry_backoff_max=600,  # Max 10 minutes backoff
    task_retry_jitter=True,  # Add randomness to prevent thundering herd
    # Beat schedule (for periodic tasks)
    beat_schedule={
        "cleanup-old-tasks": {
            "task": "backend.tasks.pipeline_tasks.cleanup_old_results",
            "schedule": 3600.0,  # Every hour
        },
        "cleanup-expired-results": {
            "task": "backend.tasks.celery_app.cleanup_expired_results",
            "schedule": 1800.0,  # Every 30 minutes
        },
    },
)

# Task lifecycle signals
@task_prerun.connect
def task_prerun_handler(
    sender=None, task_id=None, task=None, args=None, kwargs=None, **extra
):
    """Log task start."""
    logger.info("Task %s [%s] started", task.name, task_id)

@task_postrun.connect
def task_postrun_handler(
    sender=None,
    task_id=None,
    task=None,
    args=None,
    kwargs=None,
    retval=None,
    state=None,
    **extra,
):
    """Log task completion."""
    logger.info("Task %s [%s] completed with state: %s", task.name, task_id, state)

@task_success.connect
def task_success_handler(sender=None, result=None, **extra):
    """Handle successful task completion."""
    logger.info("Task %s completed successfully", sender.name)

@task_failure.connect
def task_failure_handler(
    sender=None,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **extra,
):
    """Handle task failure - log and optionally alert.

    Note: Alerting mechanism (email, Slack, etc.) can be implemented by:
    1. Adding an alert service in backend/services/alerts.py
    2. Importing and calling send_alert(f"Task {sender.name} failed: {exception}")
    3. Configure alert channels via environment variables (SLACK_WEBHOOK_URL, ALERT_EMAIL, etc.)
    """
    logger.error("Task %s [%s] failed: %s", sender.name, task_id, exception)
    logger.error("Traceback: %s", traceback)

    # Alerting mechanism can be added here when needed
    # send_alert(f"Task {sender.name} failed: {exception}")

@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **extra):
    """Handle task retry."""
    logger.warning("Task %s [%s] is being retried: %s", sender.name, task_id, reason)

# Utility functions
def get_task_info(task_id: str) -> dict:
    """Get task status and result."""
    result = celery_app.AsyncResult(task_id)

    info = {
        "task_id": task_id,
        "state": result.state,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
        "failed": result.failed() if result.ready() else None,
    }

    if result.ready():
        if result.successful():
            info["result"] = result.result
        elif result.failed():
            info["error"] = str(result.info)
            info["traceback"] = result.traceback
    else:
        # Task in progress - get metadata
        if hasattr(result.info, "get"):
            info["progress"] = result.info.get("progress", 0)
            info["stage"] = result.info.get("stage", "unknown")
            info["message"] = result.info.get("message", "")

    return info

def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """Cancel a running task."""
    try:
        celery_app.control.revoke(task_id, terminate=terminate, signal="SIGKILL")
        logger.info("Task %s revoked (terminate=%s)", task_id, terminate)
        return True
    except (OSError, ConnectionError, TimeoutError) as e:
        logger.error("Failed to revoke task %s: %s", task_id, e)
        return False

# Export
__all__ = ["celery_app", "get_task_info", "revoke_task"]

def _is_task_expired(result_data: bytes, current_time, expiry_threshold) -> bool:
    """Check if a task result is expired and should be deleted."""
    import json

    from dateutil import parser  # type: ignore[import-untyped]

    try:
        result = json.loads(result_data)

        # Only delete completed tasks
        if result.get("status") not in ["SUCCESS", "FAILURE", "REVOKED"]:
            return False

        date_done = result.get("date_done")
        if not date_done:
            return False

        done_time = parser.parse(date_done)
        return bool(current_time - done_time.replace(tzinfo=None) > expiry_threshold)

    except json.JSONDecodeError:
        return False

def _delete_expired_keys(
    redis_client, result_keys: list, current_time, expiry_threshold
) -> int:
    """Delete expired task result keys from Redis."""
    deleted_count = 0

    for key in result_keys:
        try:
            result_data = redis_client.get(key)
            if result_data and _is_task_expired(
                result_data, current_time, expiry_threshold
            ):
                redis_client.delete(key)
                deleted_count += 1
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning("Failed to process key %s: %s", key, e)

    return deleted_count

# Cleanup task implementation
@celery_app.task(name="backend.tasks.celery_app.cleanup_expired_results")
def cleanup_expired_results():
    """Clean up expired task results from Redis backend."""
    import redis

    logger.info("Starting cleanup of expired task results")

    try:
        redis_client = redis.from_url(RESULT_BACKEND, decode_responses=False)
        result_keys = redis_client.keys("celery-task-meta-*")

        current_time = datetime.now(UTC)
        expiry_threshold = timedelta(hours=24)

        deleted_count = _delete_expired_keys(
            redis_client, result_keys, current_time, expiry_threshold
        )

        logger.info("Cleanup completed: deleted %s expired task results", deleted_count)
        return {"deleted_count": deleted_count}

    except (OSError, ConnectionError, TimeoutError) as e:
        logger.error("Cleanup task failed: %s", e)
        raise
