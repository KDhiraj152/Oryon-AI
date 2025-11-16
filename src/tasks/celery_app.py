"""Celery application configuration for async task processing."""
import os
from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure, task_success, task_retry
from kombu import Queue
import logging

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = os.getenv('REDIS_PORT', 6379)
REDIS_DB = os.getenv('CELERY_REDIS_DB', 1)  # Use different DB than cache
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

# Build Redis URL
if REDIS_PASSWORD:
    BROKER_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
    RESULT_BACKEND = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
else:
    BROKER_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
    RESULT_BACKEND = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'

# Create Celery app
celery_app = Celery(
    'shiksha_setu',
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=[
        'src.tasks.pipeline_tasks',
        'src.tasks.qa_tasks'
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        'pipeline.*': {'queue': 'pipeline'},
        'src.tasks.pipeline_tasks.*': {'queue': 'pipeline'},
        'src.tasks.qa_tasks.*': {'queue': 'pipeline'}
    },
    
    # Task queues
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('pipeline', routing_key='pipeline'),
        Queue('ocr', routing_key='ocr'),
        Queue('ml_gpu', routing_key='ml_gpu'),
        Queue('ml_cpu', routing_key='ml_cpu'),
        Queue('dead_letter', routing_key='dead_letter'),  # Dead letter queue
    ),
    
    # Result settings
    result_expires=3600,  # 1 hour default
    result_extended=True,
    result_backend_transport_options={
        'master_name': 'mymaster',
        'visibility_timeout': 3600,
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
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Retries
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    task_autoretry_for=(Exception,),  # Auto retry on exceptions
    task_retry_backoff=True,  # Exponential backoff
    task_retry_backoff_max=600,  # Max 10 minutes backoff
    task_retry_jitter=True,  # Add randomness to prevent thundering herd
    
    # Beat schedule (for periodic tasks)
    beat_schedule={
        'cleanup-old-tasks': {
            'task': 'src.tasks.pipeline_tasks.cleanup_old_results',
            'schedule': 3600.0,  # Every hour
        },
    },
)


# Task lifecycle signals
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Log task start."""
    logger.info(f"Task {task.name} [{task_id}] started")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, 
                         retval=None, state=None, **extra):
    """Log task completion."""
    logger.info(f"Task {task.name} [{task_id}] completed with state: {state}")


@task_success.connect
def task_success_handler(sender=None, result=None, **extra):
    """Handle successful task completion."""
    logger.info(f"Task {sender.name} completed successfully")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None, 
                         kwargs=None, traceback=None, einfo=None, **extra):
    """Handle task failure - log and optionally alert."""
    logger.error(f"Task {sender.name} [{task_id}] failed: {exception}")
    logger.error(f"Traceback: {traceback}")
    
    # TODO: Add alerting mechanism (email, Slack, etc.)
    # Example: send_alert(f"Task {sender.name} failed: {exception}")


@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **extra):
    """Handle task retry."""
    logger.warning(f"Task {sender.name} [{task_id}] is being retried: {reason}")


# Utility functions
def get_task_info(task_id: str) -> dict:
    """Get task status and result."""
    result = celery_app.AsyncResult(task_id)
    
    info = {
        'task_id': task_id,
        'state': result.state,
        'ready': result.ready(),
        'successful': result.successful() if result.ready() else None,
        'failed': result.failed() if result.ready() else None,
    }
    
    if result.ready():
        if result.successful():
            info['result'] = result.result
        elif result.failed():
            info['error'] = str(result.info)
            info['traceback'] = result.traceback
    else:
        # Task in progress - get metadata
        if hasattr(result.info, 'get'):
            info['progress'] = result.info.get('progress', 0)
            info['stage'] = result.info.get('stage', 'unknown')
            info['message'] = result.info.get('message', '')
    
    return info


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """Cancel a running task."""
    try:
        celery_app.control.revoke(task_id, terminate=terminate, signal='SIGKILL')
        logger.info(f"Task {task_id} revoked (terminate={terminate})")
        return True
    except Exception as e:
        logger.error(f"Failed to revoke task {task_id}: {e}")
        return False


# Export
__all__ = ['celery_app', 'get_task_info', 'revoke_task']
