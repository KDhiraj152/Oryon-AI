"""
Celery Worker Configuration (Principle N)
==========================================
Each Celery worker loads (at most) one heavyweight model.

Strategy:
- Dedicated queues for each model type
- Worker specialization (one model per worker)
- Memory-efficient worker pools
- Graceful model loading/unloading

Reference: "Each Celery worker loads (at most) one heavyweight model"
"""

import logging
import os

from celery import Celery
from kombu import Exchange, Queue

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "oryon",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "backend.tasks.simplify_tasks",
        "backend.tasks.translate_tasks",
        "backend.tasks.ocr_tasks",
        "backend.tasks.embedding_tasks",
        "backend.tasks.rag_tasks",
    ],
)

# =============================================================================
# Queue Configuration (Principle N: One model per worker)
# =============================================================================

# Define exchanges
default_exchange = Exchange("oryon", type="direct")

# Define queues - each model type gets its own queue
# Workers subscribe to only ONE queue to ensure single-model loading
TASK_QUEUES = (
    # LLM Simplification Queue (Qwen3-8B)
    Queue(
        "simplify",
        default_exchange,
        routing_key="simplify",
        queue_arguments={
            "x-max-priority": 10,
            "x-message-ttl": 600000,  # 10 min TTL
        },
    ),
    # Translation Queue (IndicTrans2-1B)
    Queue(
        "translate",
        default_exchange,
        routing_key="translate",
        queue_arguments={
            "x-max-priority": 10,
            "x-message-ttl": 600000,
        },
    ),
    # OCR Queue (GOT-OCR2)
    Queue(
        "ocr",
        default_exchange,
        routing_key="ocr",
        queue_arguments={
            "x-max-priority": 5,
            "x-message-ttl": 900000,  # 15 min TTL (OCR is slower)
        },
    ),
    # Embedding Queue (BGE-M3)
    Queue(
        "embedding",
        default_exchange,
        routing_key="embedding",
        queue_arguments={
            "x-max-priority": 10,
            "x-message-ttl": 300000,  # 5 min TTL
        },
    ),
    # RAG/Search Queue (uses embeddings + optional reranker)
    Queue(
        "rag",
        default_exchange,
        routing_key="rag",
        queue_arguments={
            "x-max-priority": 10,
            "x-message-ttl": 300000,
        },
    ),
    # Default queue for non-model tasks
    Queue(
        "default",
        default_exchange,
        routing_key="default",
    ),
)

# Route tasks to appropriate queues
TASK_ROUTES = {
    # Simplification tasks → simplify queue
    "backend.tasks.simplify_tasks.*": {"queue": "simplify", "routing_key": "simplify"},
    # Translation tasks → translate queue
    "backend.tasks.translate_tasks.*": {
        "queue": "translate",
        "routing_key": "translate",
    },
    # OCR tasks → ocr queue
    "backend.tasks.ocr_tasks.*": {"queue": "ocr", "routing_key": "ocr"},
    # Embedding tasks → embedding queue
    "backend.tasks.embedding_tasks.*": {
        "queue": "embedding",
        "routing_key": "embedding",
    },
    # RAG tasks → rag queue
    "backend.tasks.rag_tasks.*": {"queue": "rag", "routing_key": "rag"},
}

# =============================================================================
# Celery Configuration
# =============================================================================

celery_app.conf.update(
    # Queue settings
    task_queues=TASK_QUEUES,
    task_routes=TASK_ROUTES,
    task_default_queue="default",
    task_default_exchange="oryon",
    task_default_routing_key="default",
    # Worker settings (Principle N: Single model per worker)
    worker_concurrency=1,  # CRITICAL: Only 1 task at a time per worker
    worker_prefetch_multiplier=1,  # Don't prefetch - model might change
    # Memory management
    worker_max_tasks_per_child=100,  # Restart after 100 tasks to free memory
    worker_max_memory_per_child=8000,  # 8GB max per worker (kill if exceeded)
    # Task execution
    task_acks_late=True,  # Ack after task completes
    task_reject_on_worker_lost=True,  # Requeue if worker dies
    task_time_limit=600,  # 10 min hard limit
    task_soft_time_limit=540,  # 9 min soft limit
    # Result backend
    result_expires=3600,  # Results expire after 1 hour
    result_compression="gzip",
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Timezone
    timezone="UTC",
    enable_utc=True,
)

# =============================================================================
# Worker Specialization Classes
# =============================================================================

class WorkerType:
    """Worker type identifiers."""

    SIMPLIFY = "simplify"
    TRANSLATE = "translate"
    OCR = "ocr"
    EMBEDDING = "embedding"
    RAG = "rag"
    DEFAULT = "default"

# Model memory requirements (GB)
MODEL_MEMORY = {
    WorkerType.SIMPLIFY: 4.6,  # Qwen3-8B 4-bit MLX
    WorkerType.TRANSLATE: 1.5,  # IndicTrans2-1B
    WorkerType.OCR: 2.0,  # GOT-OCR2
    WorkerType.EMBEDDING: 2.5,  # BGE-M3
    WorkerType.RAG: 3.5,  # BGE-M3 + Reranker
}

def get_worker_queues(worker_type: str) -> list:
    """
    Get queues for a specific worker type.

    Each worker should subscribe to ONLY ONE model queue.

    Args:
        worker_type: Type of worker

    Returns:
        List of queue names for this worker
    """
    # Default queue is always included for utility tasks
    base_queues = ["default"]

    if worker_type == WorkerType.SIMPLIFY:
        return ["simplify", *base_queues]
    elif worker_type == WorkerType.TRANSLATE:
        return ["translate", *base_queues]
    elif worker_type == WorkerType.OCR:
        return ["ocr", *base_queues]
    elif worker_type == WorkerType.EMBEDDING:
        return ["embedding", *base_queues]
    elif worker_type == WorkerType.RAG:
        return ["rag", "embedding", *base_queues]  # RAG needs embeddings
    else:
        return base_queues

def get_worker_memory_limit(worker_type: str) -> int:
    """
    Get memory limit for worker type in MB.

    Args:
        worker_type: Type of worker

    Returns:
        Memory limit in MB
    """
    base_gb = MODEL_MEMORY.get(worker_type, 1.0)
    overhead_gb = 1.5  # Python overhead, OS, etc.
    return int((base_gb + overhead_gb) * 1024)

# =============================================================================
# Worker Startup Commands
# =============================================================================

"""
Start specialized workers with these commands (Principle N compliant):

# Simplification worker (Qwen3-8B)
celery -A backend.tasks.celery_config worker \
    --queues=simplify,default \
    --concurrency=1 \
    --max-memory-per-child=4096000 \
    --hostname=simplify@%h \
    --loglevel=INFO

# Translation worker (IndicTrans2-1B)
celery -A backend.tasks.celery_config worker \
    --queues=translate,default \
    --concurrency=1 \
    --max-memory-per-child=3072000 \
    --hostname=translate@%h \
    --loglevel=INFO

# OCR worker (GOT-OCR2)
celery -A backend.tasks.celery_config worker \
    --queues=ocr,default \
    --concurrency=1 \
    --max-memory-per-child=3584000 \
    --hostname=ocr@%h \
    --loglevel=INFO

# Embedding worker (BGE-M3)
celery -A backend.tasks.celery_config worker \
    --queues=embedding,default \
    --concurrency=1 \
    --max-memory-per-child=4096000 \
    --hostname=embedding@%h \
    --loglevel=INFO

# RAG worker (BGE-M3 + Reranker)
celery -A backend.tasks.celery_config worker \
    --queues=rag,embedding,default \
    --concurrency=1 \
    --max-memory-per-child=5120000 \
    --hostname=rag@%h \
    --loglevel=INFO

# Scale with multiple instances:
# celery -A backend.tasks.celery_config worker --queues=simplify,default --concurrency=1 -n simplify1@%h &
# celery -A backend.tasks.celery_config worker --queues=simplify,default --concurrency=1 -n simplify2@%h &
"""

# =============================================================================
# Worker Signals for Model Lifecycle
# =============================================================================

from celery.signals import task_postrun, task_prerun, worker_init, worker_shutdown


@worker_init.connect
def on_worker_init(sender=None, **kwargs):
    """
    Initialize model when worker starts.

    The model is determined by the queue the worker is subscribed to.
    """
    queues = sender.app.amqp.queues.consume_from.keys() if sender else []

    logger.info("Worker initializing with queues: %s", list(queues))

    # Determine worker type from queues
    worker_type = WorkerType.DEFAULT
    for q in queues:
        if q in [
            WorkerType.SIMPLIFY,
            WorkerType.TRANSLATE,
            WorkerType.OCR,
            WorkerType.EMBEDDING,
            WorkerType.RAG,
        ]:
            worker_type = q
            break

    logger.info("Worker type: %s", worker_type)

    # Pre-load model based on worker type
    # This is done lazily in tasks, but we can pre-warm here
    # model_lifecycle.pre_warm(worker_type)

@worker_shutdown.connect
def on_worker_shutdown(sender=None, **kwargs):
    """Clean up models when worker shuts down."""
    logger.info("Worker shutting down, cleaning up models...")

    # Unload models
    # model_lifecycle.unload_all()

@task_prerun.connect
def on_task_prerun(
    sender=None, task_id=None, task=None, args=None, kwargs=None, **extra
):
    """Log task start."""
    logger.debug("Task %s starting: %s", task.name, task_id)

@task_postrun.connect
def on_task_postrun(
    sender=None, task_id=None, task=None, retval=None, state=None, **extra
):
    """Log task completion."""
    logger.debug("Task %s completed: %s (%s)", task.name, task_id, state)

# =============================================================================
# Flower Monitoring Configuration
# =============================================================================

"""
Start Flower for monitoring:

flower -A backend.tasks.celery_config --port=5555

Environment variables for Flower:
- FLOWER_BASIC_AUTH=admin:password
- FLOWER_BROKER_API=redis://localhost:6379/0
"""
