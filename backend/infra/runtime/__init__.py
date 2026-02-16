"""
Orchestration Layer — Request Lifecycle Management
=====================================================

LAYER 2 in the architecture stack.

Provides:
  - Deterministic request lifecycle (receive → validate → route → execute → respond)
  - Priority-based queuing with load shedding
  - Adaptive task routing (small model vs large model)
  - Circuit breakers and retry policies
  - Request pipeline with middleware hooks

Depends on: telemetry (Layer 6)
Depended on by: api (Layer 1)
"""

from backend.infra.runtime.dispatcher import RequestDispatcher, get_dispatcher
from backend.infra.runtime.pipeline import (
    PipelineContext,
    PipelineStage,
    RequestPipeline,
)
from backend.infra.runtime.queue import PriorityQueueManager, get_queue_manager
from backend.infra.runtime.router import AdaptiveRouter, get_router

__all__ = [
    "AdaptiveRouter",
    "PipelineContext",
    "PipelineStage",
    "PriorityQueueManager",
    "RequestDispatcher",
    "RequestPipeline",
    "get_dispatcher",
    "get_queue_manager",
    "get_router",
]
