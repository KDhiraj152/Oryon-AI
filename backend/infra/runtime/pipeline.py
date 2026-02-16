"""
Request Pipeline — Deterministic Request Lifecycle
=====================================================

Implements a staged request pipeline:
  RECEIVE → VALIDATE → ROUTE → ENQUEUE → EXECUTE → TRANSFORM → RESPOND

Each stage is a composable async function with:
  - Input/output contracts
  - Timeout enforcement
  - Cancellation support
  - Telemetry emission

The pipeline ensures every request follows an identical lifecycle
regardless of task type (chat, translate, embed, etc.).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from backend.infra.telemetry import get_logger, get_metrics, get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)
metrics = get_metrics()

class PipelineStage(StrEnum):
    """Stages in the request lifecycle."""

    RECEIVE = "receive"
    VALIDATE = "validate"
    ROUTE = "route"
    ENQUEUE = "enqueue"
    EXECUTE = "execute"
    TRANSFORM = "transform"
    RESPOND = "respond"
    ERROR = "error"

class RequestPriority(IntEnum):
    """Request priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

@dataclass
class PipelineContext:
    """
    Mutable request context flowing through the pipeline.

    This context is passed through every pipeline stage and mutated in place
    (e.g. ``mark_stage``, setting ``result`` / ``error``).
    It is **not** thread-safe — each request should have its own instance.

    Every field added here is automatically available to all stages,
    telemetry, and error handlers.
    """

    # Identity
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    correlation_id: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None

    # Timing
    received_at: float = field(default_factory=time.monotonic)
    stage_times: dict[str, float] = field(default_factory=dict)

    # Request
    task_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    priority: RequestPriority = RequestPriority.NORMAL
    timeout_s: float = 30.0

    # Routing
    target_model: str | None = None
    target_device: str | None = None
    batch_eligible: bool = False

    # Execution
    result: Any = None
    error: Exception | None = None
    cancelled: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    current_stage: PipelineStage = PipelineStage.RECEIVE

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.received_at) * 1000

    def mark_stage(self, stage: PipelineStage) -> None:
        self.current_stage = stage
        self.stage_times[stage.value] = time.monotonic()

    def to_telemetry(self) -> dict[str, Any]:
        """Extract telemetry-safe attributes."""
        return {
            "request_id": self.request_id,
            "task_type": self.task_type,
            "priority": self.priority.name,
            "target_model": self.target_model,
            "target_device": self.target_device,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "stage": self.current_stage.value,
        }

# ── Stage Handler Type ─────────────────────────────────────────────

StageHandler = Callable[[PipelineContext], Awaitable[PipelineContext]]

# ── Request Pipeline ───────────────────────────────────────────────

class RequestPipeline:
    """
    Deterministic request pipeline with pluggable stages.

    The pipeline processes every request through the same sequence of stages.
    Each stage can modify the context and pass it to the next stage.
    Stages are async functions: (PipelineContext) -> PipelineContext.

    Usage:
        pipeline = RequestPipeline()
        pipeline.add_stage(PipelineStage.VALIDATE, validate_request)
        pipeline.add_stage(PipelineStage.ROUTE, route_request)
        pipeline.add_stage(PipelineStage.EXECUTE, execute_request)

        result = await pipeline.process(context)
    """

    def __init__(self) -> None:
        self._stages: dict[PipelineStage, list[StageHandler]] = {
            stage: [] for stage in PipelineStage
        }
        self._error_handlers: list[StageHandler] = []
        self._before_hooks: list[StageHandler] = []
        self._after_hooks: list[StageHandler] = []

    def add_stage(self, stage: PipelineStage, handler: StageHandler) -> None:
        """Register a handler for a pipeline stage."""
        self._stages[stage].append(handler)

    def on_error(self, handler: StageHandler) -> None:
        """Register an error handler."""
        self._error_handlers.append(handler)

    def before_each(self, hook: StageHandler) -> None:
        """Register a hook to run before each stage."""
        self._before_hooks.append(hook)

    def after_each(self, hook: StageHandler) -> None:
        """Register a hook to run after each stage."""
        self._after_hooks.append(hook)

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Execute the full request pipeline.

        Processes stages in order: RECEIVE → VALIDATE → ROUTE → ENQUEUE → EXECUTE → TRANSFORM → RESPOND.
        On error, jumps to ERROR stage and runs error handlers.
        """
        execution_order = [
            PipelineStage.RECEIVE,
            PipelineStage.VALIDATE,
            PipelineStage.ROUTE,
            PipelineStage.ENQUEUE,
            PipelineStage.EXECUTE,
            PipelineStage.TRANSFORM,
            PipelineStage.RESPOND,
        ]

        with tracer.span(
            "pipeline.process",
            attributes={
                "request_id": ctx.request_id,
                "task_type": ctx.task_type,
                "priority": ctx.priority.name,
            },
        ) as span:
            try:
                for stage in execution_order:
                    if ctx.cancelled:
                        logger.info(
                            "pipeline_cancelled",
                            request_id=ctx.request_id,
                            stage=stage.value,
                        )
                        break

                    handlers = self._stages.get(stage, [])
                    if not handlers:
                        continue

                    ctx.mark_stage(stage)

                    with tracer.span(
                        f"pipeline.stage.{stage.value}",
                        attributes={"request_id": ctx.request_id},
                    ):
                        # Before hooks
                        for hook in self._before_hooks:
                            ctx = await hook(ctx)

                        # Execute stage handlers
                        for handler in handlers:
                            try:
                                ctx = await asyncio.wait_for(
                                    handler(ctx),
                                    timeout=ctx.timeout_s,
                                )
                            except TimeoutError:
                                ctx.error = TimeoutError(
                                    f"Stage {stage.value} timed out after {ctx.timeout_s}s"
                                )
                                raise

                        # After hooks
                        for hook in self._after_hooks:
                            ctx = await hook(ctx)

                # Record success
                span.set_attribute("status", "success")
                span.set_attribute("total_ms", round(ctx.elapsed_ms, 2))

            except Exception as exc:
                ctx.error = exc
                ctx.mark_stage(PipelineStage.ERROR)
                span.set_attribute("status", "error")
                span.set_attribute("error_type", type(exc).__name__)

                # Run error handlers
                for handler in self._error_handlers:
                    try:
                        ctx = await handler(ctx)
                    except Exception as eh_exc:
                        logger.error(
                            "error_handler_failed",
                            exc=eh_exc,
                            request_id=ctx.request_id,
                        )

            finally:
                # Emit metrics
                elapsed_s = ctx.elapsed_ms / 1000
                status = "error" if ctx.error else "success"
                metrics.record_http_request(
                    method="PIPELINE",
                    path=ctx.task_type,
                    status=200 if status == "success" else 500,
                    latency_s=elapsed_s,
                )

                logger.info(
                    "pipeline_complete",
                    request_id=ctx.request_id,
                    task_type=ctx.task_type,
                    status=status,
                    elapsed_ms=round(ctx.elapsed_ms, 2),
                    stages=len(ctx.stage_times),
                )

        return ctx

# ── Default Pipeline Factory ──────────────────────────────────────

def create_default_pipeline() -> RequestPipeline:
    """Create a pipeline with standard stage handlers."""
    pipeline = RequestPipeline()

    async def receive_handler(ctx: PipelineContext) -> PipelineContext:
        """Set up request context for telemetry."""
        from backend.infra.telemetry.logger import set_request_context

        set_request_context(
            request_id=ctx.request_id,
            trace_id=ctx.correlation_id,
            user_id=ctx.user_id,
            tenant_id=ctx.tenant_id,
        )
        return ctx

    async def validate_handler(ctx: PipelineContext) -> PipelineContext:
        """Basic payload validation."""
        if not ctx.task_type:
            raise ValueError("task_type is required")
        return ctx

    async def respond_handler(ctx: PipelineContext) -> PipelineContext:
        """Clean up request context."""
        from backend.infra.telemetry.logger import clear_request_context

        clear_request_context()
        return ctx

    pipeline.add_stage(PipelineStage.RECEIVE, receive_handler)
    pipeline.add_stage(PipelineStage.VALIDATE, validate_handler)
    pipeline.add_stage(PipelineStage.RESPOND, respond_handler)

    return pipeline
