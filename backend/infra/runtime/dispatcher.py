"""
Request Dispatcher — Orchestration Entry Point
=================================================

The dispatcher is the single entry point from the API layer
into the orchestration layer. It:
  1. Wraps raw requests into PipelineContext
  2. Routes through pipeline stages
  3. Dispatches to execution layer
  4. Returns results to API layer

This replaces direct service calls from routes,
enforcing the layered architecture invariant.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

from backend.infra.runtime.pipeline import (
    PipelineContext,
    PipelineStage,
    RequestPipeline,
    RequestPriority,
    create_default_pipeline,
)
from backend.infra.runtime.queue import PriorityQueueManager, get_queue_manager
from backend.infra.runtime.router import AdaptiveRouter, get_router
from backend.infra.telemetry import get_logger, get_metrics, get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)
metrics = get_metrics()

class RequestDispatcher:
    """
    Central dispatcher that coordinates the request lifecycle.

    Usage from API routes:
        dispatcher = get_dispatcher()
        result = await dispatcher.dispatch(
            task_type="chat",
            payload={"prompt": "Hello", "context": [...]},
            priority=RequestPriority.NORMAL,
        )
    """

    def __init__(
        self,
        *,
        pipeline: RequestPipeline | None = None,
        router: AdaptiveRouter | None = None,
        queue: PriorityQueueManager | None = None,
    ) -> None:
        self._pipeline = pipeline or create_default_pipeline()
        self._router = router or get_router()
        self._queue = queue or get_queue_manager()
        self._execution_handler: Any = None  # Set during init
        self._initialized = False

        # Install routing stage
        self._pipeline.add_stage(PipelineStage.ROUTE, self._route_stage)
        # Install execution stage
        self._pipeline.add_stage(PipelineStage.EXECUTE, self._execute_stage)

    async def initialize(self, execution_handler: Any = None) -> None:
        """
        Initialize dispatcher with execution layer reference.

        Args:
            execution_handler: The execution runtime (InferenceRuntime)
        """
        self._execution_handler = execution_handler
        self._initialized = True
        logger.info("dispatcher_initialized")

    async def dispatch(
        self,
        *,
        task_type: str,
        payload: dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_s: float = 30.0,
        user_id: str | None = None,
        tenant_id: str | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Dispatch a request through the full pipeline.

        Returns the result payload or raises on error.
        """
        ctx = PipelineContext(
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout_s=timeout_s,
            user_id=user_id,
            tenant_id=tenant_id,
            metadata={"stream": stream},
        )

        with tracer.span(
            "dispatcher.dispatch",
            attributes={
                "task_type": task_type,
                "priority": priority.name,
            },
        ):
            result_ctx = await self._pipeline.process(ctx)

            if result_ctx.error:
                raise result_ctx.error

            return {
                "result": result_ctx.result,
                "request_id": result_ctx.request_id,
                "elapsed_ms": round(result_ctx.elapsed_ms, 2),
                "model": result_ctx.target_model,
                "device": result_ctx.target_device,
            }

    async def dispatch_stream(
        self,
        *,
        task_type: str,
        payload: dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_s: float = 60.0,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Dispatch a streaming request.

        Yields tokens as they are generated.
        """
        ctx = PipelineContext(
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout_s=timeout_s,
            user_id=user_id,
            tenant_id=tenant_id,
            metadata={"stream": True},
        )

        # Run pipeline up to EXECUTE stage
        ctx.mark_stage(PipelineStage.RECEIVE)
        ctx.mark_stage(PipelineStage.VALIDATE)
        ctx = await self._route_stage(ctx)
        ctx.mark_stage(PipelineStage.ENQUEUE)

        # Stream from execution layer
        if self._execution_handler and hasattr(self._execution_handler, "stream"):
            async for token in self._execution_handler.stream(
                task_type=task_type,
                payload=payload,
                model=ctx.target_model,
                device=ctx.target_device,
            ):
                yield token
        else:
            # Fallback: dispatch non-streaming and yield as single chunk
            result = await self.dispatch(
                task_type=task_type,
                payload=payload,
                priority=priority,
                timeout_s=timeout_s,
            )
            yield str(result.get("result", ""))

    async def dispatch_batch(
        self,
        *,
        task_type: str,
        payloads: list[dict[str, Any]],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_s: float = 120.0,
    ) -> list[dict[str, Any]]:
        """
        Dispatch a batch of requests.

        Automatically batches eligible requests for hardware efficiency.
        """
        with tracer.span(
            "dispatcher.dispatch_batch",
            attributes={"task_type": task_type, "batch_size": len(payloads)},
        ):
            # For batch-eligible tasks, delegate to execution batcher
            if self._execution_handler and hasattr(self._execution_handler, "batch_execute"):
                result = await self._execution_handler.batch_execute(
                    task_type=task_type,
                    payloads=payloads,
                )
                return cast(list[dict[str, Any]], result)

            # Fallback: parallel individual dispatch
            tasks = [
                self.dispatch(
                    task_type=task_type,
                    payload=p,
                    priority=priority,
                    timeout_s=timeout_s,
                )
                for p in payloads
            ]
            return cast(list[dict[str, Any]], await asyncio.gather(*tasks, return_exceptions=True))

    # ── Pipeline Stage Handlers ────────────────────────────────────

    async def _route_stage(self, ctx: PipelineContext) -> PipelineContext:
        """Route the request to the appropriate model."""
        text = ctx.payload.get("prompt", "") or ctx.payload.get("text", "")
        decision = self._router.route(
            task_type=ctx.task_type,
            text=text,
            force_model=ctx.payload.get("model"),
        )

        ctx.target_model = decision.model_type.value
        ctx.target_device = decision.device
        ctx.batch_eligible = ctx.task_type in ("embedding", "reranking")
        ctx.metadata["routing"] = {
            "complexity": decision.complexity.value,
            "tier": decision.model_tier.value,
            "reason": decision.reason,
        }

        return ctx

    async def _execute_stage(self, ctx: PipelineContext) -> PipelineContext:
        """Execute the request using the execution layer."""
        if self._execution_handler is None:
            # Fallback: try direct service call (backward compatibility)
            ctx.result = await self._fallback_execute(ctx)
            return ctx

        result = await self._execution_handler.execute(
            task_type=ctx.task_type,
            payload=ctx.payload,
            model=ctx.target_model,
            device=ctx.target_device,
            timeout_s=ctx.timeout_s,
        )
        ctx.result = result
        return ctx

    async def _fallback_execute(self, ctx: PipelineContext) -> Any:
        """
        Backward-compatible execution via existing service layer.

        This allows gradual migration — routes can use the dispatcher
        even before the full execution layer is wired up.
        """
        logger.warning(
            "fallback_execution",
            task_type=ctx.task_type,
            msg="Using legacy service path. Wire execution layer for full pipeline.",
        )

        task = ctx.task_type
        payload = ctx.payload

        try:
            if task in ("chat", "reasoning", "code"):
                from backend.services.chat.engine import get_ai_engine
                ai_engine: Any = get_ai_engine()
                return await ai_engine.generate(
                    prompt=payload.get("prompt", ""),
                    context=payload.get("context", []),
                )
            elif task == "embedding":
                from backend.services.chat.rag import get_rag_service
                rag = get_rag_service()
                return await asyncio.to_thread(
                    rag.embed_text, payload.get("text", "")  # type: ignore[attr-defined]
                )
            elif task == "translation":
                from backend.ml.translate.engine import TranslationEngine
                translator: Any = TranslationEngine()
                return await asyncio.to_thread(
                    translator.translate,
                    payload.get("text", ""),
                    payload.get("source_lang", "en"),
                    payload.get("target_lang", "hi"),
                )
            elif task == "tts":
                from backend.ml.speech.speech_generator import SpeechGenerator
                gen = SpeechGenerator()
                return await asyncio.to_thread(
                    gen.generate, payload.get("text", "")  # type: ignore[attr-defined]
                )
            else:
                raise ValueError(f"Unknown task type: {task}")
        except ImportError as e:
            logger.error("fallback_import_error", exc=e, task=task)
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            "initialized": self._initialized,
            "has_execution_handler": self._execution_handler is not None,
            "router": self._router.get_stats(),
            "queue": self._queue.get_stats(),
        }

# ── Singleton ──────────────────────────────────────────────────────

_dispatcher: RequestDispatcher | None = None

def get_dispatcher() -> RequestDispatcher:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _dispatcher
    if _dispatcher is not None:
        return _dispatcher
    _dispatcher = RequestDispatcher()
    return _dispatcher