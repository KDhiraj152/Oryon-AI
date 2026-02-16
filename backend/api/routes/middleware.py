"""
Middleware API Routes
=====================

Exposes the orchestration middleware's capabilities via REST endpoints:
- POST /middleware/process    → Process a request through the full pipeline
- GET  /middleware/stats      → Middleware observability dashboard
- GET  /middleware/classify   → Classify a prompt (dry run, no inference)
- POST /middleware/evaluate   → Force a heuristic adjustment cycle
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["middleware"])

# ── Request / Response Schemas ───────────────────────────────────────────────

class MiddlewareProcessRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10_000)
    session_id: str | None = None
    timeout_ms: float | None = None
    priority: int = Field(default=2, ge=0, le=4)
    payload: dict[str, Any] = Field(default_factory=dict)

class ClassifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10_000)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_middleware(request: Request):
    """Get the middleware orchestrator from app state."""
    mw = getattr(request.app.state, "middleware_orchestrator", None)
    if mw is None:
        from ...middleware.orchestrator import get_middleware
        mw = get_middleware()
    return mw

# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/process")
async def middleware_process(body: MiddlewareProcessRequest, request: Request):
    """
    Process a request through the full middleware pipeline.

    This is the high-level entry point that coordinates:
    - Intent classification
    - Model routing
    - Memory retrieval
    - Pipeline execution
    - Self-evaluation
    """
    mw = _get_middleware(request)
    response = await mw.process(
        prompt=body.prompt,
        session_id=body.session_id,
        payload=body.payload,
        timeout_ms=body.timeout_ms,
        priority=body.priority,
    )

    return {
        "request_id": response.request_id,
        "success": response.success,
        "result": response.result,
        "intent": response.intent,
        "model_used": response.model_used,
        "total_latency_ms": response.total_latency_ms,
        "stage_latencies": response.stage_latencies,
        "quality": response.evaluation_quality,
        "from_cache": response.from_cache,
        "early_exit": response.early_exit,
    }

@router.get("/stats")
async def middleware_stats(request: Request):
    """Get comprehensive middleware statistics and observability data."""
    mw = _get_middleware(request)
    return await mw.get_stats()

@router.post("/classify")
async def middleware_classify(body: ClassifyRequest, request: Request):
    """
    Classify a prompt without executing inference (dry run).

    Returns intent, complexity, model target, and routing decision.
    """
    mw = _get_middleware(request)
    result = mw.classifier.classify(body.prompt)

    return {
        "intent": result.intent.value,
        "complexity": result.complexity.name,
        "complexity_score": result.complexity_score,
        "model_target": result.model_target.value,
        "estimated_tokens": result.estimated_tokens,
        "needs_rag": result.needs_rag,
        "needs_translation": result.needs_translation,
        "needs_tts": result.needs_tts,
        "parallel_safe": result.parallel_safe,
        "source_language": result.source_language,
        "target_language": result.target_language,
        "confidence": result.confidence,
    }

@router.post("/adjust")
async def middleware_adjust(request: Request):
    """
    Force an immediate heuristic adjustment cycle.

    Analyzes the routing ledger and tunes complexity thresholds
    based on observed over/under-provisioning patterns.
    """
    mw = _get_middleware(request)
    adjustments = mw.evaluator.force_adjustment()
    return {
        "adjustments_made": len(adjustments),
        "details": adjustments,
        "current_thresholds": mw.evaluator.heuristic_adjuster.get_stats(),
    }

@router.get("/memory")
async def middleware_memory(request: Request):
    """Get memory subsystem statistics."""
    mw = _get_middleware(request)
    return await mw.memory.get_stats()

@router.get("/latency")
async def middleware_latency(request: Request):
    """Get latency control statistics and adaptive timeouts."""
    mw = _get_middleware(request)
    return mw.latency.get_stats()
