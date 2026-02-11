"""
Agent System API Routes
========================

Exposes the multi-agent system's telemetry, evaluation, and optimization
endpoints for observability and admin control.

Endpoints:
- GET  /agents/status    — Agent registry status
- GET  /agents/metrics   — System-wide evaluation metrics
- GET  /agents/sla       — SLA compliance status
- GET  /agents/regressions — Detected performance regressions
- GET  /agents/optimizations — Self-improvement optimization log
- POST /agents/optimize  — Force an optimization cycle
"""

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agents"])

_EVAL_AGENT_UNAVAILABLE = "EvaluationAgent not available"


def _get_registry():
    """Get the agent registry singleton."""
    try:
        from ...agents.base import AgentRegistry
        return AgentRegistry()
    except Exception:
        raise HTTPException(status_code=503, detail="Agent system not initialized")


@router.get("/status")
async def agent_status():
    """Get status of all registered agents."""
    registry = _get_registry()
    return {
        "agents": registry.get_all_stats(),
        "count": len(registry._agents),
    }


@router.get("/metrics")
async def agent_metrics():
    """Get comprehensive evaluation metrics from the evaluation agent."""
    registry = _get_registry()
    from ...agents.base import AgentMessage, MessageType

    eval_agent = registry._agents.get("evaluation")
    if not eval_agent:
        raise HTTPException(status_code=503, detail=_EVAL_AGENT_UNAVAILABLE)

    msg = AgentMessage(
        msg_type=MessageType.REQUEST,
        sender="api",
        recipient="evaluation",
        payload={"action": "get_report"},
    )
    response = await eval_agent.handle_message(msg)
    return response.payload if response else {}


@router.get("/sla")
async def sla_status():
    """Get SLA compliance status per model."""
    registry = _get_registry()
    from ...agents.base import AgentMessage, MessageType

    eval_agent = registry._agents.get("evaluation")
    if not eval_agent:
        raise HTTPException(status_code=503, detail=_EVAL_AGENT_UNAVAILABLE)

    msg = AgentMessage(
        msg_type=MessageType.REQUEST,
        sender="api",
        recipient="evaluation",
        payload={"action": "get_sla_status"},
    )
    response = await eval_agent.handle_message(msg)
    return response.payload if response else {}


@router.get("/regressions")
async def regressions():
    """Get detected performance regressions."""
    registry = _get_registry()
    from ...agents.base import AgentMessage, MessageType

    eval_agent = registry._agents.get("evaluation")
    if not eval_agent:
        raise HTTPException(status_code=503, detail=_EVAL_AGENT_UNAVAILABLE)

    msg = AgentMessage(
        msg_type=MessageType.REQUEST,
        sender="api",
        recipient="evaluation",
        payload={"action": "get_regressions"},
    )
    response = await eval_agent.handle_message(msg)
    return response.payload if response else {}


@router.get("/optimizations")
async def optimization_log():
    """Get self-improvement optimization proposals and their outcomes."""
    registry = _get_registry()
    from ...agents.base import AgentMessage, MessageType

    si_agent = registry._agents.get("self_improvement")
    if not si_agent:
        raise HTTPException(status_code=503, detail="SelfImprovementAgent not available")

    msg = AgentMessage(
        msg_type=MessageType.REQUEST,
        sender="api",
        recipient="self_improvement",
        payload={"action": "get_proposals"},
    )
    response = await si_agent.handle_message(msg)
    return response.payload if response else {}


@router.get("/model-stats")
async def model_execution_stats():
    """Get per-model inference statistics."""
    registry = _get_registry()
    from ...agents.base import AgentMessage, MessageType

    me_agent = registry._agents.get("model_execution")
    if not me_agent:
        raise HTTPException(status_code=503, detail="ModelExecutionAgent not available")

    msg = AgentMessage(
        msg_type=MessageType.REQUEST,
        sender="api",
        recipient="model_execution",
        payload={"action": "get_stats"},
    )
    response = await me_agent.handle_message(msg)
    return response.payload if response else {}


@router.get("/hardware-config")
async def hardware_config():
    """Get current hardware optimizer configuration."""
    registry = _get_registry()
    from ...agents.base import AgentMessage, MessageType

    hw_agent = registry._agents.get("hardware_optimizer")
    if not hw_agent:
        raise HTTPException(status_code=503, detail="HardwareOptimizerAgent not available")

    msg = AgentMessage(
        msg_type=MessageType.REQUEST,
        sender="api",
        recipient="hardware_optimizer",
        payload={"action": "get_config"},
    )
    response = await hw_agent.handle_message(msg)
    return response.payload if response else {}


@router.post("/optimize")
async def force_optimize(target: str = "all"):
    """Force a self-improvement optimization cycle (admin only)."""
    registry = _get_registry()
    from ...agents.base import AgentMessage, MessageType

    si_agent = registry._agents.get("self_improvement")
    if not si_agent:
        raise HTTPException(status_code=503, detail="SelfImprovementAgent not available")

    msg = AgentMessage(
        msg_type=MessageType.REQUEST,
        sender="api",
        recipient="self_improvement",
        payload={"action": "force_optimization", "target": target},
    )
    response = await si_agent.handle_message(msg)
    return response.payload if response else {}
