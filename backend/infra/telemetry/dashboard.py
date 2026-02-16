"""
Observability Dashboard — System-Wide Monitoring API
=======================================================

Aggregates data from all observability layers into
a single unified dashboard endpoint.

Uses a registry pattern to avoid layer violations: higher layers
register their stats providers at init time, and the dashboard
reads from the registry without importing from upper layers.

Provides:
  - Real-time system health
  - Latency percentiles per model
  - Token throughput metrics
  - GPU/CPU utilization
  - Queue depth and shed rates
  - Cache hit rates
  - Circuit breaker states
  - Worker pool utilization
  - Profiling aggregates
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from backend.infra.telemetry.logger import get_logger
from backend.infra.telemetry.metrics import get_metrics
from backend.infra.telemetry.profiler import get_profiler

logger = get_logger(__name__)

# ── Stats Provider Registry ───────────────────────────────────────
# Higher layers register their stats functions here at startup,
# avoiding downward imports from telemetry into upper layers.

_stats_providers: dict[str, Callable[[], Any]] = {}

def register_stats_provider(name: str, provider: Callable[[], Any]) -> None:
    """Register a stats provider from any layer (called at layer init time)."""
    _stats_providers[name] = provider
    logger.info("stats_provider_registered", name=name)

async def get_observability_dashboard() -> dict[str, Any]:
    """
    Build a complete observability snapshot.

    This is the single endpoint that surfaces all telemetry
    data for monitoring dashboards and alerting systems.

    Stats are gathered from registered providers (injected by upper layers)
    rather than importing from them, preserving the layer invariant.
    """
    dashboard: dict[str, Any] = {
        "timestamp": time.time(),
        "system": {},
        "inference": {},
        "orchestration": {},
        "hardware": {},
        "cache": {},
        "workers": {},
        "profiling": {},
    }

    # ── Metrics (own layer — no violation) ──
    try:
        m = get_metrics()
        dashboard["inference"] = {
            "throughput": m.get_throughput(),
            "models": {},
        }
        for model in list(m._latency_trackers.keys()):
            dashboard["inference"]["models"][model] = m.get_latency_percentiles(model)
        dashboard["hardware"]["gpu"] = m.get_gpu_percentiles()
    except (RuntimeError, ValueError, KeyError) as exc:
        dashboard["inference"]["error"] = str(exc)

    # ── Gather stats from registered providers ──
    for name, provider in _stats_providers.items():
        try:
            result = provider()
            # If the provider returns an awaitable, await it
            if hasattr(result, "__await__"):
                result = await result

            # Map provider results to dashboard sections
            if name == "hardware":
                dashboard["hardware"].update(result)
            elif name == "orchestration":
                dashboard["orchestration"] = result
            elif name == "execution":
                dashboard["workers"] = result
            elif name == "cache":
                dashboard["cache"] = result
            elif name == "circuit_breakers":
                dashboard["circuit_breakers"] = result
            elif name == "rate_limiter":
                dashboard["rate_limiter"] = result
            elif name == "health":
                dashboard["system"]["health"] = result
            else:
                dashboard[name] = result
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("stats_provider_failed", name=name, error=str(exc))

    # ── Profiling (own layer — no violation) ──
    try:
        profiler = get_profiler()
        dashboard["profiling"] = profiler.get_aggregated()
    except (RuntimeError, ValueError):
        pass

    return dashboard
