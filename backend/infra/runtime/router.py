"""
Adaptive Router — Intelligent Task Routing
=============================================

Routes requests to appropriate models based on:
  - Task complexity analysis
  - Model capabilities and availability
  - Current system load
  - Hardware constraints

Implements adaptive routing:
  - Simple tasks → lightweight model (fast response)
  - Complex tasks → strong model (high quality)
  - Embedding tasks → specialized model
  - Translation tasks → language-specific model

Routing decisions are logged and fed back for continuous optimization.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from backend.core.types import ModelTier, ModelType, TaskType
from backend.infra.telemetry import get_logger, get_metrics, get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)
metrics = get_metrics()

class ComplexityLevel(StrEnum):
    """Task complexity classification."""

    TRIVIAL = "trivial"      # Greeting, simple factual
    SIMPLE = "simple"        # Single-step reasoning
    MODERATE = "moderate"    # Multi-step, context needed
    COMPLEX = "complex"      # Deep reasoning, analysis
    EXPERT = "expert"        # Multi-domain, synthesis

@dataclass
class RoutingDecision:
    """Outcome of the routing process."""

    model_tier: ModelTier
    model_type: ModelType
    task_type: TaskType
    complexity: ComplexityLevel
    device: str
    reason: str
    confidence: float = 1.0
    estimated_latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelEndpoint:
    """Available model endpoint."""

    name: str
    tier: ModelTier
    model_type: ModelType
    supported_tasks: set[TaskType]
    max_tokens: int = 8192
    max_batch: int = 1
    loaded: bool = False
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    device: str = "cpu"

# ── Complexity Analyzer ────────────────────────────────────────────

# Patterns that indicate higher complexity
_COMPLEX_PATTERNS = [
    (re.compile(r"\b(explain|analyze|compare|evaluate|synthesize)\b", re.I), 0.3),
    (re.compile(r"\b(step.by.step|think.through|reason|derive)\b", re.I), 0.3),
    (re.compile(r"\b(code|program|algorithm|function|class)\b", re.I), 0.2),
    (re.compile(r"\b(multiple|several|various|different)\b", re.I), 0.1),
    (re.compile(r"\b(however|although|despite|nevertheless)\b", re.I), 0.1),
    (re.compile(r"[?]", re.I), 0.05),
]

_SIMPLE_PATTERNS = [
    (re.compile(r"^(hi|hello|hey|thanks|thank you|ok|yes|no)\b", re.I), -0.5),
    (re.compile(r"^(what is|who is|when did|where is)\b", re.I), -0.2),
    (re.compile(r"\b(translate|convert)\b", re.I), -0.1),
]

def analyze_complexity(text: str, task_type: str = "") -> ComplexityLevel:
    """
    Analyze text complexity to determine routing.

    Combines heuristics:
      - Text length
      - Linguistic complexity markers
      - Task type hints
      - Token density estimation
    """
    if not text:
        return ComplexityLevel.TRIVIAL

    score = 0.0

    # Length factor
    word_count = len(text.split())
    if word_count < 5:
        score -= 0.3
    elif word_count < 20:
        score += 0.0
    elif word_count < 100:
        score += 0.2
    else:
        score += 0.4

    # Pattern matching
    for pattern, weight in _COMPLEX_PATTERNS:
        if pattern.search(text):
            score += weight

    for pattern, weight in _SIMPLE_PATTERNS:
        if pattern.search(text):
            score += weight  # weight is negative

    # Task type adjustment
    task_complexity = {
        "chat": 0.0,
        "reasoning": 0.3,
        "code": 0.3,
        "summarization": 0.2,
        "translation": -0.1,
        "embedding": -0.3,
        "reranking": -0.2,
        "validation": 0.0,
    }
    score += task_complexity.get(task_type, 0.0)

    # Classify
    if score < -0.2:
        return ComplexityLevel.TRIVIAL
    elif score < 0.1:
        return ComplexityLevel.SIMPLE
    elif score < 0.3:
        return ComplexityLevel.MODERATE
    elif score < 0.5:
        return ComplexityLevel.COMPLEX
    else:
        return ComplexityLevel.EXPERT

# ── Adaptive Router ────────────────────────────────────────────────

class AdaptiveRouter:
    """
    Routes requests to optimal model endpoints.

    Balances quality, latency, and resource usage through
    complexity analysis and load-aware routing.
    """

    def __init__(self) -> None:
        self._endpoints: dict[str, ModelEndpoint] = {}
        self._routing_history: list[tuple[float, str, str]] = []  # (time, task, model)
        self._lock = threading.Lock()

        # Complexity → tier mapping
        self._tier_map: dict[ComplexityLevel, ModelTier] = {
            ComplexityLevel.TRIVIAL: ModelTier.LIGHTWEIGHT,
            ComplexityLevel.SIMPLE: ModelTier.LIGHTWEIGHT,
            ComplexityLevel.MODERATE: ModelTier.STANDARD,
            ComplexityLevel.COMPLEX: ModelTier.STRONG,
            ComplexityLevel.EXPERT: ModelTier.STRONG,
        }

        # Task → model type mapping
        self._task_model_map: dict[str, ModelType] = {
            "chat": ModelType.LLM,
            "reasoning": ModelType.LLM,
            "code": ModelType.LLM,
            "summarization": ModelType.LLM,
            "translation": ModelType.TRANSLATION,
            "embedding": ModelType.EMBEDDING,
            "reranking": ModelType.RERANKER,
            "tts": ModelType.TTS,
            "stt": ModelType.STT,
            "ocr": ModelType.OCR,
            "validation": ModelType.LLM,
        }

    def register_endpoint(self, endpoint: ModelEndpoint) -> None:
        """Register an available model endpoint."""
        with self._lock:
            self._endpoints[endpoint.name] = endpoint
            logger.info(
                "endpoint_registered",
                name=endpoint.name,
                tier=endpoint.tier.value,
                model_type=endpoint.model_type.value,
            )

    def unregister_endpoint(self, name: str) -> None:
        """Remove a model endpoint."""
        with self._lock:
            self._endpoints.pop(name, None)

    def update_endpoint_stats(
        self,
        name: str,
        *,
        loaded: bool | None = None,
        avg_latency_ms: float | None = None,
        error_rate: float | None = None,
        device: str | None = None,
    ) -> None:
        """Update endpoint performance statistics."""
        with self._lock:
            ep = self._endpoints.get(name)
            if ep:
                if loaded is not None:
                    ep.loaded = loaded
                if avg_latency_ms is not None:
                    ep.avg_latency_ms = avg_latency_ms
                if error_rate is not None:
                    ep.error_rate = error_rate
                if device is not None:
                    ep.device = device

    def route(
        self,
        *,
        task_type: str,
        text: str = "",
        force_model: str | None = None,
        force_tier: ModelTier | None = None,
    ) -> RoutingDecision:
        """
        Determine the optimal model for a request.

        Args:
            task_type: Type of task (chat, translate, embed, etc.)
            text: Input text for complexity analysis
            force_model: Force routing to a specific model
            force_tier: Force routing to a specific tier
        """
        with tracer.span("router.route", attributes={"task_type": task_type}):
            # Determine model type
            model_type = self._task_model_map.get(task_type, ModelType.LLM)

            # Analyze complexity
            complexity = analyze_complexity(text, task_type)

            # Determine tier
            tier = force_tier or self._tier_map.get(complexity, ModelTier.STANDARD)

            # Find best endpoint
            if force_model:
                endpoint = self._endpoints.get(force_model)
                if endpoint:
                    return RoutingDecision(
                        model_tier=endpoint.tier,
                        model_type=endpoint.model_type,
                        task_type=TaskType(task_type) if task_type in TaskType.__members__.values() else TaskType.CHAT,
                        complexity=complexity,
                        device=endpoint.device,
                        reason=f"forced_model={force_model}",
                    )

            best = self._find_best_endpoint(model_type, tier)

            decision = RoutingDecision(
                model_tier=best.tier if best else tier,
                model_type=model_type,
                task_type=TaskType(task_type) if task_type in [t.value for t in TaskType] else TaskType.CHAT,
                complexity=complexity,
                device=best.device if best else "cpu",
                reason=self._explain_routing(complexity, tier, best),
                estimated_latency_ms=best.avg_latency_ms if best else None,
            )

            # Record routing for feedback
            with self._lock:
                self._routing_history.append(
                    (time.monotonic(), task_type, best.name if best else "none")
                )
                # Trim history
                if len(self._routing_history) > 10000:
                    self._routing_history = self._routing_history[-5000:]

            logger.debug(
                "routing_decision",
                task_type=task_type,
                complexity=complexity.value,
                tier=decision.model_tier.value,
                device=decision.device,
                reason=decision.reason,
            )

            return decision

    def _find_best_endpoint(
        self, model_type: ModelType, preferred_tier: ModelTier
    ) -> ModelEndpoint | None:
        """Find the best available endpoint matching requirements."""
        candidates = [
            ep
            for ep in self._endpoints.values()
            if ep.model_type == model_type and ep.loaded and ep.error_rate < 0.5
        ]

        if not candidates:
            # Fall back to any endpoint of the right type
            candidates = [
                ep for ep in self._endpoints.values() if ep.model_type == model_type
            ]

        if not candidates:
            return None

        # Prefer matching tier, then sort by latency
        tier_match = [ep for ep in candidates if ep.tier == preferred_tier]
        if tier_match:
            return min(tier_match, key=lambda ep: ep.avg_latency_ms)

        # Fall back to closest tier
        return min(candidates, key=lambda ep: abs(ep.tier.value.__hash__() - preferred_tier.value.__hash__()))

    def _explain_routing(
        self,
        complexity: ComplexityLevel,
        tier: ModelTier,
        endpoint: ModelEndpoint | None,
    ) -> str:
        """Generate human-readable routing explanation."""
        parts = [f"complexity={complexity.value}", f"tier={tier.value}"]
        if endpoint:
            parts.append(f"endpoint={endpoint.name}")
            if endpoint.avg_latency_ms > 0:
                parts.append(f"est_latency={endpoint.avg_latency_ms:.0f}ms")
        else:
            parts.append("no_endpoint_available")
        return "; ".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        with self._lock:
            return {
                "registered_endpoints": len(self._endpoints),
                "loaded_endpoints": sum(
                    1 for ep in self._endpoints.values() if ep.loaded
                ),
                "routing_history_size": len(self._routing_history),
                "endpoints": {
                    name: {
                        "tier": ep.tier.value,
                        "type": ep.model_type.value,
                        "loaded": ep.loaded,
                        "avg_latency_ms": round(ep.avg_latency_ms, 1),
                        "error_rate": round(ep.error_rate, 3),
                        "device": ep.device,
                    }
                    for name, ep in self._endpoints.items()
                },
            }

# ── Singleton ──────────────────────────────────────────────────────

_router: AdaptiveRouter | None = None

def get_router() -> AdaptiveRouter:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _router
    if _router is not None:
        return _router
    _router = AdaptiveRouter()
    return _router