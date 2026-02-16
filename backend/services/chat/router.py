"""
Model Router - Configurable LLM Routing
========================================

Routes requests to appropriate models based on:
- Task complexity (simple → lightweight, complex → stronger)
- Request type (chat, reasoning, code, system)
- Resource availability and latency requirements
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, ClassVar

from ...core.types import ModelTier, TaskType

logger = logging.getLogger(__name__)

# Default model key used for standard chat fallbacks
_STANDARD_MODEL_KEY = "qwen3-8b"

# Numeric ordering for tier-ceiling enforcement
TIER_ORDER: dict[ModelTier, int] = {
    ModelTier.LIGHTWEIGHT: 0,
    ModelTier.STANDARD: 1,
    ModelTier.STRONG: 2,
    ModelTier.SPECIALIZED: 2,  # domain-specific, same cost level as STRONG
}

@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_id: str
    tier: ModelTier
    max_tokens: int
    supports_streaming: bool = True
    supports_tools: bool = False
    context_window: int = 4096
    cost_per_1k_tokens: float = 0.0  # For cost estimation
    avg_latency_ms: float = 100.0
    task_types: list[TaskType] | None = None

    def __post_init__(self):
        if self.task_types is None:
            self.task_types = [TaskType.CHAT]

@dataclass
class RoutingDecision:
    """Result of routing decision."""

    model_id: str
    tier: ModelTier
    reason: str
    estimated_latency_ms: float
    estimated_cost: float
    fallback_model_id: str | None = None
    estimated_max_tokens: int = 4096  # Dynamic token allocation based on prompt
    model_key: str = ""  # Internal short name (e.g. "qwen3-8b") for inference backend

# Token allocation based on prompt type and complexity
TOKEN_ALLOCATION = {
    # (complexity, task_type) -> token range (min, default, max)
    # Code generation needs maximum tokens
    ("high", TaskType.CODE): (4096, 6144, 8192),
    ("medium", TaskType.CODE): (2048, 4096, 6144),
    ("low", TaskType.CODE): (1024, 2048, 4096),
    # Reasoning/explanations need substantial tokens
    ("high", TaskType.REASONING): (3072, 5120, 8192),
    ("medium", TaskType.REASONING): (2048, 3072, 4096),
    ("low", TaskType.REASONING): (1024, 2048, 3072),
    # Chat - all complexity levels get adequate tokens for complete responses
    ("high", TaskType.CHAT): (4096, 6144, 8192),
    ("medium", TaskType.CHAT): (2048, 4096, 6144),
    ("low", TaskType.CHAT): (1024, 2048, 4096),  # Increased from 256-1024
    # Translation needs more tokens for Indian languages (longer scripts)
    ("high", TaskType.TRANSLATION): (1024, 2048, 4096),
    ("medium", TaskType.TRANSLATION): (512, 1024, 2048),
    ("low", TaskType.TRANSLATION): (256, 512, 1024),
    # Summarization - adequate for comprehensive summaries
    ("high", TaskType.SUMMARIZATION): (2048, 4096, 6144),
    ("medium", TaskType.SUMMARIZATION): (1024, 2048, 4096),
    ("low", TaskType.SUMMARIZATION): (512, 1024, 2048),
}

# Keywords that suggest longer output is needed
OUTPUT_EXPANSION_KEYWORDS = {
    # Code generation signals
    "write code",
    "create code",
    "implement",
    "build",
    "develop",
    "create a program",
    "write a program",
    "code for",
    "script for",
    "algorithm for",
    "function for",
    "class for",
    "solve using code",
    # Detailed explanation signals
    "explain in detail",
    "detailed explanation",
    "step by step",
    "comprehensive",
    "complete guide",
    "full tutorial",
    "walkthrough",
    "elaborate",
    "in depth",
    "thorough",
    "extensive",
    # Math/Science signals that need full derivations
    "derive",
    "prove",
    "solve step by step",
    "show all steps",
    "complete solution",
    "work through",
    "calculate",
    # Essay/writing signals
    "write an essay",
    "write about",
    "describe in detail",
    "long answer",
    "discuss",
    "analyze thoroughly",
    # List/enumeration signals
    "list all",
    "enumerate",
    "provide examples",
    "give me examples",
    "multiple examples",
    "various ways",
}

class ModelRouter:
    """
    Routes requests to appropriate models based on task and constraints.

    Features:
    - Complexity-based routing
    - Fallback chain for unavailable models
    - Cost and latency estimation
    - Dynamic model availability tracking
    """

    # Default model configurations
    DEFAULT_MODELS: ClassVar[dict] = {
        # Lightweight (fast responses)
        "qwen3-1.7b": ModelConfig(
            model_id="Qwen/Qwen3-1.7B",
            tier=ModelTier.LIGHTWEIGHT,
            max_tokens=1024,
            context_window=2048,
            avg_latency_ms=50,
            task_types=[TaskType.SYSTEM, TaskType.CHAT],
        ),
        # Standard (general chat) - PRIMARY MODEL
        _STANDARD_MODEL_KEY: ModelConfig(
            model_id="Qwen/Qwen3-8B",
            tier=ModelTier.STANDARD,
            max_tokens=4096,  # Increased from 2048 - full potential
            context_window=8192,
            avg_latency_ms=200,
            task_types=[TaskType.CHAT, TaskType.REASONING, TaskType.CODE, TaskType.VALIDATION],
        ),
        # Strong (complex reasoning)
        "qwen3-14b": ModelConfig(
            model_id="Qwen/Qwen3-14B",
            tier=ModelTier.STRONG,
            max_tokens=6144,
            context_window=32768,
            avg_latency_ms=500,
            task_types=[TaskType.REASONING, TaskType.CODE],
        ),
        # Specialized models
        "indictrans2": ModelConfig(
            model_id="ai4bharat/indictrans2-en-indic-1B",
            tier=ModelTier.SPECIALIZED,
            max_tokens=512,
            context_window=512,
            avg_latency_ms=150,
            task_types=[TaskType.TRANSLATION],
        ),
        "bge-m3": ModelConfig(
            model_id="BAAI/bge-m3",
            tier=ModelTier.SPECIALIZED,
            max_tokens=8192,
            context_window=8192,
            avg_latency_ms=30,
            task_types=[TaskType.EMBEDDING],
        ),
        "bge-reranker": ModelConfig(
            model_id="BAAI/bge-reranker-v2-m3",
            tier=ModelTier.SPECIALIZED,
            max_tokens=512,
            context_window=512,
            avg_latency_ms=20,
            task_types=[TaskType.RERANKING],
        ),
    }

    # Complexity indicators
    COMPLEXITY_KEYWORDS: ClassVar[dict] = {
        "high": [
            "analyze",
            "compare",
            "implement",
            "design",
            "architect",
            "optimize",
            "debug",
            "refactor",
            "explain in detail",
            "step by step",
            "comprehensive",
            "complex",
            "advanced",
        ],
        "low": [
            "what is",
            "define",
            "simple",
            "quick",
            "brief",
            "yes or no",
            "translate",
            "hello",
            "hi",
            "thanks",
        ],
    }

    def __init__(self, custom_models: dict[str, ModelConfig] | None = None):
        self.models = {**self.DEFAULT_MODELS}
        if custom_models:
            self.models.update(custom_models)

        # Track model availability
        self._availability: dict[str, bool] = dict.fromkeys(self.models, True)
        self._last_failure: dict[str, float] = {}

        # Compile complexity patterns
        self._high_complexity_words = set(self.COMPLEXITY_KEYWORDS["high"])
        self._low_complexity_words = set(self.COMPLEXITY_KEYWORDS["low"])

        # Pre-compile output expansion patterns for efficiency
        self._output_expansion_patterns = set(OUTPUT_EXPANSION_KEYWORDS)

    def route(
        self,
        query: str,
        task_type: TaskType = TaskType.CHAT,
        max_latency_ms: float | None = None,
        require_streaming: bool = False,
        preferred_tier: ModelTier | None = None,
        model_tier_ceiling: ModelTier | None = None,
    ) -> RoutingDecision:
        """
        Route a request to the appropriate model.

        Args:
            query: The user query
            task_type: Type of task
            max_latency_ms: Maximum acceptable latency
            require_streaming: Whether streaming is required
            preferred_tier: Preferred model tier

        Returns:
            RoutingDecision with selected model and reasoning
        """
        # Determine complexity
        complexity = self._assess_complexity(query)

        # Estimate dynamic token allocation based on prompt
        estimated_tokens = self._estimate_tokens_needed(query, task_type, complexity)

        # Get candidate models for this task type
        candidates = self._get_candidates(task_type, require_streaming)

        if not candidates:
            # Fallback to standard chat model
            return RoutingDecision(
                model_id=self.models[_STANDARD_MODEL_KEY].model_id,
                tier=ModelTier.STANDARD,
                reason="No specialized model available, using standard chat model",
                estimated_latency_ms=200,
                estimated_cost=0.0,
                estimated_max_tokens=estimated_tokens,
                model_key=_STANDARD_MODEL_KEY,
            )

        # Select based on complexity and constraints
        selected = self._select_model(
            candidates=candidates,
            complexity=complexity,
            max_latency_ms=max_latency_ms,
            preferred_tier=preferred_tier,
            model_tier_ceiling=model_tier_ceiling,
        )

        # Ensure estimated tokens don't exceed model's max
        final_tokens = min(estimated_tokens, selected.max_tokens)

        # Find fallback
        fallback = self._get_fallback(selected, candidates)

        logger.debug(
            f"Routing decision: {task_type.value}/{complexity} -> "
            f"{selected.model_id} with {final_tokens} tokens"
        )

        # Resolve model key (internal short name) for inference backend
        selected_key = next(
            (k for k, v in self.models.items() if v.model_id == selected.model_id),
            _STANDARD_MODEL_KEY,
        )

        return RoutingDecision(
            model_id=selected.model_id,
            tier=selected.tier,
            reason=f"Selected for {task_type.value} task with {complexity} complexity",
            estimated_latency_ms=selected.avg_latency_ms,
            estimated_cost=self._estimate_cost(len(query), selected),
            fallback_model_id=fallback.model_id if fallback else None,
            estimated_max_tokens=final_tokens,
            model_key=selected_key,
        )

    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity (low, medium, high)."""
        query_lower = query.lower()
        words = set(query_lower.split())

        # Check for high complexity indicators
        high_matches = len(words & self._high_complexity_words)
        low_matches = len(words & self._low_complexity_words)

        # Also consider query length
        if len(query) > 500 or high_matches >= 2:
            return "high"
        elif len(query) < 50 or low_matches >= 2:
            return "low"
        else:
            return "medium"

    def _estimate_tokens_needed(
        self, query: str, task_type: TaskType, complexity: str
    ) -> int:
        """
        Dynamically estimate how many tokens are needed for the response.

        This analyzes the prompt to determine the appropriate output length:
        - Code generation requests get maximum tokens
        - Step-by-step explanations get high tokens
        - Simple Q&A gets minimal tokens

        Args:
            query: The user's prompt
            task_type: Type of task (code, chat, reasoning, etc.)
            complexity: Assessed complexity level (low, medium, high)

        Returns:
            Estimated max_tokens for generation
        """
        query_lower = query.lower()

        # Get base allocation from task type and complexity
        key = (complexity, task_type)
        if key in TOKEN_ALLOCATION:
            min_tokens, default_tokens, max_tokens = TOKEN_ALLOCATION[key]
        else:
            # Fallback for unknown combinations
            min_tokens, default_tokens, max_tokens = 1024, 2048, 4096

        # Check for expansion signals in the prompt
        expansion_score = 0

        # Check multi-word patterns first (more specific)
        for pattern in self._output_expansion_patterns:
            if pattern in query_lower:
                expansion_score += 2

        # Code-specific patterns (highest priority)
        code_patterns = [
            "write",
            "create",
            "implement",
            "develop",
            "build",
            "code",
            "program",
            "script",
            "function",
            "class",
            "algorithm",
            "solve",
            "application",
            "app",
        ]
        code_matches = sum(1 for p in code_patterns if p in query_lower)
        if code_matches >= 2:
            # Definitely wants code - use maximum tokens
            expansion_score += 5

        # Math/derivation patterns
        math_patterns = ["derive", "prove", "calculate", "solve", "equation", "formula"]
        if any(p in query_lower for p in math_patterns):
            expansion_score += 2

        # List/enumeration patterns
        if any(p in query_lower for p in ["list", "enumerate", "all", "examples"]):
            expansion_score += 1

        # Length indicators
        if any(
            p in query_lower for p in ["detailed", "comprehensive", "complete", "full"]
        ):
            expansion_score += 2
        if any(p in query_lower for p in ["brief", "short", "quick", "simple"]):
            expansion_score -= 2

        # Determine final token count based on expansion score
        if expansion_score >= 5:
            # Maximum output needed (code generation, detailed tutorials)
            return max_tokens
        elif expansion_score >= 3:
            # Above average output needed
            return int(default_tokens + (max_tokens - default_tokens) * 0.6)
        elif expansion_score >= 1:
            # Slightly above default
            return int(default_tokens + (max_tokens - default_tokens) * 0.3)
        elif expansion_score <= -1:
            # Below default (brief responses)
            return min_tokens
        else:
            # Default allocation
            return default_tokens

    def _get_candidates(
        self, task_type: TaskType, require_streaming: bool
    ) -> list[ModelConfig]:
        """Get candidate models for a task type."""
        candidates = []

        for name, config in self.models.items():
            # Check task type support
            if config.task_types is not None and task_type not in config.task_types:
                continue

            # Check streaming requirement
            if require_streaming and not config.supports_streaming:
                continue

            # Check availability
            if not self._availability.get(name, True):
                # Check if we should retry
                last_fail = self._last_failure.get(name, 0)
                if time.time() - last_fail < 60:  # 1 minute cooldown
                    continue

            candidates.append(config)

        return candidates

    def _select_model(
        self,
        candidates: list[ModelConfig],
        complexity: str,
        max_latency_ms: float | None,
        preferred_tier: ModelTier | None,
        model_tier_ceiling: ModelTier | None = None,
    ) -> ModelConfig:
        """Select the best model from candidates."""
        # Map complexity to preferred tiers
        complexity_tiers = {
            "low": [ModelTier.LIGHTWEIGHT, ModelTier.STANDARD],
            "medium": [ModelTier.STANDARD, ModelTier.STRONG],
            "high": [ModelTier.STRONG, ModelTier.STANDARD],
        }

        preferred_tiers = complexity_tiers.get(complexity, [ModelTier.STANDARD])
        if preferred_tier:
            preferred_tiers.insert(0, preferred_tier)

        # Enforce hard model tier ceiling from compute budget
        if model_tier_ceiling is not None:
            ceiling_order = TIER_ORDER.get(model_tier_ceiling, 2)
            candidates = [
                c for c in candidates
                if TIER_ORDER.get(c.tier, 2) <= ceiling_order
            ]

        if not candidates:
            # No models within ceiling for this task — best-effort: pick
            # the smallest available model overall (minimise over-computation)
            all_sorted = sorted(
                self.models.values(),
                key=lambda c: TIER_ORDER.get(c.tier, 2),
            )
            return all_sorted[0]

        # Latency filter — keep ceiling-compliant set as fallback
        if max_latency_ms:
            latency_ok = [c for c in candidates if c.avg_latency_ms <= max_latency_ms]
            if latency_ok:
                candidates = latency_ok
            # else: keep all ceiling-compliant candidates, ignoring latency

        # Sort by tier preference
        def tier_score(config: ModelConfig) -> int:
            try:
                return preferred_tiers.index(config.tier)
            except ValueError:
                return len(preferred_tiers)

        candidates.sort(key=tier_score)
        return candidates[0]

    def _get_fallback(
        self, selected: ModelConfig, candidates: list[ModelConfig]
    ) -> ModelConfig | None:
        """Get fallback model if selected becomes unavailable."""
        for candidate in candidates:
            if candidate.model_id != selected.model_id:
                return candidate
        return None

    def _estimate_cost(self, query_length: int, config: ModelConfig) -> float:
        """Estimate cost for a query."""
        # Rough token estimation (4 chars per token)
        estimated_tokens = query_length // 4 + config.max_tokens // 2
        return (estimated_tokens / 1000) * config.cost_per_1k_tokens

    def mark_unavailable(self, model_name: str) -> None:
        """Mark a model as temporarily unavailable."""
        if model_name in self._availability:
            self._availability[model_name] = False
            self._last_failure[model_name] = time.time()
            logger.warning("Model marked unavailable: %s", model_name)

    def mark_available(self, model_name: str) -> None:
        """Mark a model as available."""
        if model_name in self._availability:
            self._availability[model_name] = True
            logger.info("Model marked available: %s", model_name)

    def get_model_config(self, model_name: str) -> ModelConfig | None:
        """Get configuration for a model."""
        return self.models.get(model_name)

    def get_available_models(self, task_type: TaskType | None = None) -> list[str]:
        """Get list of available model names."""
        result = []
        for name, config in self.models.items():
            if not self._availability.get(name, True):
                continue
            if task_type and config.task_types is not None and task_type not in config.task_types:
                continue
            result.append(name)
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        return {
            "total_models": len(self.models),
            "available_models": sum(1 for v in self._availability.values() if v),
            "unavailable_models": [
                name for name, avail in self._availability.items() if not avail
            ],
            "models_by_tier": {
                tier.value: [
                    name for name, cfg in self.models.items() if cfg.tier == tier
                ]
                for tier in ModelTier
            },
        }
