"""
ShikshaSetu Core Module
========================

Central configuration and utilities for the ShikshaSetu platform.

Structure:
- config.py: Application settings and environment configuration
- exceptions.py: Custom exception classes with retry decorators
- constants.py: Application-wide constants
- security.py: Security utilities
- storage.py: Redis/Memory storage backends
- policy.py: Content policy engine
- model_config.py: Hot-reloadable model configuration
- correlation.py: Request correlation ID logging
- circuit_breaker.py: Circuit breaker pattern for resilience
- tracing.py: OpenTelemetry distributed tracing
- optimized/: Apple Silicon M4 optimizations (DeviceRouter, RateLimiter, etc.)

For new code, prefer importing from backend.core.optimized:
    from backend.core.optimized import DeviceRouter, UnifiedRateLimiter
"""

from .config import Settings, get_settings, settings
from .constants import (
    DEFAULT_COMPLEXITY,
    MAX_COMPLEXITY,
    MIN_COMPLEXITY,
    SUPPORTED_LANGUAGES,
)
from .correlation import (
    CorrelatedLogger,
    CorrelationIdMiddleware,
    get_request_id,
    set_request_id,
    with_correlation,
)
from .exceptions import (
    AudioGenerationError,
    CircuitBreaker,
    CollaborationError,
    ModelLoadError,
    ModelTimeoutError,
    PipelineError,
    RetryConfig,
    ShikshaSetuException,
    SimplificationError,
    TranslationError,
    with_retry,
)

# Re-export optimized components for convenience
from .optimized import (
    ComputeBackend,
    DeviceRouter,
    DeviceTaskType,
    HardwareModelConfig,
    LoadedModel,
    ModelConfig,
    PerformanceConfig,
    PerformanceOptimizer,
    TaskType,
    UnifiedRateLimiter,
)
from .optimized import (
    # Model manager from optimized
    HighPerformanceModelManager as ModelRegistry,
)
from .optimized import (
    get_model_manager as get_model_registry,
)
# Canonical types
from .types import ModelTier, ModelType
from .storage import (
    HybridStorage,
    get_conversation_storage,
    get_rate_limit_storage,
    get_storage,
)

__all__ = [
    "DEFAULT_COMPLEXITY",
    "MAX_COMPLEXITY",
    "MIN_COMPLEXITY",
    "SUPPORTED_LANGUAGES",
    "AudioGenerationError",
    "CircuitBreaker",
    "CollaborationError",
    "ComputeBackend",
    "CorrelatedLogger",
    "CorrelationIdMiddleware",
    "DeviceRouter",
    "DeviceTaskType",
    "HardwareModelConfig",
    "HybridStorage",
    "LoadedModel",
    "ModelConfig",
    "ModelLoadError",
    "ModelRegistry",
    "ModelTier",
    "ModelTimeoutError",
    "ModelType",
    "PerformanceConfig",
    "PerformanceOptimizer",
    "PipelineError",
    "RetryConfig",
    "Settings",
    "ShikshaSetuException",
    "SimplificationError",
    "TaskType",
    "TranslationError",
    "UnifiedRateLimiter",
    "get_conversation_storage",
    "get_model_registry",
    "get_rate_limit_storage",
    "get_request_id",
    "get_settings",
    "get_storage",
    "set_request_id",
    "settings",
    "with_correlation",
    "with_retry",
]
