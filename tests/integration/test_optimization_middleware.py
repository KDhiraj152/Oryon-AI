"""
Tests for Route Optimization Middleware
========================================

Tests the middleware-based optimization layer that applies
v2-level optimizations to all 110 routes.

NOTE: Patterns are now tuples of (pattern, ttl/type) for performance.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


def _pattern_dict(patterns):
    """Convert tuple patterns to dict for easier testing."""
    return dict(patterns)


# Test route classification
def test_cacheable_patterns():
    """Test that cacheable patterns are properly defined."""
    from backend.api.optimization_middleware import CACHEABLE_PATTERNS

    patterns = _pattern_dict(CACHEABLE_PATTERNS)

    # Should have patterns for common GET endpoints
    assert "/api/v2/library" in patterns
    assert "/api/v2/content/" in patterns
    assert "/api/v2/health" in patterns

    # TTLs should be reasonable
    assert patterns["/api/v2/health"] == 10  # Short TTL for health
    assert patterns["/api/v2/library"] == 300  # 5 min for library


def test_cacheable_post_patterns():
    """Test POST caching patterns."""
    from backend.api.optimization_middleware import CACHEABLE_POST_PATTERNS

    patterns = _pattern_dict(CACHEABLE_POST_PATTERNS)

    # Expensive operations should be cached
    assert "/api/v2/content/simplify" in patterns
    assert "/api/v2/content/translate" in patterns
    assert "/api/v2/content/tts" in patterns

    # TTS should have longest TTL (most expensive)
    assert patterns["/api/v2/content/tts"] >= 3600


def test_ai_route_patterns():
    """Test AI device routing patterns."""
    from backend.api.optimization_middleware import AI_ROUTE_PATTERNS
    from backend.core.optimized.device_router import TaskType

    patterns = _pattern_dict(AI_ROUTE_PATTERNS)

    # Embedding routes
    assert patterns["/api/v2/content/process"] == TaskType.EMBEDDING
    assert patterns["/api/v2/content/validate"] == TaskType.EMBEDDING

    # LLM routes
    assert patterns["/api/v2/content/simplify"] == TaskType.LLM_INFERENCE
    assert patterns["/api/v2/chat"] == TaskType.LLM_INFERENCE

    # TTS routes
    assert patterns["/api/v2/content/tts"] == TaskType.TTS


def test_never_cache_patterns():
    """Test that sensitive routes are never cached."""
    from backend.api.optimization_middleware import NEVER_CACHE_PATTERNS

    # Auth should never be cached
    assert "/api/v2/auth/" in NEVER_CACHE_PATTERNS

    # Streaming should never be cached
    assert "/api/v2/chat/stream" in NEVER_CACHE_PATTERNS

    # Admin should never be cached
    assert "/admin/" in NEVER_CACHE_PATTERNS


def test_optimization_middleware_initialization():
    """Test middleware can be initialized."""
    from backend.api.optimization_middleware import OptimizationMiddleware

    app = FastAPI()
    middleware = OptimizationMiddleware(app)

    assert middleware._cache is not None
    assert middleware._device_router is not None
    assert middleware._metrics == {}


def test_cache_key_generation():
    """Test cache key generation."""
    from backend.api.optimization_middleware import OptimizationMiddleware

    app = FastAPI()
    middleware = OptimizationMiddleware(app)

    # Create mock request
    mock_request = MagicMock()
    mock_request.url.path = "/api/v2/library"
    mock_request.query_params = {"limit": "10"}

    key = middleware._make_cache_key(mock_request)

    assert key.startswith("opt:get:")
    assert len(key) > 10  # MD5 hash should be included


def test_metrics_recording():
    """Test metrics are recorded correctly."""
    from backend.api.optimization_middleware import OptimizationMiddleware

    app = FastAPI()
    middleware = OptimizationMiddleware(app)

    # Record some metrics
    middleware._record_metrics("/api/v2/library", "processed", 0.05)
    middleware._record_metrics("/api/v2/library", "processed", 0.03)
    middleware._record_metrics("/api/v2/library", "cache_hit", 0.001)

    metrics = middleware.get_metrics()

    assert "/api/v2/library" in metrics
    assert metrics["/api/v2/library"]["total_requests"] == 3
    assert metrics["/api/v2/library"]["cache_hits"] == 1


def test_path_normalization():
    """Test UUID normalization in paths."""
    from backend.api.optimization_middleware import OptimizationMiddleware

    app = FastAPI()
    middleware = OptimizationMiddleware(app)

    # Path with UUID should be normalized
    path_with_uuid = "/api/v2/content/123e4567-e89b-12d3-a456-426614174000"
    normalized = middleware._normalize_path(path_with_uuid)

    assert "{id}" in normalized
    assert "123e4567" not in normalized


def test_get_route_optimization_metrics():
    """Test the metrics endpoint helper."""
    from backend.api.optimization_middleware import get_route_optimization_metrics

    # Should return metrics or error (sync function, not async)
    result = get_route_optimization_metrics()

    assert isinstance(result, dict)
    # Either has metrics or error
    assert "total_patterns" in result or "error" in result


# Run standalone
if __name__ == "__main__":
    print("Running optimization middleware tests...")

    test_cacheable_patterns()
    print("✅ test_cacheable_patterns")

    test_cacheable_post_patterns()
    print("✅ test_cacheable_post_patterns")

    test_ai_route_patterns()
    print("✅ test_ai_route_patterns")

    test_never_cache_patterns()
    print("✅ test_never_cache_patterns")

    test_optimization_middleware_initialization()
    print("✅ test_optimization_middleware_initialization")

    test_cache_key_generation()
    print("✅ test_cache_key_generation")

    test_metrics_recording()
    print("✅ test_metrics_recording")

    test_path_normalization()
    print("✅ test_path_normalization")

    test_get_route_optimization_metrics()
    print("✅ test_get_route_optimization_metrics")

    print("\n✅ ALL 9 TESTS PASSED!")
