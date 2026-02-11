"""
Shared API Dependencies
========================

Common lazy-loaded singletons and utilities used across multiple
route modules. Extracted to eliminate duplication.

Usage:
    from ..deps import get_ai_engine, get_pipeline, json_dumps
"""

from typing import Any

__all__ = [
    "get_ai_engine",
    "get_pipeline",
    "json_dumps",
]

# ==================== Lazy-Loaded Singletons ====================

_ai_engine = None
_pipeline_service = None


def get_ai_engine():
    """Get AI engine singleton (lazy-loaded).

    Thread-safe: the underlying get_ai_engine() uses its own lock.
    """
    global _ai_engine
    if _ai_engine is None:
        from ..services.ai_core.engine import get_ai_engine as _factory

        _ai_engine = _factory()
    return _ai_engine


def get_pipeline():
    """Get pipeline service singleton (lazy-loaded)."""
    global _pipeline_service
    if _pipeline_service is None:
        from ..services.pipeline import get_pipeline_service

        _pipeline_service = get_pipeline_service()
    return _pipeline_service


# ==================== JSON Serialization ====================

try:
    import orjson

    def json_dumps(data: dict[str, Any]) -> str:
        """Fast JSON serialization using orjson with stdlib fallback."""
        return orjson.dumps(data).decode("utf-8")

except ImportError:
    import json

    def json_dumps(data: dict[str, Any]) -> str:
        """JSON serialization using stdlib."""
        return json.dumps(data)
