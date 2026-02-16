"""
Cache Module

Provides caching functionality with Redis and local backends.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Redis client singleton (lock-free benign-race pattern)
_redis_client: Any | None = None

def get_redis():
    """
    Get Redis client for caching and rate limiting (lock-free singleton).

    Returns a synchronous Redis client for use in middleware.
    """
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    try:
        import redis
        from redis.connection import ConnectionPool

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # Use connection pool for efficient connection reuse
        pool = ConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=2.0,
            max_connections=20,
            retry_on_timeout=True,
        )

        _redis_client = redis.Redis(connection_pool=pool)

        # Test connection
        _redis_client.ping()
        logger.info("Redis connected with connection pool: %s", redis_url)

        return _redis_client

    except ImportError:
        logger.warning("Redis package not installed. Caching disabled.")
        return None
    except (OSError, ValueError, TypeError) as e:
        logger.warning("Redis connection failed: %s. Caching disabled.", e)
        return None

def close_redis():
    """Close Redis connection and release pool."""
    global _redis_client
    if _redis_client is None:
        return
    try:
        _redis_client.connection_pool.disconnect()
        _redis_client.close()
    except (OSError, ValueError, TypeError):
        pass
    _redis_client = None

# Re-export cache utilities
from .embedding_cache import (
    EmbeddingCache,
    get_embedding_cache,
)
from .kv_cache import (
    KVCacheManager,
    get_kv_cache_manager,
)
from .multi_tier_cache import (
    CacheConfig,
    CacheStats,
    CacheTier,
    UnifiedCache,
    get_unified_cache,
)
from .response_cache import (
    ResponseCache,
    SemanticResponseCache,
    get_response_cache,
)

__all__ = [
    # Unified cache exports
    "CacheConfig",
    "CacheStats",
    "CacheTier",
    "EmbeddingCache",
    "KVCacheManager",
    "ResponseCache",
    "SemanticResponseCache",
    "UnifiedCache",
    "close_redis",
    "get_embedding_cache",
    "get_kv_cache_manager",
    "get_redis",
    "get_response_cache",
    "get_unified_cache",
]
