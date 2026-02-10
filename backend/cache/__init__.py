"""
Redis Cache Module

Provides caching functionality with Redis backend.
Implements Principle P: TTL=15min, Max size=256MB.
"""

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Redis client singleton with proper thread safety
_redis_client: Any | None = None
_redis_lock = threading.Lock()


def get_redis():
    """
    Get Redis client for caching and rate limiting (thread-safe singleton).

    Returns a synchronous Redis client for use in middleware.
    For async operations, use get_cache() from redis_cache.py.
    """
    global _redis_client

    # Fast path: already initialized
    if _redis_client is not None:
        return _redis_client

    # Slow path: need to initialize with lock
    with _redis_lock:
        # Double-check after acquiring lock
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
                max_connections=20,  # Match RedisCacheConfig.max_connections
                retry_on_timeout=True,
            )

            _redis_client = redis.Redis(connection_pool=pool)

            # Test connection
            _redis_client.ping()
            logger.info(f"Redis connected with connection pool: {redis_url}")

            return _redis_client

        except ImportError:
            logger.warning("Redis package not installed. Caching disabled.")
            return None
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            return None


def close_redis():
    """Close Redis connection and pool."""
    global _redis_client
    with _redis_lock:
        if _redis_client is not None:
            try:
                _redis_client.connection_pool.disconnect()
                _redis_client.close()
            except Exception:
                pass
            _redis_client = None


# Re-export async cache utilities
# Re-export unified cache utilities (formerly cache)
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
from .redis_cache import (
    CacheType,
    RedisCache,
    RedisCacheConfig,
    cache_embedding,
    cache_get,
    cache_set,
    close_cache,
    get_cache,
    get_cached_embedding,
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
    "CacheType",
    "EmbeddingCache",
    "KVCacheManager",
    "RedisCache",
    "RedisCacheConfig",
    "ResponseCache",
    "SemanticResponseCache",
    "UnifiedCache",
    "cache_embedding",
    "cache_get",
    "cache_set",
    "close_cache",
    "close_redis",
    "get_cache",
    "get_cached_embedding",
    "get_embedding_cache",
    "get_kv_cache_manager",
    "get_redis",
    "get_response_cache",
    "get_unified_cache",
]
