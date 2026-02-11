"""
Redis Cache Configuration (DEPRECATED)
========================================

.. deprecated:: 2.1.0
    This module is superseded by :mod:`backend.cache.multi_tier_cache`
    which provides the production 3-tier cache (L1 memory, L2 Redis, L3 SQLite).
    All production code paths use ``get_unified_cache()`` from multi_tier_cache.
    This file is retained for backward compatibility only.

Original design:
- TTL-based expiration (15 minutes default)
- Memory limit enforcement (256MB default)
- LRU eviction when memory limit reached
- Structured keys for different cache types
- Compression for large values
"""

import asyncio
import io
import json
import logging
import pickle
import zlib
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable
from typing import Any, Dict, List, Optional

from backend.utils.hashing import fast_hash

logger = logging.getLogger(__name__)


# SECURITY: Restricted unpickler to prevent arbitrary code execution
class RestrictedUnpickler(pickle.Unpickler):
    """
    Secure unpickler that only allows safe types.
    Prevents arbitrary code execution from malicious pickled data.
    """

    SAFE_MODULES = {
        "builtins": {
            "dict",
            "list",
            "set",
            "frozenset",
            "tuple",
            "str",
            "bytes",
            "int",
            "float",
            "bool",
            "type",
            "NoneType",
        },
        "numpy": {
            "ndarray",
            "dtype",
            "float32",
            "float64",
            "int32",
            "int64",
            "uint8",
            "bool_",
        },
        "numpy.core.multiarray": {"_reconstruct", "scalar"},
        "collections": {"OrderedDict", "defaultdict", "Counter"},
        "datetime": {"datetime", "date", "time", "timedelta", "timezone"},
        "uuid": {"UUID"},
    }

    def find_class(self, module: str, name: str):
        if module in self.SAFE_MODULES:
            allowed = self.SAFE_MODULES[module]
            if "*" in allowed or name in allowed:
                return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked unsafe class: {module}.{name}")


def safe_pickle_loads(data: bytes) -> Any:
    """Safely load pickled data using restricted unpickler."""
    return RestrictedUnpickler(io.BytesIO(data)).load()


class CacheType(Enum):
    """Types of cached data."""

    TOKENIZED = "tok"  # Pre-tokenized inputs
    EMBEDDING = "emb"  # Text embeddings
    SIMPLIFIED = "sim"  # Simplified text
    TRANSLATED = "tra"  # Translated text
    OCR_RESULT = "ocr"  # OCR results
    SEARCH = "src"  # Search results
    MODEL_OUTPUT = "out"  # General model outputs


@dataclass
class RedisCacheConfig:
    """Configuration for Redis caching (Principle P compliant)."""

    # Connection
    redis_url: str = "redis://localhost:6379/0"

    # Memory limits (Principle P: 256MB max)
    max_memory_mb: int = 256
    max_memory_policy: str = "allkeys-lru"  # LRU eviction

    # TTL settings (Principle P: 15 min)
    default_ttl_seconds: int = 900  # 15 minutes

    # Per-type TTL overrides
    ttl_by_type: dict[str, int] = field(
        default_factory=lambda: {
            CacheType.TOKENIZED.value: 300,  # 5 min - tokenization is fast
            CacheType.EMBEDDING.value: 1800,  # 30 min - embeddings are stable
            CacheType.SIMPLIFIED.value: 900,  # 15 min
            CacheType.TRANSLATED.value: 900,  # 15 min
            CacheType.OCR_RESULT.value: 3600,  # 1 hour - OCR is expensive
            CacheType.SEARCH.value: 600,  # 10 min
            CacheType.MODEL_OUTPUT.value: 900,  # 15 min
        }
    )

    # Compression
    compress_threshold_bytes: int = 1024  # Compress values > 1KB
    compression_level: int = 6

    # Key prefix
    key_prefix: str = "ssetu"

    # Connection pool
    max_connections: int = 20
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 2.0

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 0.1


class CacheSerializer:
    """Handles serialization and compression of cached values."""

    def __init__(self, config: RedisCacheConfig):
        self.config = config

    def serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        # Convert to bytes
        if isinstance(value, str):
            data = value.encode("utf-8")
            prefix = b"s:"  # String marker
        elif isinstance(value, bytes):
            data = value
            prefix = b"b:"  # Bytes marker
        elif isinstance(value, (dict, list)):
            data = json.dumps(value).encode("utf-8")
            prefix = b"j:"  # JSON marker
        else:
            data = pickle.dumps(value)
            prefix = b"p:"  # Pickle marker

        # Compress if above threshold
        if len(data) > self.config.compress_threshold_bytes:
            compressed = zlib.compress(data, self.config.compression_level)
            if len(compressed) < len(data):  # Only use if actually smaller
                return b"z:" + prefix + compressed

        return prefix + data

    def deserialize(self, data: bytes) -> Any:
        """Deserialize stored value."""
        if not data:
            return None

        # Check for compression
        if data.startswith(b"z:"):
            data = zlib.decompress(data[4:])  # Skip 'z:' and type prefix
            prefix = data[:2]
            data = data[2:]
        else:
            prefix = data[:2]
            data = data[2:]

        # Deserialize based on type
        if prefix == b"s:":
            return data.decode("utf-8")
        elif prefix == b"b:":
            return data
        elif prefix == b"j:":
            return json.loads(data.decode("utf-8"))
        elif prefix == b"p:":
            # SECURITY: Use safe unpickler
            return safe_pickle_loads(data)
        else:
            # Fallback - try string
            return data.decode("utf-8")


class RedisCache:
    """
    Redis cache client with Principle P compliance.

    Features:
    - 15 minute default TTL
    - 256MB memory limit with LRU eviction
    - Automatic compression
    - Structured key namespacing
    - Connection pooling
    """

    def __init__(self, config: RedisCacheConfig | None = None):
        self.config = config or RedisCacheConfig()
        self.serializer = CacheSerializer(self.config)
        self._redis: Any | None = None
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0, "bytes_saved": 0}

    async def connect(self) -> bool:
        """Initialize Redis connection."""
        try:
            import redis.asyncio as aioredis

            self._redis = await aioredis.from_url(
                self.config.redis_url,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=False,  # We handle encoding
            )

            # Configure memory limits (Principle P: 256MB)
            await self._redis.config_set("maxmemory", f"{self.config.max_memory_mb}mb")
            await self._redis.config_set(
                "maxmemory-policy", self.config.max_memory_policy
            )

            logger.info(
                f"Redis connected: {self.config.redis_url}, "
                f"max_memory={self.config.max_memory_mb}MB, "
                f"policy={self.config.max_memory_policy}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._stats["errors"] += 1
            return False

    async def disconnect(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis disconnected")

    def _make_key(self, key: str, cache_type: CacheType) -> str:
        """Create namespaced cache key."""
        return f"{self.config.key_prefix}:{cache_type.value}:{key}"

    def _hash_key(self, content: str) -> str:
        """Create hash-based key from content. Uses fast xxhash if available."""
        return fast_hash(content, length=16)

    def _get_ttl(self, cache_type: CacheType) -> int:
        """Get TTL for cache type."""
        return self.config.ttl_by_type.get(
            cache_type.value, self.config.default_ttl_seconds
        )

    async def get(
        self, key: str, cache_type: CacheType = CacheType.MODEL_OUTPUT
    ) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key
            cache_type: Type of cached data

        Returns:
            Cached value or None if not found
        """
        if not self._redis:
            return None

        full_key = self._make_key(key, cache_type)

        try:
            data = await self._redis.get(full_key)

            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return self.serializer.deserialize(data)

        except Exception as e:
            logger.error(f"Redis GET error for {full_key}: {e}")
            self._stats["errors"] += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        cache_type: CacheType = CacheType.MODEL_OUTPUT,
        ttl: int | None = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cached data
            ttl: Optional TTL override in seconds

        Returns:
            True if successful
        """
        if not self._redis:
            return False

        full_key = self._make_key(key, cache_type)
        ttl = ttl or self._get_ttl(cache_type)

        try:
            serialized = self.serializer.serialize(value)
            await self._redis.setex(full_key, ttl, serialized)

            self._stats["sets"] += 1
            self._stats["bytes_saved"] += len(serialized)
            return True

        except Exception as e:
            logger.error(f"Redis SET error for {full_key}: {e}")
            self._stats["errors"] += 1
            return False

    async def delete(
        self, key: str, cache_type: CacheType = CacheType.MODEL_OUTPUT
    ) -> bool:
        """Delete value from cache."""
        if not self._redis:
            return False

        full_key = self._make_key(key, cache_type)

        try:
            await self._redis.delete(full_key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error for {full_key}: {e}")
            return False

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[..., Any],
        cache_type: CacheType = CacheType.MODEL_OUTPUT,
        ttl: int | None = None,
    ) -> Any:
        """
        Get from cache or compute and cache result.

        Args:
            key: Cache key
            compute_fn: Async function to compute value if not cached
            cache_type: Type of cached data
            ttl: Optional TTL override

        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = await self.get(key, cache_type)
        if cached is not None:
            return cached

        # Compute value
        if asyncio.iscoroutinefunction(compute_fn):
            value = await compute_fn()
        else:
            value = compute_fn()

        # Cache result
        await self.set(key, value, cache_type, ttl)

        return value

    async def mget(
        self, keys: list[str], cache_type: CacheType = CacheType.MODEL_OUTPUT
    ) -> list[Any | None]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys
            cache_type: Type of cached data

        Returns:
            List of cached values (None for misses)
        """
        if not self._redis or not keys:
            return [None] * len(keys)

        full_keys = [self._make_key(k, cache_type) for k in keys]

        try:
            values = await self._redis.mget(*full_keys)
            results: list[Any | None] = []

            for v in values:
                if v is None:
                    self._stats["misses"] += 1
                    results.append(None)
                else:
                    self._stats["hits"] += 1
                    results.append(self.serializer.deserialize(v))

            return results

        except Exception as e:
            logger.error(f"Redis MGET error: {e}")
            self._stats["errors"] += 1
            return [None] * len(keys)

    async def mset(
        self,
        items: dict[str, Any],
        cache_type: CacheType = CacheType.MODEL_OUTPUT,
        ttl: int | None = None,
    ) -> bool:
        """
        Set multiple values in cache.

        Args:
            items: Dict of key -> value
            cache_type: Type of cached data
            ttl: Optional TTL override

        Returns:
            True if successful
        """
        if not self._redis or not items:
            return False

        ttl = ttl or self._get_ttl(cache_type)

        try:
            pipe = self._redis.pipeline()

            for key, value in items.items():
                full_key = self._make_key(key, cache_type)
                serialized = self.serializer.serialize(value)
                pipe.setex(full_key, ttl, serialized)
                self._stats["bytes_saved"] += len(serialized)

            await pipe.execute()
            self._stats["sets"] += len(items)
            return True

        except Exception as e:
            logger.error(f"Redis MSET error: {e}")
            self._stats["errors"] += 1
            return False

    async def clear_type(self, cache_type: CacheType) -> int:
        """
        Clear all entries of a specific type.

        Args:
            cache_type: Type to clear

        Returns:
            Number of keys deleted
        """
        if not self._redis:
            return 0

        pattern = f"{self.config.key_prefix}:{cache_type.value}:*"

        try:
            deleted = 0
            cursor = 0

            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break

            logger.info(f"Cleared {deleted} keys of type {cache_type.value}")
            return deleted

        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0

    async def get_info(self) -> dict[str, Any]:
        """Get Redis server info and statistics."""
        if not self._redis:
            return {"status": "disconnected", "stats": self._stats}

        try:
            info = await self._redis.info("memory")

            return {
                "status": "connected",
                "memory_used_mb": info.get("used_memory", 0) / (1024 * 1024),
                "memory_peak_mb": info.get("used_memory_peak", 0) / (1024 * 1024),
                "memory_limit_mb": self.config.max_memory_mb,
                "evicted_keys": info.get("evicted_keys", 0),
                "stats": self._stats,
                "hit_rate": self._stats["hits"]
                / max(self._stats["hits"] + self._stats["misses"], 1),
            }

        except Exception as e:
            logger.error(f"Redis info error: {e}")
            return {"status": "error", "error": str(e), "stats": self._stats}

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "hit_rate": self._stats["hits"] / max(total, 1),
            "total_requests": total,
        }


# Global cache instance with async lock for thread safety
_cache: RedisCache | None = None
_cache_lock = asyncio.Lock()


async def get_cache() -> RedisCache:
    """Get or create global Redis cache instance (async-safe with double-checked locking)."""
    global _cache

    if _cache is None:
        async with _cache_lock:
            # Double-check after acquiring lock
            if _cache is None:
                from backend.core.config import get_settings
        settings = get_settings()

        config = RedisCacheConfig(
            redis_url=settings.REDIS_URL,
            default_ttl_seconds=900,  # Principle P: 15 min
            max_memory_mb=256,  # Principle P: 256MB
        )

        _cache = RedisCache(config)
        await _cache.connect()

    return _cache


async def close_cache():
    """Close global cache connection."""
    global _cache
    if _cache:
        await _cache.disconnect()
        _cache = None


# Convenience functions
async def cache_get(key: str, cache_type: str = "out") -> Any | None:
    """Quick cache get."""
    cache = await get_cache()
    return await cache.get(key, CacheType(cache_type))


async def cache_set(
    key: str, value: Any, cache_type: str = "out", ttl: int | None = None
) -> bool:
    """Quick cache set."""
    cache = await get_cache()
    return await cache.set(key, value, CacheType(cache_type), ttl)


async def cache_embedding(text: str, embedding: list[float]) -> bool:
    """Cache text embedding."""
    cache = await get_cache()
    key = cache._hash_key(text)
    return await cache.set(key, embedding, CacheType.EMBEDDING)


async def get_cached_embedding(text: str) -> list[float] | None:
    """Get cached embedding for text."""
    cache = await get_cache()
    key = cache._hash_key(text)
    return await cache.get(key, CacheType.EMBEDDING)
