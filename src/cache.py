"""
Simplified Redis cache - direct Redis usage.

Replaces: repository/redis_cache.py, repository/cache_manager.py
Simplified to essential caching functionality.
"""

import redis
import json
import os
import logging
from typing import Optional, Any, Dict
import hashlib

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

# Global Redis client
_redis_client: Optional[redis.Redis] = None


def get_redis() -> Optional[redis.Redis]:
    """Get or create Redis client."""
    global _redis_client
    
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            _redis_client.ping()
            logger.info(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            _redis_client = None
    
    return _redis_client


def is_redis_available() -> bool:
    """Check if Redis is available."""
    try:
        client = get_redis()
        return client is not None and client.ping()
    except Exception:
        return False


# =============================================================================
# CACHING FUNCTIONS
# =============================================================================

def cache_get(key: str) -> Optional[Any]:
    """Get value from cache."""
    try:
        client = get_redis()
        if not client:
            return None
        
        value = client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        return None


def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> bool:
    """Set value in cache with TTL."""
    try:
        client = get_redis()
        if not client:
            return False
        
        client.setex(key, ttl_seconds, json.dumps(value))
        return True
    except Exception as e:
        logger.error(f"Cache set error: {e}")
        return False


def cache_delete(key: str) -> bool:
    """Delete value from cache."""
    try:
        client = get_redis()
        if not client:
            return False
        
        client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Cache delete error: {e}")
        return False


def cache_delete_pattern(pattern: str) -> int:
    """Delete all keys matching pattern."""
    try:
        client = get_redis()
        if not client:
            return 0
        
        keys = client.keys(pattern)
        if keys:
            return client.delete(*keys)
        return 0
    except Exception as e:
        logger.error(f"Cache delete pattern error: {e}")
        return 0


# =============================================================================
# RATE LIMITING
# =============================================================================

def check_rate_limit(key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
    """
    Check rate limit using sliding window.
    
    Returns:
        (allowed, remaining_requests)
    """
    try:
        client = get_redis()
        if not client:
            return True, -1  # Allow if Redis unavailable
        
        import time
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Remove old entries
        client.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_count = client.zcard(key)
        
        if current_count >= limit:
            return False, 0
        
        # Add current request
        client.zadd(key, {str(current_time): current_time})
        client.expire(key, window_seconds)
        
        remaining = limit - (current_count + 1)
        return True, remaining
        
    except Exception as e:
        logger.error(f"Rate limit check error: {e}")
        return True, -1  # Allow on error


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_cache_key(*args) -> str:
    """Generate cache key from arguments."""
    key_string = ":".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cache_stats() -> Dict[str, Any]:
    """Get Redis statistics."""
    try:
        client = get_redis()
        if not client:
            return {'available': False}
        
        info = client.info()
        return {
            'available': True,
            'used_memory': info.get('used_memory_human', 'N/A'),
            'connected_clients': info.get('connected_clients', 0),
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0)
        }
    except Exception as e:
        return {'available': False, 'error': str(e)}


# Legacy compatibility
class RedisCache:
    """Legacy wrapper for backward compatibility"""
    
    def is_available(self):
        return is_redis_available()
    
    def get_cached_response(self, key):
        return cache_get(key)
    
    def set_cached_response(self, key, value, ttl_seconds=300):
        return cache_set(key, value, ttl_seconds)
    
    def check_rate_limit(self, key, limit, window):
        return check_rate_limit(key, limit, window)
    
    def get_stats(self):
        return get_cache_stats()


def get_redis_cache():
    """Legacy compatibility"""
    return RedisCache()


__all__ = [
    'get_redis',
    'is_redis_available',
    'cache_get',
    'cache_set',
    'cache_delete',
    'cache_delete_pattern',
    'check_rate_limit',
    'generate_cache_key',
    'get_cache_stats',
    'RedisCache',
    'get_redis_cache'
]
