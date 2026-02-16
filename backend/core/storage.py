"""
Persistent Storage Backends - Redis with In-Memory Fallback.

Production-ready storage for:
- Rate limiting
- Conversation history
- Session data
- Model caching

Automatically uses Redis when available, falls back to in-memory.
"""

import asyncio
import contextlib
import functools
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from typing import Any, TypeVar, cast

from .config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# =============================================================================
# ABSTRACT STORAGE INTERFACE
# =============================================================================

class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """Get value by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """Set value with optional TTL in seconds."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment value by amount."""
        pass

    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key."""
        pass

    @abstractmethod
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key (-1 if no TTL, -2 if not exists)."""
        pass

    @abstractmethod
    async def keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        pass

    @abstractmethod
    async def hget(self, name: str, key: str) -> str | None:
        """Get hash field."""
        pass

    @abstractmethod
    async def hset(self, name: str, key: str, value: str) -> bool:
        """Set hash field."""
        pass

    @abstractmethod
    async def hgetall(self, name: str) -> dict[str, str]:
        """Get all hash fields."""
        pass

    @abstractmethod
    async def lpush(self, key: str, *values: str) -> int:
        """Push to list head."""
        pass

    @abstractmethod
    async def rpush(self, key: str, *values: str) -> int:
        """Push to list tail."""
        pass

    @abstractmethod
    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        """Get list range."""
        pass

    @abstractmethod
    async def llen(self, key: str) -> int:
        """Get list length."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and release resources. Override in subclasses."""
        pass

# =============================================================================
# REDIS BACKEND
# =============================================================================

class RedisBackend(StorageBackend):
    """Redis storage backend using aioredis."""

    def __init__(self, url: str | None = None):
        self.url = url or settings.REDIS_URL
        self._client = None
        self._available: bool | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is not None:
            return self._client

        async with self._lock:
            if self._client is not None:
                return self._client

            try:
                import redis.asyncio as aioredis

                self._client = await aioredis.from_url(
                    self.url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                # Test connection
                await self._client.ping()
                self._available = True
                logger.info(
                    f"[RedisBackend] Connected to Redis at {self.url.split('@')[-1]}"
                )
                return self._client

            except (ImportError, OSError, ConnectionError) as e:
                self._available = False
                logger.warning("[RedisBackend] Failed to connect to Redis: %s", e)
                return None

    def is_available(self) -> bool:
        """Check if Redis is available."""
        if self._available is not None:
            return self._available

        try:
            import redis

            r = redis.from_url(self.url, socket_connect_timeout=2)
            r.ping()
            self._available = True
        except (ImportError, OSError, ConnectionError):
            self._available = False

        return self._available

    async def get(self, key: str) -> str | None:
        client = await self._get_client()
        if not client:
            return None
        try:
            result = await client.get(key)
            return cast("str | None", result)
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("[Redis] GET failed: %s", e)
            return None

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        client = await self._get_client()
        if not client:
            return False
        try:
            if ttl:
                await client.setex(key, ttl, value)
            else:
                await client.set(key, value)
            return True
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("[Redis] SET failed: %s", e)
            return False

    async def delete(self, key: str) -> bool:
        client = await self._get_client()
        if not client:
            return False
        try:
            await client.delete(key)
            return True
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("[Redis] DELETE failed: %s", e)
            return False

    async def exists(self, key: str) -> bool:
        client = await self._get_client()
        if not client:
            return False
        try:
            return cast(bool, await client.exists(key) > 0)
        except (ConnectionError, TimeoutError, OSError):
            return False

    async def incr(self, key: str, amount: int = 1) -> int:
        client = await self._get_client()
        if not client:
            return 0
        try:
            return cast(int, await client.incrby(key, amount))
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("[Redis] INCR failed: %s", e)
            return 0

    async def expire(self, key: str, ttl: int) -> bool:
        client = await self._get_client()
        if not client:
            return False
        try:
            return cast(bool, await client.expire(key, ttl))
        except (ConnectionError, TimeoutError, OSError):
            return False

    async def ttl(self, key: str) -> int:
        client = await self._get_client()
        if not client:
            return -2
        try:
            return cast(int, await client.ttl(key))
        except (ConnectionError, TimeoutError, OSError):
            return -2

    async def keys(self, pattern: str) -> list[str]:
        client = await self._get_client()
        if not client:
            return []
        try:
            return cast("list[str]", await client.keys(pattern))
        except (ConnectionError, TimeoutError, OSError):
            return []

    async def hget(self, name: str, key: str) -> str | None:
        client = await self._get_client()
        if not client:
            return None
        try:
            result = await client.hget(name, key)
            return cast("str | None", result)
        except (ConnectionError, TimeoutError, OSError):
            return None

    async def hset(self, name: str, key: str, value: str) -> bool:
        client = await self._get_client()
        if not client:
            return False
        try:
            await client.hset(name, key, value)
            return True
        except (ConnectionError, TimeoutError, OSError):
            return False

    async def hgetall(self, name: str) -> dict[str, str]:
        client = await self._get_client()
        if not client:
            return {}
        try:
            return cast("dict[str, str]", await client.hgetall(name))
        except (ConnectionError, TimeoutError, OSError):
            return {}

    async def lpush(self, key: str, *values: str) -> int:
        client = await self._get_client()
        if not client:
            return 0
        try:
            return cast(int, await client.lpush(key, *values))
        except (ConnectionError, TimeoutError, OSError):
            return 0

    async def rpush(self, key: str, *values: str) -> int:
        client = await self._get_client()
        if not client:
            return 0
        try:
            return cast(int, await client.rpush(key, *values))
        except (ConnectionError, TimeoutError, OSError):
            return 0

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        client = await self._get_client()
        if not client:
            return []
        try:
            return cast("list[str]", await client.lrange(key, start, stop))
        except (ConnectionError, TimeoutError, OSError):
            return []

    async def llen(self, key: str) -> int:
        client = await self._get_client()
        if not client:
            return 0
        try:
            return cast(int, await client.llen(key))
        except (ConnectionError, TimeoutError, OSError):
            return 0

    async def close(self) -> None:
        """Close the async Redis connection."""
        if self._client is not None:
            with contextlib.suppress(ConnectionError, TimeoutError, OSError):
                await self._client.aclose()
            self._client = None
            self._available = None

# =============================================================================
# IN-MEMORY BACKEND (FALLBACK)
# =============================================================================

@dataclass
class MemoryEntry:
    """Entry in memory storage with TTL tracking."""

    value: str
    expires_at: float | None = None  # Unix timestamp

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

class MemoryBackend(StorageBackend):
    """In-memory storage backend with TTL support."""

    MAX_KEYS = 100_000  # Prevent unbounded memory growth

    def __init__(self):
        self._data: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._hashes: dict[str, dict[str, str]] = defaultdict(dict)
        self._lists: dict[str, list[str]] = defaultdict(list)
        self._lock = asyncio.Lock()

        # Start cleanup task
        self._cleanup_interval = 60  # seconds
        self._last_cleanup = time.time()

        logger.info("[MemoryBackend] Initialized (fallback mode)")

    async def _cleanup_expired(self):
        """Remove expired keys."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        async with self._lock:
            expired = [k for k, v in self._data.items() if v.is_expired()]
            for k in expired:
                del self._data[k]
            self._last_cleanup = now

    def is_available(self) -> bool:
        return True

    async def close(self) -> None:
        """Close memory backend (no-op)."""
        self._data.clear()

    async def get(self, key: str) -> str | None:
        await self._cleanup_expired()
        async with self._lock:
            entry = self._data.get(key)
            if entry is None or entry.is_expired():
                if entry and entry.is_expired():
                    del self._data[key]
                return None
            return entry.value

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        async with self._lock:
            expires_at = time.time() + ttl if ttl else None
            self._data[key] = MemoryEntry(value=value, expires_at=expires_at)
            self._data.move_to_end(key)
            # Evict LRU entries if over limit
            while len(self._data) > self.MAX_KEYS:
                self._data.popitem(last=False)
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._data:
                del self._data[key]
            return True

    async def exists(self, key: str) -> bool:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._data[key]
                return False
            return True

    async def incr(self, key: str, amount: int = 1) -> int:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None or entry.is_expired():
                self._data[key] = MemoryEntry(value=str(amount))
                return amount

            try:
                new_val = int(entry.value) + amount
                entry.value = str(new_val)
                return new_val
            except ValueError:
                return 0

    async def expire(self, key: str, ttl: int) -> bool:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            entry.expires_at = time.time() + ttl
            return True

    async def ttl(self, key: str) -> int:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return -2
            if entry.expires_at is None:
                return -1
            remaining = int(entry.expires_at - time.time())
            return max(0, remaining)

    async def keys(self, pattern: str) -> list[str]:
        import fnmatch

        await self._cleanup_expired()
        async with self._lock:
            # Use fnmatchcase for case-sensitive matching (matches Redis behavior).
            # fnmatch.fnmatch uses os.path.normcase which is case-insensitive on macOS.
            return [k for k in self._data if fnmatch.fnmatchcase(k, pattern)]

    async def hget(self, name: str, key: str) -> str | None:
        async with self._lock:
            return self._hashes.get(name, {}).get(key)

    async def hset(self, name: str, key: str, value: str) -> bool:
        async with self._lock:
            self._hashes[name][key] = value
            return True

    async def hgetall(self, name: str) -> dict[str, str]:
        async with self._lock:
            return dict(self._hashes.get(name, {}))

    async def lpush(self, key: str, *values: str) -> int:
        async with self._lock:
            for v in reversed(values):
                self._lists[key].insert(0, v)
            return len(self._lists[key])

    async def rpush(self, key: str, *values: str) -> int:
        async with self._lock:
            self._lists[key].extend(values)
            return len(self._lists[key])

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        async with self._lock:
            lst = self._lists.get(key, [])
            if stop == -1:
                return lst[start:]
            return lst[start : stop + 1]

    async def llen(self, key: str) -> int:
        async with self._lock:
            return len(self._lists.get(key, []))

# =============================================================================
# HYBRID STORAGE (AUTO-SELECTS BACKEND)
# =============================================================================

class HybridStorage:
    """
    Hybrid storage that uses Redis when available, falls back to memory.

    Automatically detects Redis availability at startup.
    """

    def __init__(self, prefer_redis: bool = True):
        self._redis = RedisBackend() if prefer_redis else None
        self._memory = MemoryBackend()
        self._backend: StorageBackend = self._memory
        self._initialized = False

    async def initialize(self):
        """Initialize and select best backend."""
        if self._initialized:
            return

        if self._redis and self._redis.is_available():
            try:
                # Test async connection
                await self._redis.set("__test__", "1", ttl=1)
                self._backend = self._redis
                logger.info("[HybridStorage] Using Redis backend")
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning("[HybridStorage] Redis test failed, using memory: %s", e)
                self._backend = self._memory
        else:
            logger.info("[HybridStorage] Redis unavailable, using memory backend")
            self._backend = self._memory

        self._initialized = True

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    @property
    def is_redis(self) -> bool:
        return isinstance(self._backend, RedisBackend)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate all StorageBackend methods to the active backend.

        Wraps async methods to ensure initialize() is called before delegation.
        This replaces 17 hand-written delegation methods with a single proxy.
        """
        attr = getattr(self._backend, name)
        if callable(attr) and asyncio.iscoroutinefunction(attr):

            @functools.wraps(attr)
            async def _proxy(*args: Any, **kwargs: Any) -> Any:
                await self.initialize()
                return await attr(*args, **kwargs)

            return _proxy
        return attr

    async def close(self) -> None:
        """Close the active backend connection."""
        await self._backend.close()

# =============================================================================
# SPECIALIZED STORAGE SERVICES
# =============================================================================

class RateLimitStorage:
    """Rate limiting with sliding window using storage backend."""

    def __init__(self, storage: HybridStorage | None = None, prefix: str = "ratelimit"):
        self.storage = storage or HybridStorage()
        self.prefix = prefix

    async def check_rate_limit(
        self,
        client_id: str,
        endpoint: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        """
        Check if client is within rate limit.

        Returns:
            Tuple of (allowed, remaining_requests, reset_in_seconds)
        """
        key = f"{self.prefix}:{client_id}:{endpoint}"

        # Get current count
        count_str = await self.storage.get(key)
        current_count = int(count_str) if count_str else 0

        # Get TTL for reset time
        ttl = await self.storage.ttl(key)
        reset_in = max(0, ttl) if ttl > 0 else window_seconds

        if current_count >= max_requests:
            return False, 0, reset_in

        # Increment
        if current_count == 0:
            await self.storage.set(key, "1", ttl=window_seconds)
            new_count = 1
        else:
            new_count = await self.storage.incr(key)

        remaining = max(0, max_requests - new_count)
        return True, remaining, reset_in

class ConversationStorage:
    """Persistent conversation storage."""

    def __init__(self, storage: HybridStorage | None = None, prefix: str = "conv"):
        self.storage = storage or HybridStorage()
        self.prefix = prefix
        self.ttl = 86400 * 7  # 7 days

    async def create_conversation(
        self,
        conversation_id: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Create a new conversation."""
        key = f"{self.prefix}:{conversation_id}"
        data = {
            "id": conversation_id,
            "user_id": user_id,
            "created_at": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }
        result = await self.storage.set(key, json.dumps(data), ttl=self.ttl)
        return bool(result)

    async def get_conversation(self, conversation_id: str) -> dict | None:
        """Get conversation metadata."""
        key = f"{self.prefix}:{conversation_id}"
        data = await self.storage.get(key)
        if data:
            return cast("dict[Any, Any]", json.loads(data))
        return None

    async def add_message(
        self,
        conversation_id: str,
        message_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add message to conversation."""
        key = f"{self.prefix}:{conversation_id}:messages"
        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }
        count = await self.storage.rpush(key, json.dumps(message))
        await self.storage.expire(key, self.ttl)
        return bool(count > 0)

    async def get_messages(
        self,
        conversation_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get conversation messages."""
        key = f"{self.prefix}:{conversation_id}:messages"
        messages = await self.storage.lrange(key, -limit, -1)
        return [json.loads(m) for m in messages]

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and messages."""
        await self.storage.delete(f"{self.prefix}:{conversation_id}")
        await self.storage.delete(f"{self.prefix}:{conversation_id}:messages")
        return True

# =============================================================================
# SINGLETON INSTANCES (Thread-safe with double-checked locking)
# =============================================================================

_storage_instance: HybridStorage | None = None
_rate_limit_instance: RateLimitStorage | None = None
_conversation_instance: ConversationStorage | None = None

def get_storage() -> HybridStorage:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance
    _storage_instance = HybridStorage()
    return _storage_instance
def get_rate_limit_storage() -> RateLimitStorage:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _rate_limit_instance
    if _rate_limit_instance is not None:
        return _rate_limit_instance
    _rate_limit_instance = RateLimitStorage(get_storage())
    return _rate_limit_instance
def get_conversation_storage() -> ConversationStorage:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _conversation_instance
    if _conversation_instance is not None:
        return _conversation_instance
    _conversation_instance = ConversationStorage(get_storage())
    return _conversation_instance
# Export
__all__ = [
    "ConversationStorage",
    "HybridStorage",
    "MemoryBackend",
    "RateLimitStorage",
    "RedisBackend",
    "StorageBackend",
    "get_conversation_storage",
    "get_rate_limit_storage",
    "get_storage",
]
