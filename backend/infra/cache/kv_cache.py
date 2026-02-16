"""
KV Cache Manager - Efficient Key-Value Cache for Transformers
==============================================================

Manages KV cache for transformer models to enable:
- Faster generation (no recomputation)
- Memory-efficient long contexts
- Cache persistence across requests

M4 Optimizations:
- Memory budget aware (uses MLX allocation from M4_MEMORY_BUDGET)
- 8192 max sequence length for M4's unified memory
- Efficient eviction aligned with GPU memory pressure
"""

import contextlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from backend.utils.hashing import fast_hash

logger = logging.getLogger(__name__)

# Import M4 memory budget if available
try:
    from backend.core.optimized.device_router import M4_MEMORY_BUDGET, get_device_router

    _HAS_M4_ROUTER = True
except ImportError:
    _HAS_M4_ROUTER = False
    M4_MEMORY_BUDGET = {}

def get_m4_kv_memory_limit() -> float:
    """Get optimal KV cache memory limit for M4."""
    if _HAS_M4_ROUTER:
        try:
            router = get_device_router()
            if router.capabilities.is_m4:
                # Use 1GB of the 4GB MLX budget for KV cache
                mlx_budget = M4_MEMORY_BUDGET.get("mlx_llm", 4.0)
                return mlx_budget * 0.25  # 1GB for KV cache
        except (RuntimeError, OSError):
            pass
    return 2.0  # Default 2GB

@dataclass
class KVCacheEntry:
    """Single KV cache entry."""

    cache_tensors: Any  # The actual KV tensors
    sequence_length: int
    created_at: float
    last_accessed: float
    access_count: int = 1
    model_id: str = ""

    def _tensor_memory_bytes(self, tensor: Any) -> int:
        """Calculate memory bytes for a single tensor."""
        if hasattr(tensor, "element_size"):
            return int(tensor.element_size() * tensor.nelement())
        return 0

    def _nested_tensor_memory_bytes(self, item: Any) -> int:
        """Calculate memory bytes for potentially nested tensor structure."""
        if hasattr(item, "element_size"):
            return self._tensor_memory_bytes(item)
        if isinstance(item, (list, tuple)):
            return sum(
                self._tensor_memory_bytes(t) for t in item if hasattr(t, "element_size")
            )
        return 0

    @property
    def memory_bytes(self) -> int:
        """Estimate memory usage."""
        if self.cache_tensors is None:
            return 0
        try:
            # For PyTorch tensors with direct element_size
            if hasattr(self.cache_tensors, "element_size"):
                return sum(
                    self._tensor_memory_bytes(t)
                    for layer in self.cache_tensors
                    for t in layer
                    if hasattr(t, "element_size")
                )
            # For tuple/list of tensors
            if isinstance(self.cache_tensors, (list, tuple)):
                return sum(
                    self._nested_tensor_memory_bytes(item)
                    for item in self.cache_tensors
                )
        except (OSError, ValueError, TypeError):
            pass
        return 0

class KVCacheManager:
    """
    Manages KV cache for multiple conversations/requests.

    Features:
    - LRU eviction based on memory limit
    - Cache key based on prefix tokens
    - Partial cache reuse (prompt caching)

    M4 Optimizations:
    - Memory limit from M4_MEMORY_BUDGET
    - 8192 max sequence for unified memory
    """

    def __init__(
        self,
        max_memory_gb: float | None = None,
        max_entries: int = 100,
        max_sequence_length: int = 8192,
    ):
        # Auto-detect M4 memory limit
        if max_memory_gb is None:
            max_memory_gb = get_m4_kv_memory_limit()

        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.max_entries = max_entries
        self.max_sequence_length = max_sequence_length

        # Cache storage
        self._cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._current_memory = 0

        # Track M4 optimization
        self._is_m4 = False
        if _HAS_M4_ROUTER:
            with contextlib.suppress(Exception):
                self._is_m4 = get_device_router().capabilities.is_m4

        logger.info(
            f"[KVCacheManager] Init: {max_memory_gb:.1f}GB, seq={max_sequence_length}, M4={self._is_m4}"
        )

    def _make_key(self, prefix_tokens: list[int], model_id: str = "") -> str:
        """Create cache key from prefix tokens. Uses fast xxhash if available."""
        # Use hash of token sequence for efficiency
        token_str = ",".join(str(t) for t in prefix_tokens[:512])  # Limit key size
        key_input = f"{model_id}:{token_str}"
        return fast_hash(key_input, length=32)

    def get(
        self,
        prefix_tokens: list[int],
        model_id: str = "",
    ) -> tuple[Any, int] | None:
        """
        Get cached KV for prefix tokens.

        Returns:
            Tuple of (kv_cache, cached_length) or None
        """
        key = self._make_key(prefix_tokens, model_id)

        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[key]
            entry.last_accessed = time.time()
            entry.access_count += 1

            # Move to end (LRU)
            self._cache.move_to_end(key)

            self.hits += 1
            return (entry.cache_tensors, entry.sequence_length)

    def set(
        self,
        prefix_tokens: list[int],
        kv_cache: Any,
        sequence_length: int,
        model_id: str = "",
    ) -> bool:
        """
        Store KV cache for prefix tokens.

        Returns:
            True if stored successfully
        """
        if sequence_length > self.max_sequence_length:
            logger.debug("KV cache sequence too long: %s", sequence_length)
            return False

        key = self._make_key(prefix_tokens, model_id)

        with self._lock:
            # Create entry
            entry = KVCacheEntry(
                cache_tensors=kv_cache,
                sequence_length=sequence_length,
                created_at=time.time(),
                last_accessed=time.time(),
                model_id=model_id,
            )

            entry_memory = entry.memory_bytes

            # Evict if necessary
            self._evict_if_needed(entry_memory)

            # Remove existing if present
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._current_memory -= old_entry.memory_bytes

            # Add new entry
            self._cache[key] = entry
            self._current_memory += entry_memory

            return True

    def _evict_if_needed(self, needed_bytes: int) -> int:
        """Evict entries to make room. Returns number evicted."""
        evicted = 0

        # Evict by count
        while len(self._cache) >= self.max_entries:
            _, entry = self._cache.popitem(last=False)
            self._current_memory -= entry.memory_bytes
            evicted += 1
            self.evictions += 1

        # Evict by memory
        while (
            self._current_memory + needed_bytes > self.max_memory_bytes and self._cache
        ):
            _, entry = self._cache.popitem(last=False)
            self._current_memory -= entry.memory_bytes
            evicted += 1
            self.evictions += 1

        if evicted > 0:
            logger.debug("KV cache evicted %s entries", evicted)

        return evicted

    def find_longest_prefix_match(
        self,
        tokens: list[int],
        model_id: str = "",
    ) -> tuple[Any, int] | None:
        """
        Find longest cached prefix that matches input tokens.

        Uses binary search over prefix lengths for O(log n * hash_cost)
        instead of linear O(n * hash_cost) scan.
        Useful for prompt caching - reuse KV from common prefixes.
        """
        if not tokens or not self._cache:
            return None

        # Binary search: find the longest prefix length that has a cache hit
        lo, hi = 1, len(tokens)
        best_result = None
        best_length = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            prefix = tokens[:mid]
            result = self.get(prefix, model_id)
            if result is not None:
                best_result = result
                best_length = mid
                lo = mid + 1  # Try longer
            else:
                hi = mid - 1  # Try shorter

        if best_result is not None:
            logger.debug("KV cache prefix match at length %s", best_length)
        return best_result

    def clear(self) -> None:
        """Clear all cached KV states."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            logger.info("[KVCacheManager] Cleared")

    def clear_model(self, model_id: str) -> int:
        """Clear cache for specific model."""
        removed = 0
        with self._lock:
            keys_to_remove = [
                k for k, v in self._cache.items() if v.model_id == model_id
            ]
            for key in keys_to_remove:
                entry = self._cache.pop(key)
                self._current_memory -= entry.memory_bytes
                removed += 1
        return removed

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "evictions": self.evictions,
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "memory_mb": self._current_memory / (1024**2),
            "max_memory_mb": self.max_memory_bytes / (1024**2),
            "m4_optimized": self._is_m4,
        }

# Global singleton
_kv_cache_manager: KVCacheManager | None = None

def get_kv_cache_manager(
    max_memory_gb: float | None = None,
) -> KVCacheManager:
    """Get global KV cache manager with M4 auto-sizing."""
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _kv_cache_manager
    if _kv_cache_manager is not None:
        return _kv_cache_manager
    _kv_cache_manager = KVCacheManager(max_memory_gb=max_memory_gb)
    return _kv_cache_manager
