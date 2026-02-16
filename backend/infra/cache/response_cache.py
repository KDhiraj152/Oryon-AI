"""
Response Cache - Semantic Caching for LLM Responses
=====================================================

Features:
- Exact match caching for deterministic responses (temp=0)
- Semantic similarity caching for similar queries
- TTL-based expiration
- Memory-efficient storage
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from backend.utils.hashing import fast_hash

logger = logging.getLogger(__name__)

@dataclass
class CachedResponse:
    """Cached LLM response."""

    response: str
    prompt_hash: str
    created_at: float
    access_count: int = 1
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ResponseCache:
    """
    LRU cache for LLM responses with TTL.

    Caches deterministic responses (temperature=0) to avoid
    redundant inference calls.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300,
        min_prompt_length: int = 10,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.min_prompt_length = min_prompt_length

        self._cache: OrderedDict[str, CachedResponse] = OrderedDict()
        self._lock = threading.Lock()

        # Statistics
        self.hits = 0
        self.misses = 0

    def _hash_prompt(self, prompt: str, **kwargs) -> str:
        """Create deterministic hash from prompt and params. Uses fast xxhash if available."""
        key_parts = [prompt]

        # Include relevant parameters that affect output
        for key in sorted(kwargs.keys()):
            if key in ("temperature", "max_tokens", "top_p", "model"):
                key_parts.append(f"{key}={kwargs[key]}")

        key_str = "|".join(key_parts)
        return fast_hash(key_str, length=32)

    def get(self, prompt: str, temperature: float = 0.0, **kwargs) -> str | None:
        """
        Get cached response if available.

        Caches for low temperature values (< 0.3) which are near-deterministic.
        For educational queries, even moderate temperature responses are useful.
        """
        # Skip caching for high randomness or short prompts
        if temperature > 0.5 or len(prompt) < self.min_prompt_length:
            return None

        # Include temperature range in key for cache segmentation
        temp_bucket = (
            0 if temperature < 0.3 else 1
        )  # Two buckets: deterministic / low-random
        key = self._hash_prompt(prompt, temperature=temp_bucket, **kwargs)

        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            cached = self._cache[key]

            # Check TTL
            if time.time() - cached.created_at > self.ttl_seconds:
                del self._cache[key]
                self.misses += 1
                return None

            # Update access stats and move to end (LRU)
            cached.access_count += 1
            self._cache.move_to_end(key)

            self.hits += 1
            logger.debug("Response cache HIT for prompt hash %s", key[:8])
            return cached.response

    def set(
        self, prompt: str, response: str, temperature: float = 0.0, **kwargs
    ) -> None:
        """Cache a response."""
        # Cache for reasonable temperature values
        if temperature > 0.5 or len(prompt) < self.min_prompt_length:
            return

        temp_bucket = 0 if temperature < 0.3 else 1
        key = self._hash_prompt(prompt, temperature=temp_bucket, **kwargs)

        with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CachedResponse(
                response=response,
                prompt_hash=key,
                created_at=time.time(),
                metadata=kwargs,
            )

    def clear(self) -> None:
        """Clear all cached responses."""
        with self._lock:
            self._cache.clear()
            logger.info("[ResponseCache] Cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "size": len(self._cache),
            "max_size": self.max_size,
        }

class SemanticResponseCache:
    """
    Semantic similarity-based response cache.

    Uses embedding similarity to find cached responses
    for semantically similar queries.
    """

    def __init__(
        self,
        max_size: int = 500,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 600,
    ):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        # Store: (prompt, response, embedding, timestamp)
        self._cache: list[tuple[str, str, Any, float]] = []
        self._lock = threading.Lock()

        # Embedding function (lazy loaded)
        self._embed_func = None

        # Statistics
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0

    def set_embedding_function(self, func):
        """Set the embedding function to use."""
        self._embed_func = func

    async def get(
        self,
        prompt: str,
        embedding: Any | None = None,
    ) -> str | None:
        """
        Get cached response by semantic similarity.

        Args:
            prompt: Query prompt
            embedding: Pre-computed embedding (optional)
        """
        import numpy as np

        if not self._cache:
            self.misses += 1
            return None

        # Get embedding if not provided
        if embedding is None:
            if self._embed_func is None:
                self.misses += 1
                return None
            try:
                embedding = await self._embed_func(prompt)
            except (OSError, ValueError, TypeError) as e:
                logger.warning("SemanticCache embedding error: %s", e)
                self.misses += 1
                return None

        # SIMD-optimized similarity
        try:
            from backend.core.optimized.simd_ops import cosine_similarity_single

            use_simd = True
        except ImportError:
            use_simd = False
            query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        best_match = None
        best_similarity = 0.0

        with self._lock:
            now = time.time()
            valid_entries = []

            for cached_prompt, cached_response, cached_emb, timestamp in self._cache:
                # Skip expired
                if now - timestamp > self.ttl_seconds:
                    continue

                valid_entries.append(
                    (cached_prompt, cached_response, cached_emb, timestamp)
                )

                # Exact match check first
                if cached_prompt == prompt:
                    self.exact_hits += 1
                    return cached_response

                # Semantic similarity (SIMD-optimized)
                if use_simd:
                    similarity = cosine_similarity_single(embedding, cached_emb)
                else:
                    cached_norm = cached_emb / (np.linalg.norm(cached_emb) + 1e-8)
                    similarity = float(np.dot(query_norm, cached_norm))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_response

            # Update cache (remove expired)
            self._cache = valid_entries

        if best_similarity >= self.similarity_threshold:
            self.semantic_hits += 1
            return best_match

        self.misses += 1
        return None

    async def set(
        self,
        prompt: str,
        response: str,
        embedding: Any | None = None,
    ) -> None:
        """Cache response with embedding."""

        # Get embedding if not provided
        if embedding is None:
            if self._embed_func is None:
                return
            try:
                embedding = await self._embed_func(prompt)
            except (OSError, ValueError, TypeError) as e:
                logger.warning("SemanticCache embedding error: %s", e)
                return

        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.pop(0)

            self._cache.append((prompt, response, embedding, time.time()))

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.exact_hits + self.semantic_hits + self.misses
        hit_rate = (self.exact_hits + self.semantic_hits) / total if total > 0 else 0.0

        return {
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "size": len(self._cache),
            "similarity_threshold": self.similarity_threshold,
        }

# Global singletons
_response_cache: ResponseCache | None = None
_semantic_cache: SemanticResponseCache | None = None

def get_response_cache(
    max_size: int = 1000,
    ttl_seconds: int = 300,
    semantic: bool = False,
) -> "ResponseCache | SemanticResponseCache":
    """Get response cache instance."""
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _response_cache, _semantic_cache

    if semantic:
        if _semantic_cache is not None:
            return _semantic_cache
        _semantic_cache = SemanticResponseCache(
            max_size=max_size // 2,
            ttl_seconds=ttl_seconds * 2,
        )
        return _semantic_cache

    if _response_cache is not None:
        return _response_cache
    _response_cache = ResponseCache(
        max_size=max_size,
        ttl_seconds=ttl_seconds,
    )
    return _response_cache
