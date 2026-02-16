"""
Memory Manager — Phase 3
=========================

Separates and coordinates:
- Short-term memory  (in-flight conversation context, per-session)
- Long-term memory    (persisted embeddings, user history, knowledge)
- Retrieval system   (similarity search with caching)

Design:
- Avoids redundant embedding calls via content-hash deduplication
- Intelligent caching with TTL and access-frequency tracking
- Tiered retrieval: L1 cache → L2 vector index → L3 database
- All public methods are async; internal state is lock-protected
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ── Enums ────────────────────────────────────────────────────────────────────

class MemoryTier(StrEnum):
    SHORT_TERM = "short_term"      # Current conversation context
    LONG_TERM = "long_term"        # Persisted knowledge / embeddings
    RETRIEVAL = "retrieval"        # Search results cache

class RetrievalSource(StrEnum):
    CACHE_L1 = "cache_l1"          # In-memory recent results
    VECTOR_INDEX = "vector_index"  # pgvector / HNSW
    DATABASE = "database"          # Full-text / SQL

# ── Data Contracts ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """Single memory record."""

    key: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tier: MemoryTier = MemoryTier.SHORT_TERM
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_s: float = 0.0  # 0 = no expiry

@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Result from memory retrieval."""

    entries: tuple[MemoryEntry, ...]
    source: RetrievalSource
    latency_ms: float
    query_hash: str
    from_cache: bool = False

@dataclass
class MemoryStats:
    """Observable memory statistics."""

    short_term_entries: int = 0
    long_term_entries: int = 0
    retrieval_cache_entries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    embedding_calls_saved: int = 0
    total_retrievals: int = 0
    avg_retrieval_ms: float = 0.0
    _retrieval_latencies: list[float] = field(default_factory=list)

    def record_retrieval(self, latency_ms: float) -> None:
        self.total_retrievals += 1
        self._retrieval_latencies.append(latency_ms)
        # Rolling average over last 100
        recent = self._retrieval_latencies[-100:]
        self.avg_retrieval_ms = sum(recent) / len(recent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "short_term_entries": self.short_term_entries,
            "long_term_entries": self.long_term_entries,
            "retrieval_cache_entries": self.retrieval_cache_entries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "embedding_calls_saved": self.embedding_calls_saved,
            "total_retrievals": self.total_retrievals,
            "avg_retrieval_ms": round(self.avg_retrieval_ms, 2),
        }

# ── Protocols (dependency inversion) ────────────────────────────────────────

class EmbeddingProvider(Protocol):
    """Interface for embedding generation — implemented by existing services."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

class VectorSearchProvider(Protocol):
    """Interface for vector similarity search."""

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...

# ── Short-Term Memory ────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Per-session conversation context.

    - Bounded by max_messages to prevent context overflow
    - Supports context summarization when window exceeds threshold
    - No persistence — lives only for the session lifetime
    """

    def __init__(self, max_messages: int = 50, summarize_threshold: int = 30):
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._max_messages = max_messages
        self._summarize_threshold = summarize_threshold
        self._lock = asyncio.Lock()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []

            messages = self._sessions[session_id]
            messages.append({
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": time.time(),
            })

            # Evict oldest if over capacity
            if len(messages) > self._max_messages:
                self._sessions[session_id] = messages[-self._max_messages:]

    async def get_context(
        self,
        session_id: str,
        last_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get conversation context for a session."""
        async with self._lock:
            messages = self._sessions.get(session_id, [])
            if last_n:
                return list(messages[-last_n:])
            return list(messages)

    async def get_context_text(
        self,
        session_id: str,
        last_n: int | None = None,
    ) -> str:
        """Get context as a formatted text block."""
        messages = await self.get_context(session_id, last_n)
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        return "\n".join(parts)

    async def clear_session(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def session_count(self) -> int:
        async with self._lock:
            return len(self._sessions)

    async def total_messages(self) -> int:
        async with self._lock:
            return sum(len(msgs) for msgs in self._sessions.values())

    async def should_summarize(self, session_id: str) -> bool:
        """Check if session context should be summarized to save tokens."""
        async with self._lock:
            messages = self._sessions.get(session_id, [])
            return len(messages) >= self._summarize_threshold

    async def replace_with_summary(
        self,
        session_id: str,
        summary: str,
        keep_recent: int = 5,
    ) -> None:
        """Replace old messages with a summary, keeping recent ones."""
        async with self._lock:
            messages = self._sessions.get(session_id, [])
            recent = messages[-keep_recent:] if len(messages) > keep_recent else messages
            self._sessions[session_id] = [
                {
                    "role": "system",
                    "content": f"[Conversation summary]: {summary}",
                    "metadata": {"is_summary": True},
                    "timestamp": time.time(),
                },
                *recent,
            ]

# ── Long-Term Memory ────────────────────────────────────────────────────────

class LongTermMemory:
    """
    Persistent knowledge store with content-hash deduplication.

    - Deduplicates by content hash to avoid redundant embedding calls
    - Tracks access frequency for intelligent eviction
    - Delegates actual embedding to an external provider
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        max_entries: int = 50_000,
    ):
        self._provider = embedding_provider
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._content_hashes: dict[str, str] = {}  # content_hash → entry_key
        self._max_entries = max_entries
        self._lock = asyncio.Lock()
        self._embedding_calls_saved = 0

    async def store(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        embed: bool = True,
    ) -> MemoryEntry:
        """Store content, deduplicating embedding calls."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:20]

        async with self._lock:
            # Check for duplicate content
            existing_key = self._content_hashes.get(content_hash)
            if existing_key and existing_key in self._store:
                self._embedding_calls_saved += 1
                existing = self._store[existing_key]
                # Move to end (LRU) and return
                self._store.move_to_end(existing_key)
                return existing

            # Generate embedding if needed
            embedding = None
            if embed and self._provider:
                try:
                    embeddings = await self._provider.embed([content])
                    if embeddings:
                        embedding = embeddings[0]
                except (RuntimeError, ValueError, OSError) as e:
                    logger.warning("Embedding generation failed: %s", e)

            entry = MemoryEntry(
                key=key,
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                tier=MemoryTier.LONG_TERM,
            )

            self._store[key] = entry
            self._content_hashes[content_hash] = key

            # Evict if over capacity
            while len(self._store) > self._max_entries:
                evicted_key, _ = self._store.popitem(last=False)
                # Remove hash mapping
                self._content_hashes = {
                    h: k for h, k in self._content_hashes.items() if k != evicted_key
                }

            return entry

    async def get(self, key: str) -> MemoryEntry | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry:
                self._store.move_to_end(key)
            return entry

    async def search_by_text(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Simple text-based search (substring match) for fallback."""
        async with self._lock:
            query_lower = query.lower()
            scored = []
            for entry in self._store.values():
                if query_lower in entry.content.lower():
                    scored.append(entry)
            return scored[:top_k]

    async def entry_count(self) -> int:
        async with self._lock:
            return len(self._store)

    @property
    def embedding_calls_saved(self) -> int:
        return self._embedding_calls_saved

# ── Retrieval Cache ──────────────────────────────────────────────────────────

class RetrievalCache:
    """
    LRU cache for vector retrieval results.

    Keyed by query content hash → avoids re-embedding + re-searching
    for identical or near-identical queries.
    """

    def __init__(self, max_entries: int = 1024, default_ttl_s: float = 300.0):
        self._cache: OrderedDict[str, _CachedRetrieval] = OrderedDict()
        self._max_entries = max_entries
        self._default_ttl = default_ttl_s
        self._hits = 0
        self._misses = 0

    def get(self, query_hash: str) -> list[dict[str, Any]] | None:
        entry = self._cache.get(query_hash)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired():
            del self._cache[query_hash]
            self._misses += 1
            return None
        entry.access_count += 1
        self._cache.move_to_end(query_hash)
        self._hits += 1
        return entry.results

    def put(
        self,
        query_hash: str,
        results: list[dict[str, Any]],
        ttl_s: float | None = None,
    ) -> None:
        if len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
        self._cache[query_hash] = _CachedRetrieval(
            results=results,
            created_at=time.time(),
            ttl_s=ttl_s or self._default_ttl,
        )

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

@dataclass
class _CachedRetrieval:
    results: list[dict[str, Any]]
    created_at: float
    ttl_s: float
    access_count: int = 0

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_s

# ── Unified Memory Manager ──────────────────────────────────────────────────

class MemoryManager:
    """
    Unified facade coordinating all three memory tiers.

    Public API:
        - add_message()       → short-term
        - store_knowledge()   → long-term with dedup
        - retrieve()          → tiered retrieval (cache → vector → DB)
        - get_context()       → session context
        - get_stats()         → observability
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        vector_provider: VectorSearchProvider | None = None,
        short_term_max: int = 50,
        long_term_max: int = 50_000,
        cache_max: int = 1024,
        cache_ttl_s: float = 300.0,
    ):
        self.short_term = ShortTermMemory(max_messages=short_term_max)
        self.long_term = LongTermMemory(
            embedding_provider=embedding_provider,
            max_entries=long_term_max,
        )
        self.retrieval_cache = RetrievalCache(
            max_entries=cache_max,
            default_ttl_s=cache_ttl_s,
        )
        self._embedding_provider = embedding_provider
        self._vector_provider = vector_provider
        self._stats = MemoryStats()

    # ── Short-term ───────────────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.short_term.add_message(session_id, role, content, metadata)

    async def get_context(
        self,
        session_id: str,
        last_n: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self.short_term.get_context(session_id, last_n)

    async def get_context_text(
        self,
        session_id: str,
        last_n: int | None = None,
    ) -> str:
        return await self.short_term.get_context_text(session_id, last_n)

    # ── Long-term ────────────────────────────────────────────────────

    async def store_knowledge(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        entry = await self.long_term.store(key, content, metadata)
        self._stats.long_term_entries = await self.long_term.entry_count()
        return entry

    # ── Retrieval ────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Three-tier retrieval:
        1. L1: Check retrieval cache (< 0.1ms)
        2. L2: Vector similarity search (~2ms)
        3. L3: Database / text fallback (~20ms)
        """
        t0 = time.perf_counter()
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        # L1: Cache
        cached = self.retrieval_cache.get(query_hash)
        if cached is not None:
            latency = (time.perf_counter() - t0) * 1000
            self._stats.cache_hits += 1
            self._stats.record_retrieval(latency)
            return RetrievalResult(
                entries=tuple(
                    MemoryEntry(key=r.get("id", ""), content=r.get("text", ""))
                    for r in cached
                ),
                source=RetrievalSource.CACHE_L1,
                latency_ms=round(latency, 2),
                query_hash=query_hash,
                from_cache=True,
            )

        self._stats.cache_misses += 1

        # L2: Vector search
        if self._embedding_provider and self._vector_provider:
            try:
                embeddings = await self._embedding_provider.embed([query])
                if embeddings:
                    results = await self._vector_provider.search(
                        embeddings[0], top_k=top_k, filters=filters
                    )
                    # Cache the results
                    self.retrieval_cache.put(query_hash, results)
                    latency = (time.perf_counter() - t0) * 1000
                    self._stats.record_retrieval(latency)
                    return RetrievalResult(
                        entries=tuple(
                            MemoryEntry(
                                key=r.get("id", ""),
                                content=r.get("text", ""),
                                metadata=r.get("metadata", {}),
                            )
                            for r in results
                        ),
                        source=RetrievalSource.VECTOR_INDEX,
                        latency_ms=round(latency, 2),
                        query_hash=query_hash,
                    )
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Vector search failed, falling back: %s", e)

        # L3: Text fallback
        entries = await self.long_term.search_by_text(query, top_k)
        results_dicts = [{"id": e.key, "text": e.content} for e in entries]
        self.retrieval_cache.put(query_hash, results_dicts)
        latency = (time.perf_counter() - t0) * 1000
        self._stats.record_retrieval(latency)

        return RetrievalResult(
            entries=tuple(entries),
            source=RetrievalSource.DATABASE,
            latency_ms=round(latency, 2),
            query_hash=query_hash,
        )

    # ── Observability ────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        self._stats.short_term_entries = await self.short_term.total_messages()
        self._stats.long_term_entries = await self.long_term.entry_count()
        self._stats.retrieval_cache_entries = self.retrieval_cache.size
        self._stats.embedding_calls_saved = self.long_term.embedding_calls_saved
        return self._stats.to_dict()

    # ── Session Management ───────────────────────────────────────────

    async def clear_session(self, session_id: str) -> None:
        await self.short_term.clear_session(session_id)

    async def should_summarize(self, session_id: str) -> bool:
        return await self.short_term.should_summarize(session_id)

    async def summarize_context(
        self,
        session_id: str,
        summary: str,
        keep_recent: int = 5,
    ) -> None:
        await self.short_term.replace_with_summary(session_id, summary, keep_recent)
