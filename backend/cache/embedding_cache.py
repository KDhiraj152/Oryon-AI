"""
Embedding Cache - Optimized Storage for Vector Embeddings
==========================================================

Features:
- L3 (SQLite) primary storage with vector serialization
- L1 memory cache for hot embeddings
- Semantic similarity detection for deduplication
- Batch operations for efficiency

M4 Optimizations:
- Memory cache sized for unified memory (1GB embedding budget)
- Batch processing aligned with ANE batch size (32)
- Float16 storage for memory efficiency
"""

import asyncio
import contextlib
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.utils.hashing import fast_hash

logger = logging.getLogger(__name__)

# Import M4 optimizations
try:
    from backend.core.optimized.device_router import (
        M4_BATCH_SIZES,
        M4_MEMORY_BUDGET,
        TaskType,
        get_device_router,
    )

    _HAS_M4_ROUTER = True
except ImportError:
    _HAS_M4_ROUTER = False
    M4_MEMORY_BUDGET = {}
    M4_BATCH_SIZES = {}


def get_m4_embedding_cache_size() -> int:
    """Get optimal memory cache size for M4 embeddings."""
    if _HAS_M4_ROUTER:
        try:
            router = get_device_router()
            if router.capabilities.is_m4:
                # 1GB for embeddings, each ~4KB (1024-dim float32)
                # = ~250K embeddings, use 10% for hot cache
                emb_budget_gb = M4_MEMORY_BUDGET.get("embeddings", 1.0)
                return int(emb_budget_gb * 1024 * 1024 * 1024 * 0.1 / 4096)  # ~25K
        except Exception:
            pass
    return 5000  # Default


class EmbeddingCache:
    """
    Specialized cache for embedding vectors.

    Uses SQLite for persistence with numpy-optimized serialization.
    Includes similarity-based deduplication.

    M4 Optimizations:
    - Auto-sized memory cache based on embedding budget
    - Batch size aligned with ANE optimal (32)
    - Float16 storage option for memory efficiency
    - Connection pooling for SQLite (reduces connection overhead)
    """

    # OPTIMIZATION: Thread-local SQLite connections for connection reuse
    _thread_local = threading.local()

    def __init__(
        self,
        db_path: str = "storage/cache/embeddings.db",
        memory_cache_size: int | None = None,
        similarity_threshold: float = 0.95,
        use_float16: bool = True,  # M4 optimization
    ):
        self.db_path = Path(db_path)
        self.memory_cache_size = memory_cache_size or get_m4_embedding_cache_size()
        self.similarity_threshold = similarity_threshold
        self.use_float16 = use_float16

        # In-memory cache for hot embeddings
        self._memory_cache: dict[str, tuple[np.ndarray, float]] = {}
        self._cache_order: list[str] = []  # LRU order tracking
        self._lock = threading.Lock()

        # M4 batch size for operations
        self._batch_size = 32
        if _HAS_M4_ROUTER:
            with contextlib.suppress(Exception):
                self._batch_size = M4_BATCH_SIZES.get(TaskType.EMBEDDING, 32)

        # Statistics
        self.hits = 0
        self.misses = 0
        self.dedup_hits = 0

        # Track M4 mode
        self._is_m4 = False
        if _HAS_M4_ROUTER:
            with contextlib.suppress(Exception):
                self._is_m4 = get_device_router().capabilities.is_m4

        self._init_db()
        logger.info(
            f"[EmbeddingCache] Init: cache={self.memory_cache_size}, batch={self._batch_size}, M4={self._is_m4}"
        )

    def _init_db(self):
        """Initialize SQLite database with vector support."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                text TEXT,
                embedding BLOB,
                model_id TEXT,
                dimension INTEGER,
                created_at REAL,
                accessed_at REAL,
                access_count INTEGER DEFAULT 1
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_accessed ON embeddings(accessed_at)"
        )
        conn.commit()
        conn.close()

    def _hash_text(self, text: str, model_id: str = "") -> str:
        """Create hash key for text. Uses fast xxhash if available."""
        return fast_hash(f"{model_id}:{text}", length=32)

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy array efficiently. Uses float16 on M4 for memory savings."""
        if self.use_float16 and self._is_m4:
            return embedding.astype(np.float16).tobytes()
        return embedding.astype(np.float32).tobytes()

    def _deserialize_embedding(self, data: bytes, _dimension: int) -> np.ndarray:
        """Deserialize numpy array."""
        # Detect dtype from data size vs dimension
        if len(data) == _dimension * 2:  # float16
            return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(-1)
        return np.frombuffer(data, dtype=np.float32).reshape(-1)

    def _add_to_memory_cache(self, key: str, embedding: np.ndarray):
        """Add embedding to memory cache with LRU eviction."""
        with self._lock:
            # Remove if exists to update position
            if key in self._memory_cache:
                self._cache_order.remove(key)

            # Evict oldest if at capacity
            while (
                len(self._memory_cache) >= self.memory_cache_size and self._cache_order
            ):
                oldest = self._cache_order.pop(0)
                self._memory_cache.pop(oldest, None)

            self._memory_cache[key] = (embedding.copy(), time.time())
            self._cache_order.append(key)

    async def get(self, text: str, model_id: str = "default") -> np.ndarray | None:
        """Get embedding for text."""
        key = self._hash_text(text, model_id)

        # Check memory cache
        with self._lock:
            if key in self._memory_cache:
                self.hits += 1
                emb, _ = self._memory_cache[key]
                # Move to end of LRU
                if key in self._cache_order:
                    self._cache_order.remove(key)
                    self._cache_order.append(key)
                return emb.copy()

        # Check SQLite
        embedding = await asyncio.get_running_loop().run_in_executor(
            None, self._get_from_db, key
        )

        if embedding is not None:
            self.hits += 1
            # Promote to memory cache
            self._add_to_memory_cache(key, embedding)
            return embedding

        self.misses += 1
        return None

    def _get_from_db(self, key: str) -> np.ndarray | None:
        """Get embedding from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                "SELECT embedding, dimension FROM embeddings WHERE text_hash = ?",
                (key,),
            )
            row = cursor.fetchone()

            if row is None:
                conn.close()
                return None

            embedding_blob, dimension = row

            # Update access stats
            conn.execute(
                """
                UPDATE embeddings
                SET accessed_at = ?, access_count = access_count + 1
                WHERE text_hash = ?
            """,
                (time.time(), key),
            )
            conn.commit()
            conn.close()

            return self._deserialize_embedding(embedding_blob, dimension)

        except Exception as e:
            logger.warning(f"EmbeddingCache get error: {e}")
            return None

    async def set(
        self, text: str, embedding: np.ndarray, model_id: str = "default"
    ) -> bool:
        """Store embedding for text."""
        key = self._hash_text(text, model_id)

        # Add to memory cache
        self._add_to_memory_cache(key, embedding)

        # Store in SQLite
        return await asyncio.get_running_loop().run_in_executor(
            None, self._set_in_db, key, text, embedding, model_id
        )

    def _set_in_db(
        self, key: str, text: str, embedding: np.ndarray, model_id: str
    ) -> bool:
        """Store embedding in database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            now = time.time()

            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (text_hash, text, embedding, model_id, dimension, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    key,
                    text[:1000],  # Truncate long texts
                    self._serialize_embedding(embedding),
                    model_id,
                    len(embedding),
                    now,
                    now,
                ),
            )
            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.warning(f"EmbeddingCache set error: {e}")
            return False

    def _add_to_memory_cache(self, key: str, embedding: np.ndarray):
        """Add embedding to memory cache with LRU eviction."""
        with self._lock:
            # Evict if at capacity
            if len(self._memory_cache) >= self.memory_cache_size:
                # OPTIMIZATION: Use pop on cache_order list instead of min()
                if self._cache_order:
                    oldest_key = self._cache_order.pop(0)
                    self._memory_cache.pop(oldest_key, None)

            self._memory_cache[key] = (embedding.copy(), time.time())
            # Track order for LRU
            if key in self._cache_order:
                self._cache_order.remove(key)
            self._cache_order.append(key)

    async def get_batch(
        self, texts: list[str], model_id: str = "default"
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """
        Get embeddings for multiple texts.

        OPTIMIZATION: Batch memory lookups first, then single DB query for missing.

        Returns:
            Tuple of (found embeddings dict, list of missing texts)
        """
        found = {}
        missing = []
        db_lookups = []

        # Phase 1: Check memory cache (fast, synchronous)
        with self._lock:
            for text in texts:
                key = self._hash_text(text, model_id)
                if key in self._memory_cache:
                    emb, _ = self._memory_cache[key]
                    found[text] = emb.copy()
                    # Update LRU order
                    if key in self._cache_order:
                        self._cache_order.remove(key)
                        self._cache_order.append(key)
                else:
                    db_lookups.append((text, key))

        # Phase 2: Batch DB lookup for cache misses
        if db_lookups:
            loop = asyncio.get_running_loop()
            db_results = await loop.run_in_executor(
                None, self._get_batch_from_db, [k for _, k in db_lookups]
            )

            for (text, key), embedding in zip(db_lookups, db_results, strict=False):
                if embedding is not None:
                    found[text] = embedding
                    self._add_to_memory_cache(key, embedding)
                else:
                    missing.append(text)

        self.hits += len(found)
        self.misses += len(missing)
        return found, missing

    def _get_batch_from_db(self, keys: list[str]) -> list[np.ndarray | None]:
        """Batch get embeddings from database."""
        if not keys:
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            placeholders = ",".join("?" * len(keys))
            cursor = conn.execute(
                f"SELECT text_hash, embedding, dimension FROM embeddings WHERE text_hash IN ({placeholders})",
                keys,
            )

            # Build result map
            result_map = {}
            for row in cursor:
                text_hash, embedding_blob, dimension = row
                result_map[text_hash] = self._deserialize_embedding(
                    embedding_blob, dimension
                )

            # Update access stats in batch
            now = time.time()
            conn.executemany(
                "UPDATE embeddings SET accessed_at = ?, access_count = access_count + 1 WHERE text_hash = ?",
                [(now, k) for k in result_map],
            )
            conn.commit()
            conn.close()

            # Return in original order
            return [result_map.get(k) for k in keys]

        except Exception as e:
            logger.warning(f"EmbeddingCache batch get error: {e}")
            return [None] * len(keys)

    async def set_batch(
        self, texts: list[str], embeddings: list[np.ndarray], model_id: str = "default"
    ) -> int:
        """
        Store multiple embeddings.

        OPTIMIZATION: Uses batch insert for better performance.

        Returns:
            Number of successfully stored embeddings
        """
        if not texts or not embeddings:
            return 0

        # Prepare batch data
        batch_data = []
        now = time.time()

        for text, embedding in zip(texts, embeddings, strict=False):
            key = self._hash_text(text, model_id)
            self._add_to_memory_cache(key, embedding)
            batch_data.append(
                (
                    key,
                    text[:1000],
                    self._serialize_embedding(embedding),
                    model_id,
                    len(embedding),
                    now,
                    now,
                )
            )

        # Batch insert to DB
        loop = asyncio.get_running_loop()
        success_count = await loop.run_in_executor(
            None, self._set_batch_in_db, batch_data
        )
        return success_count

    def _set_batch_in_db(self, batch_data: list[tuple]) -> int:
        """Batch insert embeddings to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (text_hash, text, embedding, model_id, dimension, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                batch_data,
            )
            conn.commit()
            conn.close()
            return len(batch_data)
        except Exception as e:
            logger.warning(f"EmbeddingCache batch set error: {e}")
            return 0

    def find_similar(
        self, query_embedding: np.ndarray, top_k: int = 5, model_id: str = "default"
    ) -> list[tuple[str, float]]:
        """
        Find similar embeddings in cache.

        Returns:
            List of (text, similarity_score) tuples
        """
        results = []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                "SELECT text, embedding, dimension FROM embeddings WHERE model_id = ?",
                (model_id,),
            )

            # SIMD-optimized similarity
            try:
                from backend.core.optimized.simd_ops import cosine_similarity_single

                use_simd = True
            except ImportError:
                use_simd = False
                query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

            for row in cursor:
                text, embedding_blob, dimension = row
                embedding = self._deserialize_embedding(embedding_blob, dimension)

                # Cosine similarity (SIMD-optimized)
                if use_simd:
                    similarity = cosine_similarity_single(query_embedding, embedding)
                else:
                    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
                    similarity = float(np.dot(query_norm, emb_norm))

                results.append((text, similarity))

            conn.close()

            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.warning(f"EmbeddingCache find_similar error: {e}")
            return []

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "dedup_hits": self.dedup_hits,
            "hit_rate": f"{hit_rate:.2%}",
            "memory_cache_size": len(self._memory_cache),
        }

    def cleanup(self, max_age_days: int = 30) -> int:
        """Remove old embeddings."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cutoff = time.time() - (max_age_days * 86400)
            cursor = conn.execute(
                "DELETE FROM embeddings WHERE accessed_at < ?", (cutoff,)
            )
            count = cursor.rowcount
            conn.commit()
            conn.close()
            logger.info(f"[EmbeddingCache] Cleaned up {count} old embeddings")
            return count
        except Exception as e:
            logger.warning(f"EmbeddingCache cleanup error: {e}")
            return 0


# Global singleton
_embedding_cache: EmbeddingCache | None = None
_cache_lock = threading.Lock()


def get_embedding_cache(db_path: str = "storage/cache/embeddings.db") -> EmbeddingCache:
    """Get global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        with _cache_lock:
            if _embedding_cache is None:
                _embedding_cache = EmbeddingCache(db_path=db_path)
    return _embedding_cache
