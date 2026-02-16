"""
Pre-tokenization Worker (Principle G)
=====================================
Pre-tokenize text before sending to LLM to save 300-600ms per call.

Strategy:
- Tokenize input text in a separate worker before LLM inference
- Cache tokenized representations with TTL
- Batch tokenization for efficiency
- Support multiple tokenizer types (LLM, embedding, translation)

Reference: "Pre-tokenize text before sending to LLM"
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from backend.utils.hashing import fast_hash

logger = logging.getLogger(__name__)

class TokenizerType(Enum):
    """Types of tokenizers supported."""

    LLM = "llm"  # Qwen3-8B tokenizer
    EMBEDDING = "embedding"  # BGE-M3 tokenizer
    TRANSLATION = "translation"  # IndicTrans2 tokenizer

@dataclass
class TokenizedInput:
    """Tokenized input ready for model inference."""

    input_ids: list[int]
    attention_mask: list[int]
    token_count: int
    text_hash: str
    tokenizer_type: TokenizerType
    created_at: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: float = 300) -> bool:
        """Check if tokenized input has expired."""
        return (time.time() - self.created_at) > ttl_seconds

@dataclass
class TokenizationConfig:
    """Configuration for pre-tokenization."""

    # Max sequence lengths per model type
    max_length_llm: int = 4096  # Qwen3-8B context
    max_length_embedding: int = 8192  # BGE-M3 context
    max_length_translation: int = 512  # IndicTrans2 optimal

    # Batching
    batch_size: int = 16

    # Cache settings
    cache_ttl_seconds: float = 300  # 5 minutes
    max_cache_entries: int = 10000

    # Padding strategy
    pad_to_multiple_of: int = 8  # Tensor core alignment

class TokenizerPool:
    """
    Pool of tokenizers for different model types.
    Loads tokenizers lazily and reuses them.
    """

    def __init__(self, config: TokenizationConfig):
        self.config = config
        self._tokenizers: dict[TokenizerType, Any] = {}
        self._lock = asyncio.Lock()

    async def get_tokenizer(self, tokenizer_type: TokenizerType) -> Any:
        """Get or load tokenizer for given type."""
        if tokenizer_type not in self._tokenizers:
            async with self._lock:
                # Double-check after acquiring lock
                if tokenizer_type not in self._tokenizers:
                    self._tokenizers[tokenizer_type] = await self._load_tokenizer(
                        tokenizer_type
                    )
        return self._tokenizers[tokenizer_type]

    async def _load_tokenizer(self, tokenizer_type: TokenizerType) -> Any:
        """Load tokenizer for given type."""
        try:
            from transformers import AutoTokenizer

            # Import settings
            from backend.core.config import get_settings

            settings = get_settings()

            model_id = {
                TokenizerType.LLM: settings.SIMPLIFICATION_MODEL_ID,
                TokenizerType.EMBEDDING: settings.EMBEDDING_MODEL_ID,
                TokenizerType.TRANSLATION: settings.TRANSLATION_MODEL_ID,
            }.get(tokenizer_type)

            if not model_id:
                raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

            logger.info("Loading tokenizer for %s: %s", tokenizer_type.value, model_id)

            # Load tokenizer with cache_dir for local loading
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True,  # Use fast tokenizer when available
                cache_dir=str(settings.MODEL_CACHE_DIR),
            )

            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info("Loaded tokenizer for %s", tokenizer_type.value)
            return tokenizer

        except (RuntimeError, ValueError) as e:
            logger.error("Failed to load tokenizer for %s: %s", tokenizer_type.value, e)
            raise

class PreTokenizationWorker:
    """
    Worker that pre-tokenizes text before LLM inference.

    Benefits:
    - Saves 300-600ms per LLM call by pre-computing tokens
    - Enables batch tokenization for efficiency
    - Caches tokenized inputs to avoid re-computation
    - Validates sequence lengths before inference
    """

    def __init__(self, config: TokenizationConfig | None = None):
        self.config = config or TokenizationConfig()
        self.tokenizer_pool = TokenizerPool(self.config)
        self._cache: dict[str, TokenizedInput] = {}
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_processed": 0,
            "batch_count": 0,
            "avg_tokenization_time_ms": 0.0,
        }
        self._total_time_ms = 0.0

    def _compute_hash(self, text: str, tokenizer_type: TokenizerType) -> str:
        """Compute cache key hash for text. Uses fast xxhash if available."""
        content = f"{tokenizer_type.value}:{text}"
        return fast_hash(content, length=16)

    async def tokenize(
        self, text: str, tokenizer_type: TokenizerType, use_cache: bool = True
    ) -> TokenizedInput:
        """
        Tokenize text for given model type.

        Args:
            text: Input text to tokenize
            tokenizer_type: Type of tokenizer to use
            use_cache: Whether to use/update cache

        Returns:
            TokenizedInput ready for model inference
        """
        text_hash = self._compute_hash(text, tokenizer_type)

        # Check cache
        if use_cache and text_hash in self._cache:
            cached = self._cache[text_hash]
            if not cached.is_expired(self.config.cache_ttl_seconds):
                self._stats["cache_hits"] += 1
                return cached
            else:
                del self._cache[text_hash]

        self._stats["cache_misses"] += 1

        # Get tokenizer and tokenize
        start_time = time.time()

        tokenizer = await self.tokenizer_pool.get_tokenizer(tokenizer_type)
        max_length = self._get_max_length(tokenizer_type)

        # Tokenize
        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length" if self.config.pad_to_multiple_of else False,
            return_tensors=None,  # Return lists
        )

        elapsed_ms = (time.time() - start_time) * 1000
        self._update_stats(elapsed_ms, len(encoding["input_ids"]))

        result = TokenizedInput(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_count=sum(encoding["attention_mask"]),  # Actual token count
            text_hash=text_hash,
            tokenizer_type=tokenizer_type,
        )

        # Update cache
        if use_cache:
            self._cache_put(text_hash, result)

        return result

    async def tokenize_batch(
        self, texts: list[str], tokenizer_type: TokenizerType, use_cache: bool = True
    ) -> list[TokenizedInput]:
        """
        Batch tokenize multiple texts.

        More efficient than individual tokenization due to batched processing.

        Args:
            texts: List of texts to tokenize
            tokenizer_type: Type of tokenizer to use
            use_cache: Whether to use/update cache

        Returns:
            List of TokenizedInput objects
        """
        results: list[TokenizedInput] = []
        to_tokenize: list[tuple[int, str]] = []  # (index, text)

        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = self._compute_hash(text, tokenizer_type)

            if use_cache and text_hash in self._cache:
                cached = self._cache[text_hash]
                if not cached.is_expired(self.config.cache_ttl_seconds):
                    self._stats["cache_hits"] += 1
                    results.append((i, cached))
                    continue
                else:
                    del self._cache[text_hash]

            self._stats["cache_misses"] += 1
            to_tokenize.append((i, text))

        # Batch tokenize uncached texts
        if to_tokenize:
            start_time = time.time()

            tokenizer = await self.tokenizer_pool.get_tokenizer(tokenizer_type)
            max_length = self._get_max_length(tokenizer_type)

            # Extract texts for batch tokenization
            batch_texts = [text for _, text in to_tokenize]

            # Batch tokenize
            encodings = tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                pad_to_multiple_of=self.config.pad_to_multiple_of,
                return_tensors=None,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            self._stats["batch_count"] += 1

            # Create TokenizedInput for each
            for j, (idx, text) in enumerate(to_tokenize):
                text_hash = self._compute_hash(text, tokenizer_type)

                input_ids = encodings["input_ids"][j]
                attention_mask = encodings["attention_mask"][j]

                self._update_stats(elapsed_ms / len(batch_texts), sum(attention_mask))

                result = TokenizedInput(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_count=sum(attention_mask),
                    text_hash=text_hash,
                    tokenizer_type=tokenizer_type,
                )

                if use_cache:
                    self._cache_put(text_hash, result)

                results.append((idx, result))

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    async def validate_length(
        self, text: str, tokenizer_type: TokenizerType
    ) -> tuple[bool, int, int]:
        """
        Validate that text fits within model's context window.

        Returns:
            Tuple of (is_valid, token_count, max_length)
        """
        tokenized = await self.tokenize(text, tokenizer_type, use_cache=True)
        max_length = self._get_max_length(tokenizer_type)

        return (tokenized.token_count <= max_length, tokenized.token_count, max_length)

    async def chunk_for_model(
        self, text: str, tokenizer_type: TokenizerType, overlap_tokens: int = 50
    ) -> list[str]:
        """
        Split text into chunks that fit within model's context window.

        Uses token-aware chunking to ensure no chunk exceeds max length.

        Args:
            text: Text to chunk
            tokenizer_type: Target model tokenizer
            overlap_tokens: Number of overlapping tokens between chunks

        Returns:
            List of text chunks
        """
        tokenizer = await self.tokenizer_pool.get_tokenizer(tokenizer_type)
        max_length = self._get_max_length(tokenizer_type)

        # Account for special tokens and leave room for generation
        effective_max = max_length - 100  # Reserve for special tokens

        # Tokenize full text
        full_encoding = tokenizer.encode(text, add_special_tokens=False)

        if len(full_encoding) <= effective_max:
            return [text]

        # Split into chunks with overlap
        chunks = []
        start = 0

        while start < len(full_encoding):
            end = min(start + effective_max, len(full_encoding))
            chunk_ids = full_encoding[start:end]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Move start with overlap
            start = end - overlap_tokens if end < len(full_encoding) else end

        logger.info("Split text into %s chunks for %s", len(chunks), tokenizer_type.value)
        return chunks

    def _get_max_length(self, tokenizer_type: TokenizerType) -> int:
        """Get max sequence length for tokenizer type."""
        return {
            TokenizerType.LLM: self.config.max_length_llm,
            TokenizerType.EMBEDDING: self.config.max_length_embedding,
            TokenizerType.TRANSLATION: self.config.max_length_translation,
        }[tokenizer_type]

    def _cache_put(self, key: str, value: TokenizedInput) -> None:
        """Add item to cache, evicting oldest if necessary."""
        # Simple size-based eviction
        if len(self._cache) >= self.config.max_cache_entries:
            # Remove oldest entries (first 10%)
            to_remove = list(self._cache.keys())[: self.config.max_cache_entries // 10]
            for k in to_remove:
                del self._cache[k]

        self._cache[key] = value

    def _update_stats(self, elapsed_ms: float, token_count: int) -> None:
        """Update performance statistics."""
        self._stats["total_tokens_processed"] += token_count

        # Running average of tokenization time
        total_ops = self._stats["cache_misses"]
        self._total_time_ms += elapsed_ms
        self._stats["avg_tokenization_time_ms"] = self._total_time_ms / max(
            total_ops, 1
        )

    def get_stats(self) -> dict[str, Any]:
        """Get tokenization statistics."""
        cache_total = self._stats["cache_hits"] + self._stats["cache_misses"]
        return {
            **self._stats,
            "cache_hit_rate": self._stats["cache_hits"] / max(cache_total, 1),
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Clear tokenization cache."""
        self._cache.clear()
        logger.info("Cleared tokenization cache")

# Global worker instance
_worker: PreTokenizationWorker | None = None

def get_tokenization_worker() -> PreTokenizationWorker:
    """Get or create global tokenization worker."""
    global _worker
    if _worker is None:
        _worker = PreTokenizationWorker()
    return _worker

# Convenience functions
async def pre_tokenize(text: str, model_type: str = "llm") -> TokenizedInput:
    """
    Pre-tokenize text for given model type.

    Args:
        text: Input text
        model_type: One of 'llm', 'embedding', 'translation'

    Returns:
        TokenizedInput ready for inference
    """
    worker = get_tokenization_worker()
    tokenizer_type = TokenizerType(model_type)
    return await worker.tokenize(text, tokenizer_type)

async def pre_tokenize_batch(
    texts: list[str], model_type: str = "llm"
) -> list[TokenizedInput]:
    """
    Batch pre-tokenize texts for given model type.

    Args:
        texts: List of input texts
        model_type: One of 'llm', 'embedding', 'translation'

    Returns:
        List of TokenizedInput objects
    """
    worker = get_tokenization_worker()
    tokenizer_type = TokenizerType(model_type)
    return await worker.tokenize_batch(texts, tokenizer_type)

async def chunk_text(
    text: str, model_type: str = "llm", overlap: int = 50
) -> list[str]:
    """
    Chunk text to fit within model's context window.

    Args:
        text: Input text
        model_type: Target model type
        overlap: Token overlap between chunks

    Returns:
        List of text chunks
    """
    worker = get_tokenization_worker()
    tokenizer_type = TokenizerType(model_type)
    return await worker.chunk_for_model(text, tokenizer_type, overlap)
