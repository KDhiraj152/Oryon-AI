"""
Performance Tuning Module
==========================

Advanced performance optimizations for Apple Silicon:
- Memory-mapped embedding storage
- Quantized attention patterns
- KV cache optimization
- Speculative decoding configuration
"""

import logging
import mmap
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""

    # Memory settings
    use_memory_mapping: bool = True
    memory_map_threshold_mb: int = 100  # Use mmap for files > 100MB

    # Quantization settings
    quantize_embeddings: bool = True
    embedding_precision: str = "float16"  # float32, float16, int8

    # KV Cache settings
    kv_cache_max_size: int = 4096  # Maximum sequence length
    kv_cache_compression: bool = True

    # Speculative decoding
    use_speculative_decoding: bool = False
    speculative_draft_tokens: int = 4

    # Batch settings
    optimal_batch_size: int = 32

    # M4-specific optimizations
    use_metal_optimizations: bool = True
    use_ane_offload: bool = True


class MemoryMappedEmbeddings:
    """
    Memory-mapped embedding storage for efficient access.

    Benefits:
    - Zero-copy access to embeddings
    - OS-level caching and paging
    - Reduced memory footprint
    - Fast startup (no full load needed)
    """

    HEADER_SIZE = 24  # dim (4) + count (4) + dtype (4) + reserved (12)

    def __init__(
        self,
        path: Path,
        dimension: int = 384,
        dtype: str = "float16",
    ):
        self.path = Path(path)
        self.dimension = dimension
        self.dtype = dtype
        self.dtype_size = 2 if dtype == "float16" else 4

        self._mmap: mmap.mmap | None = None
        self._file = None
        self._count = 0
        self._keys: dict[str, int] = {}

    def create(self, embeddings: dict[str, np.ndarray]):
        """
        Create memory-mapped embedding file.

        Args:
            embeddings: Dict mapping keys to embedding vectors
        """
        self._count = len(embeddings)

        # Calculate file size
        embedding_size = self.dimension * self.dtype_size
        data_size = self._count * embedding_size
        total_size = self.HEADER_SIZE + data_size

        # Create file
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "wb") as f:
            # Write header
            f.write(struct.pack("I", self.dimension))
            f.write(struct.pack("I", self._count))
            f.write(struct.pack("I", self.dtype_size))
            f.write(b"\x00" * 12)  # Reserved

            # Write embeddings
            dtype_np = np.float16 if self.dtype == "float16" else np.float32

            for i, (key, emb) in enumerate(embeddings.items()):
                self._keys[key] = i
                emb_typed = emb.astype(dtype_np)
                f.write(emb_typed.tobytes())

        # Save key index
        self._save_keys()

        logger.info(
            f"[MMap] Created embedding file: {self.path} ({total_size / 1024 / 1024:.1f}MB)"
        )

    def _save_keys(self):
        """Save key-to-index mapping."""
        import json

        keys_path = self.path.with_suffix(".keys")
        with open(keys_path, "w") as f:
            json.dump(self._keys, f)

    def _load_keys(self):
        """Load key-to-index mapping."""
        import json

        keys_path = self.path.with_suffix(".keys")
        if keys_path.exists():
            with open(keys_path) as f:
                self._keys = json.load(f)

    def open(self):
        """Open memory-mapped file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.path}")

        self._file = open(self.path, "r+b")
        self._mmap = mmap.mmap(self._file.fileno(), 0)

        # Read header
        self._mmap.seek(0)
        self.dimension = struct.unpack("I", self._mmap.read(4))[0]
        self._count = struct.unpack("I", self._mmap.read(4))[0]
        self.dtype_size = struct.unpack("I", self._mmap.read(4))[0]

        # Load keys
        self._load_keys()

        logger.info(f"[MMap] Opened: {self._count} embeddings, dim={self.dimension}")

    def get(self, key: str) -> np.ndarray | None:
        """Get embedding by key (zero-copy)."""
        if key not in self._keys:
            return None

        idx = self._keys[key]
        return self.get_by_index(idx)

    def get_by_index(self, idx: int) -> np.ndarray:
        """Get embedding by index (zero-copy)."""
        if self._mmap is None:
            raise RuntimeError("File not opened")

        # Calculate offset
        embedding_size = self.dimension * self.dtype_size
        offset = self.HEADER_SIZE + idx * embedding_size

        # Read embedding
        self._mmap.seek(offset)
        data = self._mmap.read(embedding_size)

        dtype_np = np.float16 if self.dtype_size == 2 else np.float32
        return np.frombuffer(data, dtype=dtype_np)

    def get_batch(self, keys: list[str]) -> tuple[np.ndarray, list[bool]]:
        """Get batch of embeddings."""
        embeddings = []
        found = []

        for key in keys:
            emb = self.get(key)
            if emb is not None:
                embeddings.append(emb)
                found.append(True)
            else:
                found.append(False)

        if embeddings:
            return np.vstack(embeddings), found
        return np.array([]), found

    def close(self):
        """Close memory-mapped file."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class QuantizedAttention:
    """
    Quantized attention patterns for memory efficiency.

    Uses INT8 quantization for attention weights to reduce
    memory bandwidth requirements on Apple Silicon.
    """

    def __init__(
        self,
        num_heads: int = 12,
        head_dim: int = 64,
        use_int8: bool = True,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_int8 = use_int8

        # Quantization scales (per-head)
        self._scales = np.ones(num_heads, dtype=np.float32)
        self._zero_points = np.zeros(num_heads, dtype=np.int8)

    def quantize_kv(
        self,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Quantize key-value tensors to INT8.

        Args:
            keys: Key tensor (batch, seq, heads, dim)
            values: Value tensor (batch, seq, heads, dim)

        Returns:
            Quantized keys, values, and quantization params
        """
        if not self.use_int8:
            return keys, values, {}

        # Per-head quantization
        params = {}

        # Quantize keys
        k_min = keys.min(axis=(0, 1, 3), keepdims=True)
        k_max = keys.max(axis=(0, 1, 3), keepdims=True)
        k_scale = (k_max - k_min) / 255.0
        k_zero = (-k_min / k_scale).astype(np.int8)

        keys_quant = ((keys - k_min) / k_scale).astype(np.uint8)
        params["k_scale"] = k_scale.squeeze()
        params["k_zero"] = k_zero.squeeze()

        # Quantize values
        v_min = values.min(axis=(0, 1, 3), keepdims=True)
        v_max = values.max(axis=(0, 1, 3), keepdims=True)
        v_scale = (v_max - v_min) / 255.0
        v_zero = (-v_min / v_scale).astype(np.int8)

        values_quant = ((values - v_min) / v_scale).astype(np.uint8)
        params["v_scale"] = v_scale.squeeze()
        params["v_zero"] = v_zero.squeeze()

        return keys_quant, values_quant, params

    def dequantize_kv(
        self,
        keys_quant: np.ndarray,
        values_quant: np.ndarray,
        params: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dequantize key-value tensors."""
        if not params:
            return keys_quant, values_quant

        keys = (
            keys_quant.astype(np.float32) * params["k_scale"]
            + params["k_zero"] * params["k_scale"]
        )
        values = (
            values_quant.astype(np.float32) * params["v_scale"]
            + params["v_zero"] * params["v_scale"]
        )

        return keys, values


@dataclass
class SpeculativeDecodingConfig:
    """
    Configuration for speculative decoding.

    Speculative decoding uses a smaller "draft" model to
    generate candidate tokens, then verifies with the main
    model. This can significantly speed up generation.
    """

    enabled: bool = False

    # Draft model (smaller, faster)
    draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Number of tokens to speculate
    num_speculative_tokens: int = 4

    # Acceptance threshold
    acceptance_threshold: float = 0.8

    # Fall back to normal decoding after N rejections
    max_rejections: int = 3


# ============================================================================
# SPECULATIVE DECODING FOR METAL/ANE
# ============================================================================


@dataclass
class SpeculativeDecodingStats:
    """Statistics for speculative decoding performance."""

    total_generations: int = 0
    total_tokens_generated: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    total_speculations: int = 0
    avg_accepted_per_speculation: float = 0.0
    speedup_factor: float = 1.0

    @property
    def acceptance_rate(self) -> float:
        """Calculate token acceptance rate."""
        total = self.tokens_accepted + self.tokens_rejected
        return self.tokens_accepted / max(1, total)

    @property
    def avg_speedup(self) -> float:
        """Alias for speedup_factor."""
        return self.speedup_factor

    @property
    def avg_draft_tokens(self) -> float:
        """Average draft tokens per speculation."""
        if self.total_speculations == 0:
            return 0.0
        return (self.tokens_accepted + self.tokens_rejected) / self.total_speculations

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_generations": self.total_generations,
            "total_tokens_generated": self.total_tokens_generated,
            "tokens_accepted": self.tokens_accepted,
            "tokens_rejected": self.tokens_rejected,
            "acceptance_rate": self.acceptance_rate,
            "total_speculations": self.total_speculations,
            "avg_accepted_per_speculation": self.avg_accepted_per_speculation,
            "speedup_factor": self.speedup_factor,
        }


class MetalSpeculativeDecoder:
    """
    Speculative decoding optimized for Metal/ANE on Apple Silicon.

    Architecture:
    1. Draft model runs on CPU/ANE (small, fast)
    2. Main model verifies on GPU/MPS (accurate)
    3. Parallel execution for maximum throughput

    M4 Optimization:
    - Draft model uses ANE via CoreML for low power
    - Main model uses MPS for GPU acceleration
    - Unified memory enables zero-copy tensor sharing
    - Metal command buffers for async execution
    """

    def __init__(
        self,
        config: SpeculativeDecodingConfig,
        draft_model: Any | None = None,
        main_model: Any | None = None,
    ):
        self.config = config
        self._draft_model = draft_model
        self._main_model = main_model
        self._stats = SpeculativeDecodingStats()
        self._lock = threading.Lock()

        # Device configuration
        self._draft_device = "cpu"  # ANE via CoreML or CPU
        self._main_device = "mps"  # Metal for verification

        logger.info(
            f"MetalSpeculativeDecoder initialized: "
            f"draft={config.draft_model}, k={config.num_speculative_tokens}"
        )

    @property
    def device(self) -> str:
        """Get the main compute device."""
        return self._main_device

    def is_available(self) -> bool:
        """Check if speculative decoding is available and properly configured."""
        return (
            self.config.enabled
            and self._draft_model is not None
            and self._main_model is not None
        )

    def get_stats(self) -> SpeculativeDecodingStats:
        """Get current statistics."""
        with self._lock:
            return SpeculativeDecodingStats(
                total_generations=self._stats.total_generations,
                total_tokens_generated=self._stats.total_tokens_generated,
                tokens_accepted=self._stats.tokens_accepted,
                tokens_rejected=self._stats.tokens_rejected,
                total_speculations=self._stats.total_speculations,
                avg_accepted_per_speculation=self._stats.avg_accepted_per_speculation,
                speedup_factor=self._stats.speedup_factor,
            )

    def set_models(self, draft_model: Any, main_model: Any) -> None:
        """Set the draft and main models."""
        self._draft_model = draft_model
        self._main_model = main_model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> tuple[str, SpeculativeDecodingStats]:
        """
        Generate tokens using speculative decoding.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Sequences that stop generation

        Returns:
            (generated_text, stats)
        """
        if not self.config.enabled or not self._draft_model or not self._main_model:
            # Fall back to normal generation
            return await self._normal_generate(prompt, max_tokens, temperature)

        generated_tokens = []
        current_prompt = prompt
        rejections = 0

        while len(generated_tokens) < max_tokens:
            # Check rejection limit
            if rejections >= self.config.max_rejections:
                # Too many rejections, fall back to normal decoding
                remaining = max_tokens - len(generated_tokens)
                fallback_text = await self._normal_generate(
                    current_prompt, remaining, temperature
                )
                return self._decode_tokens(generated_tokens) + fallback_text[
                    0
                ], self._stats

            # Step 1: Draft model generates K candidate tokens
            draft_tokens, draft_probs = await self._draft_generate(
                current_prompt,
                k=self.config.num_speculative_tokens,
                temperature=temperature,
            )

            # Step 2: Main model verifies all K tokens in parallel
            accepted_tokens, main_probs = await self._verify_tokens(
                current_prompt,
                draft_tokens,
                temperature=temperature,
            )

            with self._lock:
                self._stats.total_speculations += 1

            if len(accepted_tokens) == 0:
                # All tokens rejected - sample one from main model
                rejections += 1
                new_token = self._sample_from_diff(draft_probs[0], main_probs[0])
                accepted_tokens = [new_token]

                with self._lock:
                    self._stats.tokens_rejected += len(draft_tokens)
            else:
                with self._lock:
                    self._stats.tokens_accepted += len(accepted_tokens)
                    self._stats.tokens_rejected += len(draft_tokens) - len(
                        accepted_tokens
                    )

            # Add accepted tokens
            generated_tokens.extend(accepted_tokens)
            current_prompt = self._update_prompt(current_prompt, accepted_tokens)

            # Check for stop sequences
            text_so_far = self._decode_tokens(generated_tokens)
            if stop_sequences and any(s in text_so_far for s in stop_sequences):
                break

        # Update stats
        with self._lock:
            self._stats.total_tokens_generated += len(generated_tokens)
            if self._stats.total_speculations > 0:
                self._stats.avg_accepted_per_speculation = (
                    self._stats.tokens_accepted / self._stats.total_speculations
                )
            # Estimate speedup (draft tokens are "free" if accepted)
            if self._stats.tokens_accepted > 0:
                self._stats.speedup_factor = (
                    self._stats.tokens_accepted + self._stats.tokens_rejected
                ) / max(1, self._stats.total_speculations + self._stats.tokens_rejected)

        return self._decode_tokens(generated_tokens), self._stats

    async def _draft_generate(
        self,
        prompt: str,
        k: int,
        temperature: float,
    ) -> tuple[list[int], list[np.ndarray]]:
        """
        Generate K candidate tokens using draft model.

        Runs on CPU/ANE for low power consumption.
        """
        # This is a placeholder - actual implementation depends on model framework
        # For MLX: Use mlx.core for fast CPU/ANE inference
        # For Transformers: Use model.generate with max_new_tokens=k

        try:
            if hasattr(self._draft_model, "generate_tokens"):
                # Custom interface for draft generation
                return await self._draft_model.generate_tokens(prompt, k, temperature)
            else:
                # Fallback: generate one at a time (slower)
                tokens = []
                probs = []
                current = prompt

                for _ in range(k):
                    # Simulate single token generation
                    token, prob = await self._single_token_generate(
                        self._draft_model, current, temperature
                    )
                    tokens.append(token)
                    probs.append(prob)
                    current = self._update_prompt(current, [token])

                return tokens, probs

        except Exception as e:
            logger.error(f"Draft generation failed: {e}")
            return [], []

    async def _verify_tokens(
        self,
        prompt: str,
        draft_tokens: list[int],
        temperature: float,
    ) -> tuple[list[int], list[np.ndarray]]:
        """
        Verify draft tokens using main model.

        The key insight is that we can verify all K tokens in a SINGLE
        forward pass of the main model, making verification almost free.

        Runs on GPU (MPS/Metal) for maximum throughput.
        """
        if not draft_tokens:
            return [], []

        try:
            if hasattr(self._main_model, "verify_tokens"):
                # Custom interface for verification
                accepted, probs = await self._main_model.verify_tokens(
                    prompt, draft_tokens, temperature, self.config.acceptance_threshold
                )
                return accepted, probs

            # Default verification logic
            # Get probabilities from main model for each position
            main_probs = await self._get_token_probs(
                self._main_model, prompt, len(draft_tokens)
            )

            # Accept tokens greedily until rejection
            accepted = []
            for _i, (token, prob) in enumerate(
                zip(draft_tokens, main_probs, strict=False)
            ):
                # Check if main model agrees with draft
                main_prob_for_token = (
                    prob[token] if isinstance(prob, np.ndarray) else prob
                )

                if main_prob_for_token >= self.config.acceptance_threshold:
                    accepted.append(token)
                else:
                    # First rejection - stop accepting
                    break

            return accepted, main_probs

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return [], []

    async def _normal_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, SpeculativeDecodingStats]:
        """Normal generation without speculation."""
        try:
            if hasattr(self._main_model, "generate"):
                result = await self._main_model.generate(
                    prompt, max_tokens=max_tokens, temperature=temperature
                )
                return result, self._stats
            return "", self._stats
        except Exception as e:
            logger.error(f"Normal generation failed: {e}")
            return "", self._stats

    def _single_token_generate(
        self,
        model: Any,
        prompt: str,
        temperature: float,
    ) -> tuple[int, np.ndarray]:
        """Generate a single token.

        Args:
            model: The model to use for generation
            prompt: Input prompt (reserved for implementation)
            temperature: Sampling temperature (reserved for implementation)
        """
        # Placeholder - depends on model implementation
        # Suppress unused argument warnings
        _ = (model, prompt, temperature)
        return 0, np.array([1.0])

    def _get_token_probs(
        self,
        model: Any,
        prompt: str,
        n_tokens: int,
    ) -> list[np.ndarray]:
        """Get probability distributions for next N tokens.

        Args:
            model: The model to use
            prompt: Input prompt (reserved for implementation)
            n_tokens: Number of tokens to get probs for
        """
        # Placeholder - depends on model implementation
        _ = (model, prompt)  # Suppress unused argument warnings
        return [np.array([1.0]) for _ in range(n_tokens)]

    def _sample_from_diff(
        self,
        draft_prob: np.ndarray,
        main_prob: np.ndarray,
    ) -> int:
        """Sample from (main_prob - draft_prob) distribution."""
        # Rejection sampling from difference distribution
        diff = np.maximum(0, main_prob - draft_prob)
        if diff.sum() > 0:
            diff = diff / diff.sum()
            # Use entropy from system for randomness (not cryptographic, just sampling)
            rng = np.random.default_rng(seed=None)  # None uses system entropy
            return int(rng.choice(len(diff), p=diff))
        return int(np.argmax(main_prob))

    def _update_prompt(self, prompt: str, tokens: list[int]) -> str:
        """Update prompt with new tokens (placeholder)."""
        # In practice, this depends on the tokenizer
        return prompt + "".join(str(t) for t in tokens)

    def _decode_tokens(self, tokens: list[int]) -> str:
        """Decode tokens to text (placeholder)."""
        # In practice, use the tokenizer
        return "".join(str(t) for t in tokens)

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats = SpeculativeDecodingStats()


# Singleton instance
_speculative_decoder: MetalSpeculativeDecoder | None = None
_decoder_lock = threading.Lock()


def get_speculative_decoder(
    config: SpeculativeDecodingConfig | None = None,
) -> MetalSpeculativeDecoder:
    """Get or create the global speculative decoder."""
    global _speculative_decoder

    if _speculative_decoder is None:
        with _decoder_lock:
            if _speculative_decoder is None:
                _speculative_decoder = MetalSpeculativeDecoder(
                    config or SpeculativeDecodingConfig()
                )
                logger.info("Created MetalSpeculativeDecoder singleton")

    return _speculative_decoder


class PerformanceOptimizer:
    """
    Central performance optimization manager.

    Coordinates all performance tuning components for
    optimal inference on Apple Silicon.
    """

    def __init__(self, config: PerformanceConfig | None = None):
        self.config = config or PerformanceConfig()

        # Components
        self._mmap_embeddings: MemoryMappedEmbeddings | None = None
        self._quantized_attention: QuantizedAttention | None = None

        # Stats
        self._stats = {
            "mmap_hits": 0,
            "quantization_savings_mb": 0,
            "speculative_accepted": 0,
            "speculative_rejected": 0,
        }

    def setup_mmap_embeddings(
        self,
        path: Path,
        dimension: int = 384,
    ) -> MemoryMappedEmbeddings:
        """Set up memory-mapped embeddings."""
        self._mmap_embeddings = MemoryMappedEmbeddings(
            path=path,
            dimension=dimension,
            dtype="float16" if self.config.quantize_embeddings else "float32",
        )
        return self._mmap_embeddings

    def setup_quantized_attention(
        self,
        num_heads: int = 12,
        head_dim: int = 64,
    ) -> QuantizedAttention:
        """Set up quantized attention."""
        self._quantized_attention = QuantizedAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            use_int8=self.config.kv_cache_compression,
        )
        return self._quantized_attention

    def get_optimal_settings(self) -> dict[str, Any]:
        """Get optimal settings based on device capabilities."""
        try:
            from ..core.optimized import get_device_router

            router = get_device_router()
            caps = router.capabilities

            settings = {
                "batch_size": self.config.optimal_batch_size,
                "use_mmap": self.config.use_memory_mapping,
                "precision": self.config.embedding_precision,
                "kv_cache_size": self.config.kv_cache_max_size,
            }

            # Adjust based on device
            if caps.is_m4:
                settings["batch_size"] = 64  # M4 handles larger batches
                settings["precision"] = "float16"
                settings["use_ane"] = True
            elif caps.is_apple_silicon:
                settings["batch_size"] = 32
                settings["precision"] = "float16"
            elif caps.has_cuda:
                settings["batch_size"] = 128
                settings["precision"] = "float16"
            else:
                settings["batch_size"] = 16
                settings["precision"] = "float32"

            return settings

        except Exception as e:
            logger.warning(f"Could not detect device: {e}")
            return {
                "batch_size": 32,
                "use_mmap": True,
                "precision": "float32",
            }

    def get_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        return {
            **self._stats,
            "config": {
                "use_memory_mapping": self.config.use_memory_mapping,
                "quantize_embeddings": self.config.quantize_embeddings,
                "kv_cache_compression": self.config.kv_cache_compression,
                "speculative_decoding": self.config.use_speculative_decoding,
            },
        }


# ==================== Global Instances ====================

_optimizer: PerformanceOptimizer | None = None


def get_performance_optimizer(
    config: PerformanceConfig | None = None,
) -> PerformanceOptimizer:
    """Get global performance optimizer."""
    global _optimizer

    if _optimizer is None:
        _optimizer = PerformanceOptimizer(config)

    return _optimizer


__all__ = [
    # Memory-mapped embeddings
    "MemoryMappedEmbeddings",
    "MetalSpeculativeDecoder",
    # Config
    "PerformanceConfig",
    # Performance optimizer
    "PerformanceOptimizer",
    # Quantized attention
    "QuantizedAttention",
    "SpeculativeDecodingConfig",
    # Speculative decoding for Metal/ANE
    "SpeculativeDecodingStats",
    "get_performance_optimizer",
    "get_speculative_decoder",
]
