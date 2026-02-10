"""
Unified Inference Engine - Intelligent Backend Selection
==========================================================

Automatically selects the best inference backend based on:
- Task type (LLM, embedding, etc.)
- Available hardware (M4, CUDA, CPU)
- Model requirements
- Memory constraints

This is the main entry point for all inference operations.
"""

import asyncio
import logging
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ...cache import (
    get_embedding_cache,
    get_response_cache,
    get_unified_cache,
)
from ...core.optimized.core_affinity import (
    get_affinity_manager,
)
from ...core.optimized.device_router import (
    ComputeBackend,
    TaskType,
    get_device_router,
)
from ...core.optimized.gpu_pipeline import (
    get_gpu_scheduler,
)
from ...core.optimized.memory_pool import (
    get_memory_pool,
)
from ...core.optimized.quantization import QuantizationStrategy

logger = logging.getLogger(__name__)


class InferenceMode(str, Enum):
    """Inference modes."""

    QUALITY = "quality"  # Prioritize accuracy
    BALANCED = "balanced"  # Balance speed and quality
    FAST = "fast"  # Prioritize speed


@dataclass
class GenerationConfig:
    """Unified generation configuration."""

    max_tokens: int = 512  # Reasonable default for Q&A (most answers <300 tokens)
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: list[str] | None = None
    stream: bool = False
    mode: InferenceMode = InferenceMode.BALANCED

    # System prompt
    system_prompt: str | None = None

    # Caching
    use_cache: bool = True
    cache_ttl: int = 300

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


class UnifiedInferenceEngine:
    """
    Unified inference engine with automatic backend selection.

    Provides a single interface for:
    - LLM text generation (MLX, MPS, CUDA, ONNX)
    - Embedding generation (CoreML, MPS, CPU)
    - Streaming responses
    - Automatic caching
    """

    DEFAULT_SYSTEM_PROMPT = """You are ShikshaSetu, an educational AI assistant for Indian students.

IMPORTANT RULES:
1. Only state facts you are certain about
2. If unsure, say "I'm not certain about this"
3. Never invent dates, names, statistics, or facts
4. Base answers on provided context when available
5. Explain concepts clearly for the student's level
6. Use examples relevant to Indian education

For math: Use LaTeX ($inline$ or $$block$$)
For code: Use proper code blocks with language tags"""

    def __init__(
        self,
        llm_model: str = "qwen2.5-3b",
        embedding_model: str = "all-MiniLM-L6-v2",
        auto_load: bool = False,
    ):
        """
        Initialize unified inference engine.

        Args:
            llm_model: Model for text generation
            embedding_model: Model for embeddings
            auto_load: Whether to load models immediately
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        # Get device router
        self.device_router = get_device_router()

        # Phase 3: GPU Pipeline Scheduler
        self._gpu_scheduler = get_gpu_scheduler()

        # Phase 4: Core Affinity Manager
        self._affinity_manager = get_affinity_manager()

        # Phase 5: Memory Pool
        self._memory_pool = get_memory_pool()

        # Backends (lazy loaded)
        self._mlx_engine: Any = None
        self._mps_engine: Any = None
        self._coreml_embeddings: Any = None
        self._hf_embeddings: Any = None

        # Lock for thread safety
        self._lock = threading.Lock()

        # Caches
        self._response_cache = get_response_cache()
        self._embedding_cache = get_embedding_cache()
        self._unified_cache = get_unified_cache()

        # Statistics
        self._stats = {
            "llm_requests": 0,
            "embedding_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }

        if auto_load:
            self._ensure_llm_loaded()
            self._ensure_embedding_loaded()

        logger.info(
            f"[UnifiedEngine] Initialized with {self.device_router.capabilities.chip_name}"
        )
        logger.info(
            "[UnifiedEngine] M4 Optimizations: GPU Pipeline, Core Affinity, Memory Pool"
        )

    def _ensure_llm_loaded(self) -> None:
        """Ensure LLM backend is loaded."""
        with self._lock:
            routing = self.device_router.route(TaskType.LLM_INFERENCE)

            if routing.backend == ComputeBackend.MLX:
                if self._mlx_engine is None:
                    from .mlx_backend import MLXInferenceEngine

                    self._mlx_engine = MLXInferenceEngine(model_id=self.llm_model)
                    self._mlx_engine.load()

            elif routing.backend in (ComputeBackend.MPS, ComputeBackend.CUDA):
                if self._mps_engine is None:
                    self._mps_engine = self._create_pytorch_engine()

            else:
                # CPU fallback
                if self._mps_engine is None:
                    self._mps_engine = self._create_pytorch_engine(device="cpu")

    def _ensure_embedding_loaded(self) -> None:
        """Ensure embedding backend is loaded."""
        with self._lock:
            routing = self.device_router.route(TaskType.EMBEDDING)

            if routing.backend == ComputeBackend.COREML:
                if self._coreml_embeddings is None:
                    from .coreml_backend import CoreMLEmbeddingEngine

                    self._coreml_embeddings = CoreMLEmbeddingEngine(
                        model_id=self.embedding_model
                    )
                    self._coreml_embeddings.load()

            else:
                # HuggingFace/PyTorch fallback
                if self._hf_embeddings is None:
                    self._hf_embeddings = self._create_hf_embeddings()

    def _create_pytorch_engine(self, device: str | None = None):
        """Create PyTorch-based LLM engine."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ...core.config import settings

        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # Get quantization config
        quant_config = QuantizationStrategy.get_optimal_config(
            model_size_b=3.0,  # Assume 3B model
            device_router=self.device_router,
            accuracy_priority=True,
        )

        # Load model from local cache
        model_id = settings.SIMPLIFICATION_MODEL_ID
        cache_dir = (
            str(settings.MODEL_CACHE_DIR) if settings.MODEL_CACHE_DIR.exists() else None
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

        model_kwargs = quant_config.to_transformers_kwargs()
        model_kwargs["device_map"] = device
        model_kwargs["cache_dir"] = cache_dir

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        return {"model": model, "tokenizer": tokenizer, "device": device}

    def _create_hf_embeddings(self):
        """Create HuggingFace embedding model."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        from ...core.config import settings

        model_id = f"sentence-transformers/{self.embedding_model}"
        cache_dir = (
            str(settings.MODEL_CACHE_DIR) if settings.MODEL_CACHE_DIR.exists() else None
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)

        # Move to best device
        if torch.backends.mps.is_available():
            model = model.to("mps")
        elif torch.cuda.is_available():
            model = model.to("cuda")

        return {"model": model, "tokenizer": tokenizer}

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:
        """
        Generate text from prompt.

        OPTIMIZATION: Uses response cache for deterministic requests (temp=0).

        Args:
            prompt: User prompt
            config: Generation configuration

        Returns:
            Generated text
        """
        config = config or GenerationConfig()
        self._stats["llm_requests"] += 1

        # OPTIMIZATION: Check cache first for low-temperature requests (near-deterministic)
        if config.use_cache and config.temperature <= 0.3:
            cached = self._response_cache.get(
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            if cached:
                self._stats["cache_hits"] += 1
                return cached

        # Ensure model is loaded
        self._ensure_llm_loaded()

        start = time.perf_counter()

        # Route to appropriate backend
        routing = self.device_router.route(TaskType.LLM_INFERENCE)

        try:
            if routing.backend == ComputeBackend.MLX and self._mlx_engine:
                from .mlx_backend import MLXGenerationConfig

                mlx_config = MLXGenerationConfig(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                )

                response: str = await self._mlx_engine.generate(
                    prompt,
                    config=mlx_config,
                    system_prompt=config.system_prompt or self.DEFAULT_SYSTEM_PROMPT,
                )

            elif self._mps_engine:
                response = await self._generate_pytorch(prompt, config)

            else:
                raise RuntimeError("No LLM backend available")

            elapsed = time.perf_counter() - start
            self._stats["total_time"] += elapsed

            # Cache response
            if config.use_cache and config.temperature == 0:
                self._response_cache.set(
                    prompt,
                    response,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

            return response

        except Exception as e:
            logger.error(f"[UnifiedEngine] Generation error: {e}")
            raise

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """
        Backward-compatible async generation method.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            system_prompt: Optional system prompt
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Generated text
        """
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
        )
        return await self.generate(prompt, config)

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated text.

        Yields:
            Text chunks as they are generated
        """
        config = config or GenerationConfig()
        config.stream = True
        self._stats["llm_requests"] += 1

        self._ensure_llm_loaded()

        routing = self.device_router.route(TaskType.LLM_INFERENCE)

        if routing.backend == ComputeBackend.MLX and self._mlx_engine:
            from .mlx_backend import MLXGenerationConfig

            mlx_config = MLXGenerationConfig(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            )

            async for chunk in self._mlx_engine.generate_stream(
                prompt,
                config=mlx_config,
                system_prompt=config.system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            ):
                yield chunk

        else:
            # Fallback to non-streaming
            response = await self.generate(prompt, config)
            yield response

    async def _generate_pytorch(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """Generate with PyTorch backend."""
        import torch

        model = self._mps_engine["model"]
        tokenizer = self._mps_engine["tokenizer"]
        device = self._mps_engine["device"]

        # Check if prompt is already formatted (contains role markers)
        is_preformatted = "<|im_start|>" in prompt or "<|system|>" in prompt or "<|user|>" in prompt

        if is_preformatted:
            # Prompt is already formatted by AIEngine - use directly
            formatted = prompt
        else:
            # Format messages using chat template
            messages = []
            if config.system_prompt:
                messages.append({"role": "system", "content": config.system_prompt})
            else:
                messages.append(
                    {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}
                )
            messages.append({"role": "user", "content": prompt})

            # Apply chat template
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        # Generate
        loop = asyncio.get_running_loop()

        def _generate():
            with torch.inference_mode():  # Faster than no_grad on M4
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature if config.temperature > 0 else None,
                    top_p=config.top_p if config.temperature > 0 else None,
                    do_sample=config.temperature > 0,
                    repetition_penalty=config.repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                )
                return tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )

        # FIX: Use dedicated GPU executor (2 workers) instead of default
        # unbounded executor to prevent thread explosion under load.
        from . import get_gpu_executor

        return await loop.run_in_executor(get_gpu_executor(), _generate)

    async def _get_cached_embeddings(self, texts_list: list[str]) -> tuple:
        """Get cached embeddings and identify missing texts.

        Returns:
            Tuple of (found_dict, missing_texts)
        """
        found, missing = await self._embedding_cache.get_batch(texts_list)
        if not missing:
            self._stats["cache_hits"] += len(texts_list)
        return found, missing

    async def _generate_embeddings(self, texts_list: list[str]) -> np.ndarray:
        """Generate embeddings using the appropriate backend."""
        self._ensure_embedding_loaded()
        routing = self.device_router.route(TaskType.EMBEDDING)

        if routing.backend == ComputeBackend.COREML and self._coreml_embeddings:
            result: np.ndarray = await self._coreml_embeddings.embed(texts_list)
            return result

        if self._hf_embeddings:
            return await self._embed_huggingface(texts_list)

        raise RuntimeError("No embedding backend available")

    def _combine_cached_and_new(
        self,
        original_texts: list[str],
        found: dict,
        embeddings: np.ndarray,
        missing_texts: list[str],
    ) -> np.ndarray:
        """Combine cached and newly generated embeddings."""
        result = []
        new_idx = 0
        missing_set = set(missing_texts)

        for text in original_texts:
            if text in found:
                result.append(found[text])
            else:
                result.append(embeddings[new_idx])
                if text in missing_set:
                    new_idx += 1
        return np.vstack(result)

    async def embed(
        self,
        texts: str | list[str],
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            use_cache: Whether to use embedding cache

        Returns:
            Embedding array of shape (n, dimension)
        """
        self._stats["embedding_requests"] += 1
        original_texts = [texts] if isinstance(texts, str) else texts

        found = {}
        texts_to_embed = original_texts

        # Check cache
        if use_cache:
            found, missing = await self._get_cached_embeddings(original_texts)
            if not missing:
                return np.vstack([found[t] for t in original_texts])
            texts_to_embed = missing

        # Generate embeddings for missing texts
        embeddings = await self._generate_embeddings(texts_to_embed)

        # Cache new embeddings
        if use_cache:
            for text, emb in zip(texts_to_embed, embeddings, strict=False):
                await self._embedding_cache.set(text, emb)

        # Combine if we had partial cache hits
        if found:
            return self._combine_cached_and_new(
                original_texts, found, embeddings, texts_to_embed
            )

        return embeddings

    async def _embed_huggingface(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings with HuggingFace."""
        import torch

        model = self._hf_embeddings["model"]
        tokenizer = self._hf_embeddings["tokenizer"]
        device = next(model.parameters()).device

        loop = asyncio.get_running_loop()

        def _embed():
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.inference_mode():  # Faster than no_grad on M4
                outputs = model(**inputs)

                # Mean pooling
                attention_mask = inputs["attention_mask"]
                hidden = outputs.last_hidden_state

                mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                )
                embeddings = torch.sum(hidden * mask_expanded, 1) / torch.clamp(
                    mask_expanded.sum(1), min=1e-9
                )

                return embeddings.cpu().numpy()

        return await loop.run_in_executor(None, _embed)

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            **self._stats,
            "device": self.device_router.capabilities.chip_name,
            "llm_backend": "mlx" if self._mlx_engine else "pytorch",
            "embedding_backend": "coreml" if self._coreml_embeddings else "pytorch",
            "response_cache": self._response_cache.get_stats(),
            "embedding_cache": self._embedding_cache.get_stats(),
        }

        if self._mlx_engine:
            stats["mlx_stats"] = self._mlx_engine.get_stats()

        if self._coreml_embeddings:
            stats["coreml_stats"] = self._coreml_embeddings.get_stats()

        return stats


# Global singleton
_inference_engine: UnifiedInferenceEngine | None = None
_engine_lock = threading.Lock()


def get_inference_engine(
    llm_model: str = "qwen2.5-3b",
    embedding_model: str = "all-MiniLM-L6-v2",
    auto_load: bool = False,
) -> UnifiedInferenceEngine:
    """
    Get global unified inference engine.

    Args:
        llm_model: LLM model to use
        embedding_model: Embedding model to use
        auto_load: Whether to load models immediately

    Returns:
        UnifiedInferenceEngine instance
    """
    global _inference_engine

    if _inference_engine is None:
        with _engine_lock:
            if _inference_engine is None:
                _inference_engine = UnifiedInferenceEngine(
                    llm_model=llm_model,
                    embedding_model=embedding_model,
                    auto_load=auto_load,
                )
    elif auto_load:
        # If engine exists but auto_load requested, ensure models are loaded
        _inference_engine._ensure_llm_loaded()

    return _inference_engine
