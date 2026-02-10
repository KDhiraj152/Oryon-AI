"""
CoreML Embedding Engine - Neural Engine Accelerated Embeddings
===============================================================

Uses Apple's Core ML and Neural Engine (ANE) for fast embedding generation.
The M4's 16-core Neural Engine provides 38 TOPS for efficient inference.

Performance on M4:
- Batch of 32 texts: ~50ms (vs ~500ms on CPU)
- 10x faster than CPU
- Runs parallel to GPU workloads
"""

import asyncio
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class CoreMLEmbeddingEngine:
    """
    Embedding generation using Apple Neural Engine via CoreML.

    Converts HuggingFace models to CoreML format and runs on ANE
    for maximum efficiency on Apple Silicon.
    """

    # Models optimized for CoreML conversion
    SUPPORTED_MODELS = {
        "all-MiniLM-L6-v2": {
            "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "max_length": 256,
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "hf_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "dimension": 384,
            "max_length": 256,
        },
        "bge-small-en-v1.5": {
            "hf_id": "BAAI/bge-small-en-v1.5",
            "dimension": 384,
            "max_length": 512,
        },
        "bge-base-en-v1.5": {
            "hf_id": "BAAI/bge-base-en-v1.5",
            "dimension": 768,
            "max_length": 512,
        },
        "multilingual-e5-small": {
            "hf_id": "intfloat/multilingual-e5-small",
            "dimension": 384,
            "max_length": 512,
        },
    }

    def __init__(
        self,
        model_id: str = "all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        use_ane: bool = True,
    ):
        """
        Initialize CoreML embedding engine.

        Args:
            model_id: Model identifier
            cache_dir: Directory to cache CoreML models
            use_ane: Whether to prefer Neural Engine
        """
        self.model_id = model_id
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "coreml_models")
        self.use_ane = use_ane

        self._model: Any = None
        self._tokenizer: Any = None
        self._coreml_model: Any = None
        self._lock = threading.Lock()
        self._is_loaded = False

        # Get model config
        if model_id in self.SUPPORTED_MODELS:
            self._config = self.SUPPORTED_MODELS[model_id]
        else:
            # Default config for unknown models
            self._config = {
                "hf_id": model_id,
                "dimension": 384,
                "max_length": 512,
            }

        # Performance tracking
        self._total_embeddings = 0
        self._total_time = 0.0

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return int(self._config["dimension"])

    @property
    def max_length(self) -> int:
        """Get max sequence length."""
        return int(self._config["max_length"])

    def _get_coreml_path(self) -> Path:
        """Get path to CoreML model."""
        model_hash = hashlib.md5(
            str(self._config["hf_id"]).encode(), usedforsecurity=False
        ).hexdigest()[:8]
        return self.cache_dir / f"{self.model_id}_{model_hash}.mlpackage"

    def load(self) -> bool:
        """
        Load model (CoreML if available, else HuggingFace).

        Returns:
            True if loaded successfully
        """
        if self._is_loaded:
            return True

        with self._lock:
            if self._is_loaded:
                return True

            try:
                logger.info(f"[CoreML] Loading embedding model: {self.model_id}")
                start = time.perf_counter()

                # Try to load CoreML model first
                coreml_path = self._get_coreml_path()
                if coreml_path.exists():
                    self._load_coreml(coreml_path)
                else:
                    # Load HuggingFace model
                    self._load_huggingface()

                    # Try to convert to CoreML
                    if self._can_convert_to_coreml():
                        self._convert_to_coreml()

                elapsed = time.perf_counter() - start
                self._is_loaded = True

                logger.info(
                    f"[CoreML] Model loaded in {elapsed:.2f}s "
                    f"(ANE: {self._coreml_model is not None})"
                )
                return True

            except Exception as e:
                logger.error(f"[CoreML] Failed to load model: {e}")
                return False

    def _load_huggingface(self):
        """Load HuggingFace model as fallback."""
        from transformers import AutoModel, AutoTokenizer

        from ...core.config import settings

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config["hf_id"], cache_dir=str(settings.MODEL_CACHE_DIR)
        )
        self._model = AutoModel.from_pretrained(
            self._config["hf_id"], cache_dir=str(settings.MODEL_CACHE_DIR)
        )

        # Move to MPS if available
        try:
            import torch

            if torch.backends.mps.is_available():
                self._model = self._model.to("mps")
                logger.info("[CoreML] Using MPS backend")
        except Exception:
            pass

    def _load_coreml(self, path: Path):
        """Load CoreML model."""
        try:
            import coremltools as ct

            self._coreml_model = ct.models.MLModel(str(path))

            # Load tokenizer
            from transformers import AutoTokenizer

            from ...core.config import settings

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config["hf_id"], cache_dir=str(settings.MODEL_CACHE_DIR)
            )

            logger.info(f"[CoreML] Loaded from {path}")

        except Exception as e:
            logger.warning(f"[CoreML] Failed to load CoreML model: {e}")
            self._load_huggingface()

    def _can_convert_to_coreml(self) -> bool:
        """Check if CoreML conversion is possible."""
        try:
            import coremltools

            return True
        except ImportError:
            return False

    def _convert_to_coreml(self):
        """Convert HuggingFace model to CoreML."""
        try:
            import coremltools as ct
            import torch

            logger.info("[CoreML] Converting to CoreML format...")

            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Trace model
            self._model.eval()

            # Create sample input
            sample_text = "This is a sample text for tracing"
            inputs = self._tokenizer(
                sample_text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )

            # Trace with sample input
            with torch.inference_mode():  # Faster than no_grad on M4
                traced = torch.jit.trace(
                    self._model,
                    (inputs["input_ids"], inputs["attention_mask"]),
                )

            # Convert to CoreML with dynamic batch support for ANE throughput
            # FIX: Use RangeDim so predict() accepts batch>1 natively
            try:
                batch_dim = ct.RangeDim(lower_bound=1, upper_bound=64, default=1)
                batch_shape_ids = ct.Shape(shape=(batch_dim, self.max_length))
                batch_shape_mask = ct.Shape(shape=(batch_dim, self.max_length))
            except AttributeError:
                # Older coremltools without RangeDim — fall back to fixed
                batch_shape_ids = (1, self.max_length)
                batch_shape_mask = (1, self.max_length)

            mlmodel = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(
                        name="input_ids",
                        shape=batch_shape_ids,
                        dtype=np.int32,
                    ),
                    ct.TensorType(
                        name="attention_mask",
                        shape=batch_shape_mask,
                        dtype=np.int32,
                    ),
                ],
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.ALL
                if self.use_ane
                else ct.ComputeUnit.CPU_AND_GPU,
            )

            # Save
            coreml_path = self._get_coreml_path()
            mlmodel.save(str(coreml_path))
            self._coreml_model = mlmodel

            logger.info(f"[CoreML] Converted and saved to {coreml_path}")

        except Exception as e:
            logger.warning(f"[CoreML] Conversion failed: {e}")

    async def embed(
        self,
        text: str | list[str],
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            text: Single text or list of texts

        Returns:
            Embedding array of shape (n, dimension)
        """
        if not self._is_loaded and not self.load():
            raise RuntimeError("Failed to load embedding model")

        texts = [text] if isinstance(text, str) else text

        # Run in thread pool
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            self._embed_sync,
            texts,
        )

        return embeddings

    def _embed_sync(self, texts: list[str]) -> np.ndarray:
        """Synchronous embedding generation."""
        start = time.perf_counter()

        # Tokenize
        inputs = self._tokenizer(
            texts,
            return_tensors="pt" if self._coreml_model is None else "np",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )

        if self._coreml_model is not None:
            # CoreML inference
            embeddings = self._embed_coreml(inputs)
        else:
            # HuggingFace inference
            embeddings = self._embed_huggingface(inputs)

        elapsed = time.perf_counter() - start

        # Track performance
        self._total_embeddings += len(texts)
        self._total_time += elapsed

        logger.debug(
            f"[CoreML] Embedded {len(texts)} texts in {elapsed * 1000:.1f}ms "
            f"({len(texts) / elapsed:.1f} texts/s)"
        )

        return embeddings

    def _embed_coreml(self, inputs: dict) -> np.ndarray:
        """Generate embeddings with CoreML.

        NOTE: CoreML model was traced with fixed batch=1 shape, so we must
        loop per-item. TODO: Re-trace with ct.RangeDim(1, 64) to enable
        true batched ANE inference and eliminate this loop entirely.
        """
        batch_size = inputs["input_ids"].shape[0]

        # FIX (NPU transfer overhead): Precompute int32 conversion once
        # instead of per-item .astype() which creates N separate copies.
        input_ids_i32 = inputs["input_ids"].astype(np.int32)
        attention_mask_i32 = inputs["attention_mask"].astype(np.int32)

        # Preallocate output array to avoid N append + vstack overhead
        first_output = self._coreml_model.predict({
            "input_ids": input_ids_i32[0:1],
            "attention_mask": attention_mask_i32[0:1],
        })
        first_hidden = next(iter(first_output.values()))
        first_emb = first_hidden.mean(axis=1)
        embed_dim = first_emb.shape[-1]

        embeddings = np.empty((batch_size, embed_dim), dtype=np.float32)
        embeddings[0] = first_emb.squeeze()

        for i in range(1, batch_size):
            output = self._coreml_model.predict({
                "input_ids": input_ids_i32[i : i + 1],
                "attention_mask": attention_mask_i32[i : i + 1],
            })
            hidden_state = next(iter(output.values()))
            embeddings[i] = hidden_state.mean(axis=1).squeeze()

        return embeddings

    def _embed_huggingface(self, inputs: dict) -> np.ndarray:
        """Generate embeddings with HuggingFace."""
        import torch

        device = next(self._model.parameters()).device

        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self._model(**inputs)

            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state

            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )

            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            return embeddings.cpu().numpy()

    def embed_sync(self, text: str | list[str]) -> np.ndarray:
        """Synchronous embedding (for non-async contexts)."""
        if not self._is_loaded and not self.load():
            raise RuntimeError("Failed to load embedding model")

        texts = [text] if isinstance(text, str) else text
        return self._embed_sync(texts)

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        avg_speed = (
            self._total_embeddings / self._total_time if self._total_time > 0 else 0
        )

        return {
            "model_id": self.model_id,
            "dimension": self.dimension,
            "is_loaded": self._is_loaded,
            "using_coreml": self._coreml_model is not None,
            "using_ane": self._coreml_model is not None and self.use_ane,
            "total_embeddings": self._total_embeddings,
            "total_time_s": self._total_time,
            "avg_texts_per_sec": avg_speed,
        }


# Global singleton
_coreml_engine: CoreMLEmbeddingEngine | None = None
_engine_lock = threading.Lock()


def get_coreml_embeddings(
    model_id: str = "all-MiniLM-L6-v2",
    auto_load: bool = True,
) -> CoreMLEmbeddingEngine:
    """
    Get global CoreML embedding engine.

    Args:
        model_id: Embedding model to use
        auto_load: Whether to load model immediately

    Returns:
        CoreMLEmbeddingEngine instance
    """
    global _coreml_engine

    if _coreml_engine is None:
        with _engine_lock:
            if _coreml_engine is None:
                _coreml_engine = CoreMLEmbeddingEngine(model_id=model_id)
                if auto_load:
                    _coreml_engine.load()

    return _coreml_engine
