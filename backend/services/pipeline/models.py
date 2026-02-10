"""
AI Pipeline Models Module - Centralized model loading and management.

This module provides:
- Thread-safe lazy loading of AI models
- M4 Mac / Apple Silicon optimizations
- ANE (Apple Neural Engine) support where available
- Memory-efficient model management

Models supported:
- BGE-M3: Embeddings
- BGE-Reranker-v2-M3: Reranking
- Gemma-2-2B-IT: Response validation
- IndicTrans2-1B: Translation
"""

import os
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional, TypedDict

from ...core.optimized import M4_PERF_CONFIG, get_device_router
from ...monitoring import get_logger

logger = get_logger(__name__)


# ==================== Device Helper Functions ====================


def get_optimal_device(model_name: str, model_params_b: float, task: str) -> tuple:
    """Get optimal device using new device router.

    Returns (device_str, config) for compatibility with old code.
    """
    router = get_device_router()
    device = router.get_device_for_task(task)
    return device, M4_PERF_CONFIG


def get_torch_device(device_str: str):
    """Convert device string to torch device."""
    import torch

    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ==================== Constants ====================

MODEL_GEMMA_2B = "google/gemma-2-2b-it"
MODEL_INDICTRANS2 = "ai4bharat/indictrans2-en-indic-1B"
MODEL_BGE_M3 = "BAAI/bge-m3"
MODEL_BGE_RERANKER = "BAAI/bge-reranker-v2-m3"


# ==================== Type Definitions ====================


class ValidatorModel(TypedDict):
    """Type definition for validator model dict."""

    model: Any
    tokenizer: Any
    device: str


class TranslatorModel(TypedDict):
    """Type definition for translator model dict."""

    model: Any
    tokenizer: Any
    device: str


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    model_id: str
    model_params_b: float
    task: str
    use_ane: bool = True


# ==================== Model Registry ====================


class PipelineModelRegistry:
    """Thread-safe registry for lazy-loaded AI models.

    Provides:
    - Singleton pattern with thread safety
    - Lazy loading on first access
    - Automatic device selection (ANE/MPS/CPU)
    - Memory cleanup after loading
    """

    _instance: Optional["PipelineModelRegistry"] = None
    _lock = Lock()

    def __new__(cls) -> "PipelineModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: dict[str, Any] = {
            "embedding": None,
            "reranker": None,
            "validator": None,
            "translator": None,
        }
        self._model_locks: dict[str, Lock] = {
            "embedding": Lock(),
            "reranker": Lock(),
            "validator": Lock(),
            "translator": Lock(),
        }
        self._hw_optimizer = None
        self._device_router = None
        self._init_device_router()
        self._initialized = True

    def _init_device_router(self) -> None:
        """Initialize device router for device selection."""
        try:
            self._device_router = get_device_router()
            logger.info(f"Device router initialized: {self._device_router.device_type}")
        except Exception as e:
            logger.warning(f"Device router init failed: {e}")
            self._device_router = None

    def _setup_mps_environment(self) -> None:
        """Configure environment for MPS (Metal Performance Shaders)."""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    def _cleanup_mps_memory(self, device_type: str) -> None:
        """Clean up MPS memory after model loading."""
        if device_type == "mps":
            import torch

            torch.mps.empty_cache()

    # ==================== Embedding Model ====================

    def get_embedding_model(self) -> Any | None:
        """Get BGE-M3 embedding model with lazy loading.

        Optimized for M4 Mac with ANE support.
        Falls back to MPS/CPU if ANE unavailable.
        """
        with self._model_locks["embedding"]:
            if self._models["embedding"] is not None:
                return self._models["embedding"]

            try:
                self._models["embedding"] = self._load_embedding_model()
            except Exception as e:
                logger.warning(f"Failed to load BGE-M3: {e}")

        return self._models["embedding"]

    def _load_embedding_model(self) -> Any:
        """Load embedding model with optimal device selection."""
        self._setup_mps_environment()

        device, config = get_optimal_device(
            MODEL_BGE_M3, model_params_b=0.5, task="embedding"
        )

        # Try ANE first for embeddings (ideal for small models)
        if device == "ane":
            ane_model = self._try_load_ane_embedding()
            if ane_model is not None:
                return ane_model

        # Fallback to MPS/CPU via sentence_transformers
        from sentence_transformers import SentenceTransformer

        from ...core.config import settings

        torch_device = get_torch_device(device if device != "ane" else "mps")

        logger.info(f"Loading BGE-M3 on {torch_device} with config: {config}")

        model = SentenceTransformer(
            MODEL_BGE_M3,
            device=str(torch_device),
            cache_folder=str(settings.MODEL_CACHE_DIR),
        )

        self._cleanup_mps_memory(torch_device.type)
        logger.info(f"BGE-M3 loaded successfully on {torch_device}")

        return model

    def _try_load_ane_embedding(self) -> Any | None:
        """Try to load embedding model on ANE.

        Note: ANE is now handled by device_router via CoreML compilation.
        For embeddings, we use MPS which is already highly optimized.
        """
        # ANE support now integrated via device router
        # For embeddings, MPS provides best performance on M4
        return None

    # ==================== Reranker Model ====================

    def get_reranker_model(self) -> Any | None:
        """Get BGE-Reranker model with lazy loading.

        Optimized for M4 Mac with ANE support.
        """
        with self._model_locks["reranker"]:
            if self._models["reranker"] is not None:
                return self._models["reranker"]

            try:
                self._models["reranker"] = self._load_reranker_model()
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")

        return self._models["reranker"]

    def _load_reranker_model(self) -> Any:
        """Load reranker model with optimal device selection."""

        device, _ = get_optimal_device(
            MODEL_BGE_RERANKER, model_params_b=0.5, task="reranking"
        )

        # Try ANE first for reranking
        if device == "ane":
            ane_model = self._try_load_ane_reranker()
            if ane_model is not None:
                return ane_model

        # Fallback to MPS/CPU via CrossEncoder
        from sentence_transformers import CrossEncoder

        torch_device = get_torch_device(device if device != "ane" else "mps")

        logger.info(f"Loading BGE-Reranker on {torch_device}")
        model = CrossEncoder(MODEL_BGE_RERANKER, device=str(torch_device))
        logger.info(f"BGE-Reranker loaded on {torch_device}")

        return model

    def _try_load_ane_reranker(self) -> Any | None:
        """Try to load reranker model on ANE.

        Note: ANE is now handled by device_router via CoreML compilation.
        For reranking, we use MPS which is already highly optimized.
        """
        # ANE support now integrated via device router
        # For reranking, MPS provides best performance on M4
        return None

    # ==================== Validator Model ====================

    def get_validator_model(self) -> ValidatorModel | None:
        """Get Gemma-2-2B-IT validator model with lazy loading.

        Used for response quality validation.
        """
        with self._model_locks["validator"]:
            if self._models["validator"] is not None:
                return self._models["validator"]

            try:
                self._models["validator"] = self._load_validator_model()
            except Exception as e:
                logger.warning(f"Failed to load Gemma validator: {e}")

        return self._models["validator"]

    def _load_validator_model(self) -> ValidatorModel:
        """Load Gemma-2-2B-IT for validation."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._setup_mps_environment()

        device, _ = get_optimal_device(
            MODEL_GEMMA_2B, model_params_b=2.0, task="validation"
        )
        torch_device = get_torch_device(device)

        from ...core.config import settings

        logger.info(f"Loading Gemma-2-2B-IT validator on {torch_device}")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_GEMMA_2B, cache_dir=str(settings.MODEL_CACHE_DIR)
        )

        # M4 Optimization: float16 + low_cpu_mem_usage
        dtype = torch.float16 if torch_device.type != "cpu" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_GEMMA_2B,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=str(settings.MODEL_CACHE_DIR),
        )

        if torch_device.type in ("mps", "cpu"):
            model = model.to(torch_device)

        model.eval()  # Set eval mode for inference

        self._cleanup_mps_memory(torch_device.type)

        logger.info(f"Gemma validator loaded on {torch_device} (dtype: {dtype})")

        return ValidatorModel(
            model=model, tokenizer=tokenizer, device=str(torch_device)
        )

    # ==================== Translator Model ====================

    def get_translator_model(self) -> TranslatorModel | None:
        """Get IndicTrans2 translator model with lazy loading.

        Used for English to Indian languages translation.
        """
        with self._model_locks["translator"]:
            if self._models["translator"] is not None:
                return self._models["translator"]

            try:
                self._models["translator"] = self._load_translator_model()
            except Exception as e:
                logger.warning(f"Failed to load IndicTrans2: {e}")

        return self._models["translator"]

    def _load_translator_model(self) -> TranslatorModel:
        """Load IndicTrans2 for translation."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device, config = get_optimal_device(
            MODEL_INDICTRANS2, model_params_b=1.0, task="generation"
        )
        torch_device = get_torch_device(device)

        from ...core.config import settings

        logger.info(f"Loading IndicTrans2 on {torch_device}")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_INDICTRANS2,
            trust_remote_code=True,
            cache_dir=str(settings.MODEL_CACHE_DIR),
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_INDICTRANS2,
            trust_remote_code=True,
            torch_dtype=config.get("torch_dtype", torch.float16),
            cache_dir=str(settings.MODEL_CACHE_DIR),
        )

        if torch_device.type == "mps":
            model = model.to(torch_device)

        logger.info(f"IndicTrans2 loaded on {torch_device}")

        return TranslatorModel(
            model=model, tokenizer=tokenizer, device=str(torch_device)
        )

    # ==================== Utility Methods ====================

    def clear_model(self, model_name: str) -> None:
        """Clear a specific model from memory."""
        with self._model_locks.get(model_name, Lock()):
            if model_name in self._models:
                self._models[model_name] = None
                logger.info(f"Cleared model: {model_name}")

    def clear_all_models(self) -> None:
        """Clear all models from memory."""
        for model_name in self._models:
            self.clear_model(model_name)

        # Force garbage collection
        import gc

        gc.collect()

        # Clear MPS cache if available
        try:
            import torch

            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

        logger.info("All models cleared from memory")

    def get_loaded_models(self) -> dict[str, bool]:
        """Get status of which models are loaded."""
        return {name: model is not None for name, model in self._models.items()}


# ==================== Module-level convenience functions ====================

_registry: PipelineModelRegistry | None = None


def _get_registry() -> PipelineModelRegistry:
    """Get or create the model registry singleton."""
    global _registry
    if _registry is None:
        _registry = PipelineModelRegistry()
    return _registry


def get_embedding_model() -> Any | None:
    """Get the embedding model (lazy loaded)."""
    return _get_registry().get_embedding_model()


def get_reranker_model() -> Any | None:
    """Get the reranker model (lazy loaded)."""
    return _get_registry().get_reranker_model()


def get_validator_model() -> ValidatorModel | None:
    """Get the validator model (lazy loaded)."""
    return _get_registry().get_validator_model()


def get_translator_model() -> TranslatorModel | None:
    """Get the translator model (lazy loaded)."""
    return _get_registry().get_translator_model()


# Pre-compiled regex for score parsing
_RE_SCORE = re.compile(r"SCORE:\s*(\d+)/10")


def validate_response(
    response: str, question: str, grade_level: int = 8
) -> dict[str, Any]:
    """Validate response quality using Gemma-2-2B-IT.

    Uses the validator model to assess educational response quality.
    Returns validation result with score and reasoning.

    Args:
        response: The generated response to validate
        question: The original question
        grade_level: Target grade level (default: 8)

    Returns:
        Dict with 'valid' (bool), 'score' (float), 'reason' (str)
    """
    validator = get_validator_model()
    if not validator:
        return {"valid": True, "score": 0.8, "reason": "Validator not available"}

    try:
        import torch

        prompt = f"""Rate this educational response for Class {grade_level}:

Q: {question[:200]}
A: {response[:800]}

Score 1-10 for accuracy, clarity, completeness.
Output: SCORE: X/10 | ISSUE: none or brief issue"""

        # validator is a TypedDict - access with dict keys
        tokenizer = validator["tokenizer"]
        model = validator["model"]
        device = validator["device"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        if device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        # Use inference_mode for faster inference
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=False,  # Greedy for speed
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # KV cache for faster generation
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Memory cleanup
        if device == "mps":
            torch.mps.empty_cache()

        # Parse score using pre-compiled regex
        score_match = _RE_SCORE.search(result)
        score = int(score_match.group(1)) / 10 if score_match else 0.7

        return {
            "valid": score >= 0.6,
            "score": score,
            "reason": "passed" if score >= 0.6 else result[-50:],
        }
    except Exception as e:
        logger.warning(f"Validation error: {e}")
        return {"valid": True, "score": 0.7, "reason": str(e)}


async def refine_response(
    response: str, validation_result: dict[str, Any], llm_client: Any
) -> str:
    """Refine response if validation failed using LLM.

    Args:
        response: The original response to refine
        validation_result: Result from validate_response()
        llm_client: LLM client instance for generation

    Returns:
        Refined response or original if no refinement needed
    """
    if validation_result.get("valid", True):
        return response

    try:
        prompt = f"""Fix this educational response. Make it clearer and more accurate. Keep markdown formatting.

Issue: {validation_result.get("reason", "unclear")}

Response to fix:
{response}

Fixed response:"""

        refined = await llm_client.generate_async(
            prompt=prompt, max_tokens=4096, temperature=0.5
        )  # Increased
        return refined if refined else response
    except Exception as e:
        logger.warning(f"Refine failed: {e}")
        return response


def clear_all_models() -> None:
    """Clear all models from memory."""
    _get_registry().clear_all_models()


def get_model_status() -> dict[str, bool]:
    """Get status of which models are loaded."""
    return _get_registry().get_loaded_models()


# Export for backward compatibility
__all__ = [
    "MODEL_BGE_M3",
    "MODEL_BGE_RERANKER",
    "MODEL_GEMMA_2B",
    "MODEL_INDICTRANS2",
    "PipelineModelRegistry",
    "clear_all_models",
    "get_embedding_model",
    "get_model_status",
    "get_reranker_model",
    "get_translator_model",
    "get_validator_model",
    "refine_response",
    "validate_response",
]
