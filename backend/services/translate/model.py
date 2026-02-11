"""IndicTrans2 Translation Model - Best-in-class for Indian Languages.

Optimal 2025 Model Stack: ai4bharat/indictrans2-en-indic-1B
- Purpose-built for Indian languages
- Optimized for Apple Silicon M4 (MPS GPU acceleration)
- Supports 22 Indian languages
- Best quality for Indic language translation

Hardware Optimization:
- MPS (Metal Performance Shaders): Primary compute for M4 GPU
- CPU fallback for unsupported operations
- FP16 precision on MPS for 2x memory efficiency
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import torch  # For type hints only

from ...core.config import settings

# M4 Hardware optimization imports
try:
    from ...core.optimized import get_memory_pool
    from ...core.optimized.device_router import (
        TaskType,
        get_device_router,
        get_resource_manager,
    )

    HARDWARE_OPT_AVAILABLE = True
except ImportError:
    HARDWARE_OPT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Enable MPS fallback for operations not supported on Metal
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Try to import IndicTransToolkit for proper preprocessing
try:
    from IndicTransToolkit.processor import IndicProcessor

    INDIC_PROCESSOR_AVAILABLE = True
except ImportError:
    INDIC_PROCESSOR_AVAILABLE = False
    logger.warning(
        "IndicTransToolkit not available - translation quality may be reduced"
    )


@dataclass
class TranslationResult:
    """Result of translation operation."""

    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: float
    metadata: dict


class IndicTrans2:
    """
    IndicTrans2 translation model wrapper.

    Best-in-class for English ↔ Indian language translation.
    Supports: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati,
              Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu, etc.
    """

    # IndicTrans2 language codes (BCP-47 format)
    LANG_CODES = {
        # Indian languages
        "Hindi": "hin_Deva",
        "Tamil": "tam_Taml",
        "Telugu": "tel_Telu",
        "Bengali": "ben_Beng",
        "Marathi": "mar_Deva",
        "Gujarati": "guj_Gujr",
        "Kannada": "kan_Knda",
        "Malayalam": "mal_Mlym",
        "Punjabi": "pan_Guru",
        "Odia": "ory_Orya",
        "Assamese": "asm_Beng",
        "Urdu": "urd_Arab",
        "Konkani": "kok_Deva",
        "Maithili": "mai_Deva",
        "Nepali": "npi_Deva",
        "Sanskrit": "san_Deva",
        "Sindhi": "snd_Arab",
        "Santali": "sat_Olck",
        "Kashmiri": "kas_Arab",
        "Dogri": "doi_Deva",
        "Manipuri": "mni_Beng",
        "Bodo": "brx_Deva",
        # English
        "English": "eng_Latn",
    }

    # Reverse mapping
    CODE_TO_LANG = {v: k for k, v in LANG_CODES.items()}

    def __init__(self, model_id: str | None = None, device: str | None = None):
        """
        Initialize IndicTrans2 model.

        Args:
            model_id: Model identifier (default from config)
            device: Device to use (auto-detected if not specified)
        """
        import torch  # Lazy import — avoid GPU memory init at module load time

        self.model_id = model_id or settings.TRANSLATION_MODEL_ID

        # Use hardware optimizer for intelligent device routing if available
        if device is None and HARDWARE_OPT_AVAILABLE:
            try:
                router = get_device_router()
                routing = router.route(TaskType.TRANSLATION)
                self.device = routing.device_str
                logger.info(
                    f"IndicTrans2: Using {self.device} (via hardware optimizer, speedup: {routing.estimated_speedup}x)"
                )
            except Exception as e:
                logger.debug(f"Hardware optimizer failed, using fallback: {e}")
                device = None  # Fall through to manual detection

        # Fallback auto-detect device
        if device is None and not hasattr(self, "device"):
            from backend.core.hal import get_device
            self.device = get_device()
        elif device is not None:
            self.device = device

        self._model = None
        self._tokenizer = None
        self._processor = None  # IndicProcessor for pre/post processing

        logger.info(f"IndicTrans2 initialized: {self.model_id} on {self.device}")

    def _load_model(self):
        """Lazy load the model with memory coordination."""
        import torch  # Lazy import

        if self._model is not None:
            return

        logger.info(f"Loading IndicTrans2 model: {self.model_id}")

        # Memory coordination: reserve ~2GB for IndicTrans2 1B model
        try:
            from backend.core.optimized.memory_coordinator import get_memory_coordinator
            coordinator = get_memory_coordinator()
            coordinator.acquire_memory_sync("indictrans2", 2.0, priority=1).__enter__()
        except (ImportError, Exception) as e:
            logger.debug(f"Memory coordinator not available for IndicTrans2: {e}")
            coordinator = None

        try:
            # Try loading with AI4Bharat's IndicTrans2 library
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            # M4 Optimization: Enable MPS fallback
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=str(settings.MODEL_CACHE_DIR),
            )

            # Initialize IndicProcessor for proper pre/post processing
            if INDIC_PROCESSOR_AVAILABLE:
                self._processor = IndicProcessor(inference=True)
                logger.info("IndicProcessor initialized for translation preprocessing")
            else:
                logger.warning(
                    "IndicProcessor not available - using basic tokenization"
                )

            # Load model with device-appropriate settings
            load_kwargs = {
                "trust_remote_code": True,
                "cache_dir": str(settings.MODEL_CACHE_DIR),
                "low_cpu_mem_usage": True,  # M4: Essential for 16GB unified memory
            }

            if self.device == "cuda" and settings.USE_QUANTIZATION:
                try:
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    load_kwargs["device_map"] = "auto"
                except ImportError:
                    load_kwargs["torch_dtype"] = torch.float16
                    load_kwargs["device_map"] = "auto"
            elif self.device == "cuda":
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            elif self.device == "mps":
                # M4 Optimization: float16 for MPS
                load_kwargs["torch_dtype"] = torch.float16

            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, **load_kwargs
            )

            # Move to device if not using device_map
            if "device_map" not in load_kwargs:
                self._model = self._model.to(self.device)

            self._model.eval()

            # M4 Memory cleanup
            if self.device == "mps":
                torch.mps.empty_cache()

            logger.info(f"IndicTrans2 model loaded on {self.device} (M4 optimized)")

            # Register with memory coordinator for eviction tracking
            if coordinator is not None:
                try:
                    coordinator.register_model(
                        "indictrans2", self._model, self._unload_model
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Failed to load IndicTrans2: {e}")
            raise RuntimeError(f"IndicTrans2 model loading failed: {e}")

    def _unload_model(self):
        """Unload model to free memory (called by coordinator)."""
        import gc
        self._model = None
        self._tokenizer = None
        self._processor = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        logger.info("IndicTrans2 model unloaded")

    def get_lang_code(self, language: str) -> str:
        """Get IndicTrans2 language code."""
        return self.LANG_CODES.get(language, language)

    def translate(
        self,
        text: str,
        source_lang: str = "English",
        target_lang: str = "Hindi",
        max_length: int | None = None,
    ) -> TranslationResult:
        """
        Translate text between languages.

        Args:
            text: Source text to translate
            source_lang: Source language name
            target_lang: Target language name
            max_length: Maximum output length

        Returns:
            TranslationResult with translated text
        """
        import torch  # Lazy import

        self._load_model()

        if not text or not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=1.0,
                metadata={"error": "empty_input"},
            )

        max_length = max_length or settings.TRANSLATION_MAX_LENGTH

        src_code = self.get_lang_code(source_lang)
        tgt_code = self.get_lang_code(target_lang)

        try:
            # Use IndicProcessor for proper preprocessing if available
            if self._processor is not None:
                # Preprocess batch (even for single text)
                preprocessed = self._processor.preprocess_batch(
                    [text],
                    src_lang=src_code,
                    tgt_lang=tgt_code,
                )
                input_text = preprocessed[0]
            else:
                input_text = text

            # Tokenize
            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )

            # Move to device
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Generate translation - M4 optimized with inference_mode
            # Note: use_cache=False required due to IndicTrans2 decoder compatibility
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    min_length=0,
                    early_stopping=True,
                    num_return_sequences=1,
                    use_cache=False,  # Required for IndicTrans2 compatibility
                )

            # Note: torch.mps.empty_cache() removed from hot path — MPS manages
            # its cache efficiently; per-call flush adds ~5ms overhead with no benefit.

            # Decode
            translated_raw = self._tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            # Use IndicProcessor for proper postprocessing if available
            if self._processor is not None:
                translated_text = self._processor.postprocess_batch(
                    [translated_raw], lang=tgt_code
                )[0]
            else:
                translated_text = translated_raw.strip()

            return TranslationResult(
                source_text=text,
                translated_text=translated_text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=0.95,  # IndicTrans2 is highly accurate
                metadata={
                    "model": self.model_id,
                    "src_code": src_code,
                    "tgt_code": tgt_code,
                    "processor_used": self._processor is not None,
                },
            )

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=0.0,
                metadata={"error": str(e)},
            )

    async def translate_async(
        self,
        text: str,
        source_lang: str = "English",
        target_lang: str = "Hindi",
        max_length: int | None = None,
    ) -> TranslationResult:
        """Async wrapper for translation."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.translate(text, source_lang, target_lang, max_length)
        )

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str = "English",
        target_lang: str = "Hindi",
        max_length: int | None = None,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts efficiently.

        Args:
            texts: List of texts to translate
            source_lang: Source language
            target_lang: Target language
            max_length: Maximum output length per text

        Returns:
            List of TranslationResults
        """
        import torch  # Lazy import

        self._load_model()

        if not texts:
            return []

        max_length = max_length or settings.TRANSLATION_MAX_LENGTH
        src_code = self.get_lang_code(source_lang)
        tgt_code = self.get_lang_code(target_lang)

        results = []

        try:
            # Preprocess batch using IndicProcessor if available
            if self._processor is not None:
                preprocessed = self._processor.preprocess_batch(
                    texts,
                    src_lang=src_code,
                    tgt_lang=tgt_code,
                )
            else:
                preprocessed = texts

            # Batch tokenize
            inputs = self._tokenizer(
                preprocessed,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Generate
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    min_length=0,
                    early_stopping=True,
                    num_return_sequences=1,
                    use_cache=False,  # Required for IndicTrans2 compatibility
                )

            # Decode all
            translated_raw = self._tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # Postprocess using IndicProcessor if available
            if self._processor is not None:
                translated_texts = self._processor.postprocess_batch(
                    translated_raw, lang=tgt_code
                )
            else:
                translated_texts = [t.strip() for t in translated_raw]

            for src_text, tgt_text in zip(texts, translated_texts, strict=False):
                results.append(
                    TranslationResult(
                        source_text=src_text,
                        translated_text=tgt_text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        confidence=0.95,
                        metadata={
                            "model": self.model_id,
                            "processor_used": self._processor is not None,
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            # Return individual failures
            for text in texts:
                results.append(
                    TranslationResult(
                        source_text=text,
                        translated_text=text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        confidence=0.0,
                        metadata={"error": str(e)},
                    )
                )

        return results

    @property
    def supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        return list(self.LANG_CODES.keys())

    def is_language_supported(self, language: str) -> bool:
        """Check if language is supported."""
        return language in self.LANG_CODES

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def unload(self) -> None:
        """Unload model and free memory (for coordinated shutdown)."""
        if self._model is not None:
            logger.info("Unloading IndicTrans2 model...")

            # Release memory via coordinator
            try:
                from ...core.optimized.memory_coordinator import get_memory_coordinator

                coordinator = get_memory_coordinator()
                coordinator.release("translation")
            except ImportError:
                pass

            # Delete model references
            del self._model
            self._model = None

            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None

            if self._processor is not None:
                del self._processor
                self._processor = None

            # Free MPS cache
            try:
                import torch
                if self.device == "mps":
                    torch.mps.empty_cache()
            except Exception:
                pass

            # Force garbage collection
            import gc

            gc.collect()

            logger.info("IndicTrans2 model unloaded")


# Thread-safe singleton with memory coordinator integration
import threading

_translator_lock = threading.Lock()
_translator_instance: IndicTrans2 | None = None


def get_translator() -> IndicTrans2:
    """Get or create translator singleton (thread-safe with memory coordination)."""
    global _translator_instance

    if _translator_instance is not None and _translator_instance.is_loaded:
        return _translator_instance

    with _translator_lock:
        # Double-check pattern
        if _translator_instance is not None and _translator_instance.is_loaded:
            return _translator_instance

        # Create new instance with memory coordinator
        try:
            import asyncio

            from ...core.optimized.memory_coordinator import get_memory_coordinator

            coordinator = get_memory_coordinator()

            # Try to acquire memory (use sync version if not in async context)
            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                # We're in async context - proceed optimistically
                # Memory will be managed at the async caller level
                acquired = True
            except RuntimeError:
                # Not in async context - use sync acquisition
                acquired = coordinator.try_acquire_sync("translation")

            if not acquired:
                logger.warning(
                    "Could not acquire memory for translation model - "
                    "memory pressure may occur"
                )
        except ImportError:
            pass  # Memory coordinator not available

        _translator_instance = IndicTrans2()
        return _translator_instance


def unload_translator() -> None:
    """Unload the translator singleton."""
    global _translator_instance

    with _translator_lock:
        if _translator_instance is not None:
            _translator_instance.unload()
            _translator_instance = None


# Export
__all__ = ["IndicTrans2", "TranslationResult", "get_translator", "unload_translator"]
