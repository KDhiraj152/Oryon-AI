"""
MMS-TTS Service - Facebook's Massively Multilingual Speech TTS.

Optimal TTS for Indian languages with 1100+ language support.

Model: facebook/mms-tts-{lang_code}
- Hindi (hin), Tamil (tam), Telugu (tel), Kannada (kan), Malayalam (mal)
- Bengali (ben), Marathi (mar), Gujarati (guj), Punjabi (pan), Odia (ori)
- English (eng), and 1100+ more languages

Features:
- Native Indian language support (best quality)
- ~30MB per language model (lazy loaded)
- GPU acceleration (CUDA/MPS)
- Offline after initial download
- Audio caching for repeated synthesis (M4 optimization)

Updated: 2025-12-03 - Switched to fast xxhash for cache keys
"""

import contextlib
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from backend.utils.hashing import fast_hash

logger = logging.getLogger(__name__)

# Check MMS-TTS availability
MMS_TTS_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, VitsModel

    MMS_TTS_AVAILABLE = True
    logger.info("MMS-TTS (Facebook) available - Best for Indian languages")
except ImportError as e:
    logger.info(f"MMS-TTS not available: {e}")

# M4 Hardware optimization imports
DEVICE_ROUTER_AVAILABLE = False
try:
    from ...core.optimized.device_router import TaskType, get_device_router

    DEVICE_ROUTER_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# M4 OPTIMIZATION: Audio Cache for repeated synthesis
# ============================================================================
class AudioCache:
    """LRU cache for synthesized audio to avoid redundant TTS inference."""

    def __init__(self, max_size: int = 500, max_bytes: int = 100 * 1024 * 1024):
        """
        Args:
            max_size: Maximum number of cached audio items
            max_bytes: Maximum total cache size in bytes (default: 100MB)
        """
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, language: str) -> str:
        """Generate cache key from text and language. Uses fast xxhash if available."""
        content = f"{text}|{language}"
        return fast_hash(content, length=32)

    def get(self, text: str, language: str) -> bytes | None:
        """Get cached audio if available."""
        key = self._make_key(text, language)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)  # LRU: move to end
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, text: str, language: str, audio: bytes) -> None:
        """Cache synthesized audio."""
        key = self._make_key(text, language)
        audio_size = len(audio)

        with self._lock:
            # Evict if needed to make room
            while (
                self._current_bytes + audio_size > self._max_bytes
                or len(self._cache) >= self._max_size
            ) and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._current_bytes -= len(evicted)

            self._cache[key] = audio
            self._current_bytes += audio_size

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "bytes": self._current_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0


# Global audio cache (shared across TTS instances)
_audio_cache = AudioCache()


class MMSTTSService:
    """
    Facebook MMS-TTS (Massively Multilingual Speech) service.

    Supports 1100+ languages with excellent Indian language coverage.
    Uses VITS architecture for high-quality neural TTS.
    """

    # Indian language models (ISO 639-1 -> MMS model suffix)
    LANGUAGE_MAP = {
        # Indian Languages
        "hi": "hin",  # Hindi
        "ta": "tam",  # Tamil
        "te": "tel",  # Telugu
        "kn": "kan",  # Kannada
        "ml": "mal",  # Malayalam
        "bn": "ben",  # Bengali
        "mr": "mar",  # Marathi
        "gu": "guj",  # Gujarati
        "pa": "pan",  # Punjabi
        "or": "ory",  # Odia (corrected from 'ori')
        "as": "asm",  # Assamese
        "ur": "urd",  # Urdu
        "sa": "san",  # Sanskrit
        "ne": "nep",  # Nepali
        "sd": "snd",  # Sindhi
        "ks": "kas",  # Kashmiri
        # International
        "en": "eng",  # English
        "es": "spa",  # Spanish
        "fr": "fra",  # French
        "de": "deu",  # German
        "pt": "por",  # Portuguese
        "ru": "rus",  # Russian
        "ar": "ara",  # Arabic
        "zh": "cmn",  # Mandarin Chinese
        "ja": "jpn",  # Japanese
        "ko": "kor",  # Korean
    }

    def __init__(self):
        if not MMS_TTS_AVAILABLE:
            raise ImportError("MMS-TTS requires: pip install transformers torch")

        self.models: dict[str, Any] = {}
        self.tokenizers: dict[str, Any] = {}
        self.device: str | None = None
        self.supported_languages: list[str] = list(self.LANGUAGE_MAP.keys())

    def _get_device(self) -> str:
        """Get optimal device for inference - M4 Mac optimized using device router."""
        if self.device:
            return self.device

        import os

        import torch

        # Use device router for intelligent routing if available
        if DEVICE_ROUTER_AVAILABLE:
            try:
                router = get_device_router()
                routing = router.route(TaskType.TTS)
                self.device = routing.device_str
                logger.info(
                    f"MMS-TTS: Using {self.device} (via device router, speedup: {routing.estimated_speedup}x)"
                )
                return self.device
            except Exception as e:
                logger.debug(f"Device router failed, using fallback: {e}")

        # Fallback to manual detection
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("MMS-TTS: Using CUDA")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            # Enable MPS fallback for any unsupported ops in VITS
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            logger.info("MMS-TTS: Using MPS (Apple M4 GPU)")
        else:
            self.device = "cpu"
            logger.info("MMS-TTS: Using CPU")
        return self.device

    def _get_model_id(self, language: str) -> str:
        """Get the MMS model ID for a language."""
        lang = language.lower()[:2]
        mms_code = self.LANGUAGE_MAP.get(lang, "eng")
        return f"facebook/mms-tts-{mms_code}"

    def load_model(self, language: str = "en"):
        """Load MMS-TTS model for a specific language (lazy loading) - M4 optimized.

        Memory coordination added to prevent OOM from concurrent model loading.
        """
        lang = language.lower()[:2]

        if lang in self.models:
            return

        import torch
        from transformers import AutoTokenizer, VitsModel

        from ...core.config import settings

        model_id = self._get_model_id(lang)
        device = self._get_device()
        cache_dir = str(settings.MODEL_CACHE_DIR)

        # Memory coordination: reserve ~0.5GB per TTS language model
        coordinator = None
        try:
            from ...core.optimized.memory_coordinator import get_memory_coordinator
            coordinator = get_memory_coordinator()
            coordinator.try_acquire_sync(f"mms_tts_{lang}")
        except (ImportError, Exception) as e:
            logger.debug(f"Memory coordinator not available for MMS-TTS: {e}")

        logger.info(f"Loading MMS-TTS: {model_id} on {device}...")

        self.tokenizers[lang] = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir
        )

        # M4 Optimization: Load with float16 for MPS, use low_cpu_mem_usage
        if device == "mps":
            self.models[lang] = VitsModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
            ).to(device)
            self.models[lang].eval()  # Set eval mode
        else:
            self.models[lang] = VitsModel.from_pretrained(
                model_id, low_cpu_mem_usage=True, cache_dir=cache_dir
            ).to(device)
            self.models[lang].eval()

        # Register with memory coordinator after successful load
        if coordinator:
            try:
                coordinator.register_model(
                    f"mms_tts_{lang}",
                    self.models[lang],
                    unload_fn=lambda l=lang: self._unload_language(l),
                )
            except Exception as e:
                logger.debug(f"Memory coordinator registration failed: {e}")

        # Clear MPS cache after loading
        if device == "mps":
            torch.mps.empty_cache()

        logger.info(f"MMS-TTS loaded for {lang} on {device}")

    def _unload_language(self, lang: str) -> None:
        """Unload a specific language model (called by memory coordinator eviction)."""
        import gc
        model = self.models.pop(lang, None)
        self.tokenizers.pop(lang, None)
        del model
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        logger.info(f"MMS-TTS unloaded language: {lang}")

    def _normalize_language(self, language: str) -> str:
        """Normalize language code to 2-letter format."""
        lang = language.lower()
        if lang in [
            "hindi",
            "tamil",
            "telugu",
            "bengali",
            "marathi",
            "gujarati",
            "kannada",
            "malayalam",
            "punjabi",
            "odia",
            "urdu",
            "english",
        ]:
            lang_map = {
                "hindi": "hi",
                "tamil": "ta",
                "telugu": "te",
                "bengali": "bn",
                "marathi": "mr",
                "gujarati": "gu",
                "kannada": "kn",
                "malayalam": "ml",
                "punjabi": "pa",
                "odia": "or",
                "urdu": "ur",
                "english": "en",
            }
            lang = lang_map.get(lang, "en")
        else:
            lang = lang[:2]

        if lang not in self.LANGUAGE_MAP:
            logger.warning(f"Language {lang} not supported, using English")
            lang = "en"

        return lang

    @staticmethod
    def _encode_wav(waveform) -> bytes:
        """Encode a float32 numpy waveform to WAV bytes (CPU-bound)."""
        import io

        import numpy as np
        import soundfile as sf

        waveform = np.asarray(waveform, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, waveform, samplerate=16000, format="WAV")
        return buf.getvalue()

    def _infer_waveform(self, text: str, lang: str):
        """Run GPU inference, return raw float32 numpy waveform."""
        import torch

        self.load_model(lang)
        device = self._get_device()

        inputs = self.tokenizers[lang](text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.models[lang](**inputs)
            return output.waveform[0].cpu().float().numpy()

    def _synthesize_uncached(self, text: str, lang: str) -> bytes:
        """Internal synthesis without cache lookup."""
        waveform = self._infer_waveform(text, lang)
        return self._encode_wav(waveform)

    def synthesize(
        self, text: str, language: str = "en", use_cache: bool = True
    ) -> bytes:
        """
        Synthesize speech from text.

        M4 Optimization: Caches audio output to avoid redundant synthesis.

        Args:
            text: Text to convert to speech
            language: Language code (hi, ta, te, en, etc.) or full name (Hindi, Tamil, etc.)
            use_cache: Whether to use audio cache (default: True)

        Returns:
            Audio bytes in WAV format (16kHz)
        """
        lang = self._normalize_language(language)

        # M4 Optimization: Check cache first
        if use_cache:
            cached = _audio_cache.get(text, lang)
            if cached is not None:
                logger.debug(f"TTS cache hit for {lang}")
                return cached

        # Synthesize audio
        audio_bytes = self._synthesize_uncached(text, lang)

        # Cache the result
        if use_cache:
            _audio_cache.put(text, lang, audio_bytes)

        return audio_bytes

    def _batch_check_cache(
        self,
        texts: list[tuple[str, str]],
        results: list[bytes | None],
        use_cache: bool,
    ) -> list[tuple[int, str, str]]:
        """Check cache for batch items, return uncached (index, text, lang) tuples."""
        uncached: list[tuple[int, str, str]] = []
        for i, (text, language) in enumerate(texts):
            lang = self._normalize_language(language)
            if use_cache:
                cached = _audio_cache.get(text, lang)
                if cached is not None:
                    results[i] = cached
                    continue
            uncached.append((i, text, lang))
        return uncached

    def _batch_synthesize_language_group(
        self,
        items: list[tuple[int, str]],
        lang: str,
        results: list[bytes | None],
        use_cache: bool,
        enc_pool,
    ) -> None:
        """Synthesize all items for one language, overlapping GPU and WAV encoding."""
        self.load_model(lang)
        pending_future = None
        pending_idx = -1
        pending_text = ""
        pending_lang = ""

        for idx, text in items:
            waveform = self._infer_waveform(text, lang)

            if pending_future is not None:
                audio = pending_future.result()
                results[pending_idx] = audio
                if use_cache:
                    _audio_cache.put(pending_text, pending_lang, audio)

            pending_future = enc_pool.submit(self._encode_wav, waveform)
            pending_idx = idx
            pending_text = text
            pending_lang = lang

        if pending_future is not None:
            audio = pending_future.result()
            results[pending_idx] = audio
            if use_cache:
                _audio_cache.put(pending_text, pending_lang, audio)

    def synthesize_batch(
        self, texts: list[tuple[str, str]], use_cache: bool = True
    ) -> list[bytes]:
        """
        Batch synthesize speech for multiple texts.

        M4 Optimization: Groups by language for model reuse, uses cache.

        Args:
            texts: List of (text, language) tuples
            use_cache: Whether to use audio cache (default: True)

        Returns:
            List of audio bytes in same order as input
        """
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor

        results: list[bytes | None] = [None] * len(texts)
        uncached_items = self._batch_check_cache(texts, results, use_cache)

        # Group uncached by language for model reuse
        by_lang: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for idx, text, lang in uncached_items:
            by_lang[lang].append((idx, text))

        # Use shared I/O pool for WAV encoding (avoids creating/destroying per batch)
        from ...core.optimized.thread_pool_manager import get_io_executor
        enc_pool = get_io_executor()
        for lang, items in by_lang.items():
            self._batch_synthesize_language_group(
                items, lang, results, use_cache, enc_pool
            )

        return [r or b"" for r in results]

    def warmup(self, languages: list[str] | None = None) -> None:
        """
        Pre-load models for common languages to avoid first-request latency.

        M4 Optimization: Run at startup for frequently used languages.

        Args:
            languages: List of language codes to warm up. Default: hi, ta, te, en
        """
        languages = languages or ["hi", "ta", "te", "en"]

        logger.info(f"Warming up TTS for languages: {languages}")
        for lang_code in languages:
            lang = self._normalize_language(lang_code)
            try:
                self.load_model(lang)
                # Run a single warmup inference to JIT compile
                _ = self._synthesize_uncached("Hello", lang)
                logger.info(f"TTS warmup complete for {lang}")
            except Exception as e:
                logger.warning(f"TTS warmup failed for {lang}: {e}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get audio cache statistics."""
        return _audio_cache.stats()

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return self.supported_languages.copy()

    def get_indian_languages(self) -> list[str]:
        """Get list of supported Indian language codes."""
        return ["hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa", "or", "as", "ur"]

    @property
    def is_loaded(self) -> bool:
        """Check if any TTS model is loaded."""
        return len(self.models) > 0

    def unload(self) -> None:
        """Unload all TTS models and free memory (for coordinated shutdown)."""
        if not self.models:
            return

        logger.info(f"Unloading MMS-TTS models for {len(self.models)} languages...")

        # Release memory via coordinator
        try:
            from ...core.optimized.memory_coordinator import get_memory_coordinator

            coordinator = get_memory_coordinator()
            coordinator.release("tts")
        except ImportError:
            pass

        # Clear models and tokenizers
        import gc

        # Copy keys to list since we're modifying the dict during iteration
        model_langs = list(self.models.keys())
        tokenizer_langs = list(self.tokenizers.keys())

        for lang in model_langs:
            del self.models[lang]
        for lang in tokenizer_langs:
            del self.tokenizers[lang]

        self.models.clear()
        self.tokenizers.clear()

        # Free MPS cache
        try:
            import torch

            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        gc.collect()
        logger.info("MMS-TTS models unloaded")


# Thread-safe singleton with memory coordinator integration
_tts_service_lock = threading.Lock()
_tts_service_instance: MMSTTSService | None = None


def get_mms_tts_service() -> MMSTTSService:
    """Get or create MMS-TTS service singleton (thread-safe with memory coordination)."""
    global _tts_service_instance

    if _tts_service_instance is not None:
        return _tts_service_instance

    with _tts_service_lock:
        if _tts_service_instance is not None:
            return _tts_service_instance

        # Create new instance with memory coordinator
        try:
            from ...core.optimized.memory_coordinator import get_memory_coordinator

            coordinator = get_memory_coordinator()
            acquired = coordinator.try_acquire_sync("tts")

            if not acquired:
                logger.warning(
                    "Could not acquire memory for TTS model - memory pressure may occur"
                )
        except ImportError:
            pass  # Memory coordinator not available

        _tts_service_instance = MMSTTSService()
        return _tts_service_instance


def unload_mms_tts_service() -> None:
    """Unload the TTS service singleton."""
    global _tts_service_instance

    with _tts_service_lock:
        if _tts_service_instance is not None:
            _tts_service_instance.unload()
            _tts_service_instance = None


def is_mms_available() -> bool:
    """Check if MMS-TTS is available."""
    return MMS_TTS_AVAILABLE


__all__ = [
    "MMS_TTS_AVAILABLE",
    "MMSTTSService",
    "get_mms_tts_service",
    "is_mms_available",
    "unload_mms_tts_service",
]
