"""
Model Accessors Mixin
=====================

Provides lazy-loading accessors for all 8 models.
Uses dependency injection pattern to avoid circular imports.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelAccessorsMixin:
    """
    Mixin providing lazy-loaded accessors for all 8 AI models.

    Models:
    - Qwen2.5-3B: Main LLM for simplification/generation
    - IndicTrans2-1B: Translation between 11 Indic languages
    - BGE-M3: Multi-lingual embeddings
    - BGE-Reranker: Candidate reranking
    - Whisper-V3: Speech-to-text
    - MMS-TTS: Text-to-speech for Indic languages
    - GOT-OCR2: Document/image OCR
    - Gemma-2-2B: Validation and quality checking
    """

    def __init__(self) -> None:
        """Initialize model cache."""
        self._model_cache: dict = {}

    def _get_llm(self) -> Any | None:
        """Get Qwen2.5-3B LLM instance (lazy load)."""
        if "llm" not in self._model_cache:
            try:
                from backend.services.simplifier.engine import get_simplifier

                self._model_cache["llm"] = get_simplifier()
            except ImportError:
                logger.warning("Simplifier engine not available")
                self._model_cache["llm"] = None
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                self._model_cache["llm"] = None
        return self._model_cache["llm"]

    def _get_translator(self) -> Any | None:
        """Get IndicTrans2 translator instance (lazy load)."""
        if "translator" not in self._model_cache:
            try:
                from backend.services.translate.engine import get_translator

                self._model_cache["translator"] = get_translator()
            except ImportError:
                logger.warning("Translator engine not available")
                self._model_cache["translator"] = None
            except Exception as e:
                logger.error(f"Failed to load translator: {e}")
                self._model_cache["translator"] = None
        return self._model_cache["translator"]

    def _get_embedder(self) -> Any | None:
        """Get BGE-M3 embedder instance (lazy load)."""
        if "embedder" not in self._model_cache:
            try:
                from backend.services.embeddings.engine import get_embedder

                self._model_cache["embedder"] = get_embedder()
            except ImportError:
                logger.warning("Embedder engine not available")
                self._model_cache["embedder"] = None
            except Exception as e:
                logger.error(f"Failed to load embedder: {e}")
                self._model_cache["embedder"] = None
        return self._model_cache["embedder"]

    def _get_validator(self) -> Any | None:
        """Get Gemma-2-2B validator instance (lazy load)."""
        if "validator" not in self._model_cache:
            try:
                from backend.services.validation.engine import get_validator

                self._model_cache["validator"] = get_validator()
            except ImportError:
                logger.warning("Validator engine not available")
                self._model_cache["validator"] = None
            except Exception as e:
                logger.error(f"Failed to load validator: {e}")
                self._model_cache["validator"] = None
        return self._model_cache["validator"]

    def _get_tts(self) -> Any | None:
        """Get MMS-TTS text-to-speech instance (lazy load)."""
        if "tts" not in self._model_cache:
            try:
                from backend.services.audio.mms_tts_service import get_tts_service

                self._model_cache["tts"] = get_tts_service()
            except ImportError:
                logger.warning("TTS service not available")
                self._model_cache["tts"] = None
            except Exception as e:
                logger.error(f"Failed to load TTS: {e}")
                self._model_cache["tts"] = None
        return self._model_cache["tts"]

    def _get_stt(self) -> Any | None:
        """Get Whisper V3 speech-to-text instance (lazy load)."""
        if "stt" not in self._model_cache:
            try:
                from backend.services.audio.whisper_service import get_whisper_service

                self._model_cache["stt"] = get_whisper_service()
            except ImportError:
                logger.warning("STT service not available")
                self._model_cache["stt"] = None
            except Exception as e:
                logger.error(f"Failed to load STT: {e}")
                self._model_cache["stt"] = None
        return self._model_cache["stt"]

    def _get_reranker(self) -> Any | None:
        """Get BGE-Reranker instance (lazy load)."""
        if "reranker" not in self._model_cache:
            try:
                from backend.services.reranker.engine import get_reranker

                self._model_cache["reranker"] = get_reranker()
            except ImportError:
                logger.warning("Reranker engine not available")
                self._model_cache["reranker"] = None
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self._model_cache["reranker"] = None
        return self._model_cache["reranker"]

    def _get_ocr(self) -> Any | None:
        """Get GOT-OCR2 instance (lazy load)."""
        if "ocr" not in self._model_cache:
            try:
                from backend.services.ocr.got_ocr_service import get_ocr_service

                self._model_cache["ocr"] = get_ocr_service()
            except ImportError:
                logger.warning("OCR service not available")
                self._model_cache["ocr"] = None
            except Exception as e:
                logger.error(f"Failed to load OCR: {e}")
                self._model_cache["ocr"] = None
        return self._model_cache["ocr"]

    def clear_model_cache(self) -> None:
        """Clear all cached model instances."""
        self._model_cache.clear()
        logger.info("Model cache cleared")
