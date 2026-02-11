"""Async HuggingFace model client wrappers with httpx for non-blocking inference."""

import asyncio
import logging
import os
import time
import weakref
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import httpx

from ...core.optimized import get_device_router

logger = logging.getLogger(__name__)

CONTENT_TYPE_JSON = "application/json"

# Track all active clients for cleanup
_active_clients: weakref.WeakSet = weakref.WeakSet()


async def cleanup_all_clients():
    """Cleanup all active HTTP clients. Call during shutdown."""
    for client in _active_clients:
        try:
            await client.close()
        except Exception as e:
            logger.warning(f"Error closing client: {e}")


class RateLimiter:
    """Async rate limiter for API calls."""

    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        async with self._lock:
            now = time.time()
            # Remove calls outside the time window
            self.calls = [
                call_time
                for call_time in self.calls
                if now - call_time < self.time_window
            ]

            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    self.calls = []

            self.calls.append(now)


class BaseAsyncModelClient(ABC):
    """Base class for async HuggingFace model clients."""

    def __init__(self, model_id: str, api_key: str | None = None):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60)
        self._closed = False

        # Async HTTP client with retry
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            transport=httpx.AsyncHTTPTransport(retries=3),
        )

        # Device router for local inference fallback
        self.device_router = get_device_router()

        # Track this client for cleanup
        _active_clients.add(self)

    @property
    def is_closed(self) -> bool:
        """Check if client is closed."""
        return self._closed

    async def close(self):
        """Close HTTP client."""
        if not self._closed:
            self._closed = True
            try:
                await self.client.aclose()
                logger.debug(f"Closed HTTP client for {self.model_id}")
            except Exception as e:
                logger.warning(f"Error closing HTTP client for {self.model_id}: {e}")

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        if not self._closed and self.client is not None:
            # Can't await in __del__, schedule cleanup
            try:
                loop = asyncio.get_running_loop()
                # Store reference to prevent dangling task warning
                _cleanup_task = loop.create_task(self.close())
                # Note: Task will complete async, we just need reference for GC
                del _cleanup_task  # Explicit delete to avoid unused warning
            except RuntimeError:
                # No running event loop - try sync close
                try:
                    # Force close without await (httpx internal)
                    self.client._transport.close()
                except Exception:
                    pass  # Best effort

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Content-Type": CONTENT_TYPE_JSON}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _make_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make async API request with rate limiting and error handling."""
        await self.rate_limiter.wait_if_needed()

        try:
            response = await self.client.post(
                self.api_url, headers=self._get_headers(), json=payload
            )

            if response.status_code == 503:
                # Model is loading, wait and retry
                await asyncio.sleep(20)
                response = await self.client.post(
                    self.api_url, headers=self._get_headers(), json=payload
                )

            # Check for deprecated/moved models
            if response.status_code == 410:
                raise RuntimeError(
                    f"Model {self.model_id} is deprecated or moved. Using fallback."
                )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"HTTP request failed: {e}")
            raise RuntimeError(f"API request failed: {e}")

    @abstractmethod
    async def process(self, *args, **kwargs):
        """Process input through the model."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class QwenAsyncClient(BaseAsyncModelClient):
    """Async client for Qwen3-8B text simplification."""

    def __init__(self, model_id: str | None = None, api_key: str | None = None):
        # Use Qwen as default simplification model
        from ...core.config import settings

        model_id = model_id or getattr(
            settings, "SIMPLIFICATION_MODEL_ID", "Qwen/Qwen3-8B"
        )
        super().__init__(model_id, api_key)

    async def process(self, text: str, max_length: int = 512, **kwargs) -> str:
        """Simplify text using the model."""
        payload = {
            "inputs": f"Simplify this text for a student: {text}",
            "parameters": {
                "max_length": max_length,
                "temperature": 0.7,
                "do_sample": True,
            },
        }

        try:
            result = await self._make_request(payload)
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", text)
            return text
        except Exception as e:
            logger.warning(f"Simplification API failed, returning original: {e}")
            return text


class IndicTrans2AsyncClient(BaseAsyncModelClient):
    """Async client for IndicTrans2 translation."""

    def __init__(
        self,
        model_id: str = "ai4bharat/indictrans2-en-indic-1B",
        api_key: str | None = None,
    ):
        super().__init__(model_id, api_key)

    async def process(self, text: str, target_lang: str = "Hindi", **kwargs) -> str:
        """Translate text to target language."""
        # Map language names to codes
        lang_map = {
            "Hindi": "hin_Deva",
            "Tamil": "tam_Taml",
            "Telugu": "tel_Telu",
            "Bengali": "ben_Beng",
            "Marathi": "mar_Deva",
            "Gujarati": "guj_Gujr",
            "Kannada": "kan_Knda",
        }

        target_code = lang_map.get(target_lang, "hin_Deva")

        payload = {
            "inputs": text,
            "parameters": {"src_lang": "eng_Latn", "tgt_lang": target_code},
        }

        try:
            result = await self._make_request(payload)
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("translation_text", text)
            elif isinstance(result, dict):
                return result.get("translation_text", text)
            return text
        except Exception as e:
            logger.warning(f"Translation API failed, returning original: {e}")
            return text


class BERTAsyncClient(BaseAsyncModelClient):
    """Async client for BERT-based validation and embeddings."""

    def __init__(self, model_id: str = "BAAI/bge-m3", api_key: str | None = None):
        super().__init__(model_id, api_key)

    async def process(self, text: str, **kwargs) -> dict[str, Any]:
        """Get embeddings or perform validation."""
        payload = {"inputs": text}

        try:
            result = await self._make_request(payload)
            return {"embeddings": result, "success": True}
        except Exception as e:
            logger.warning(f"BERT API failed: {e}")
            return {"embeddings": None, "success": False, "error": str(e)}


# Default MMS-TTS model ID
DEFAULT_MMS_TTS_MODEL = "facebook/mms-tts-hin"


class MMSTTSAsyncClient(BaseAsyncModelClient):
    """Async client for MMS-TTS text-to-speech (Facebook Massively Multilingual Speech).

    Supports 1100+ languages with excellent Indian language coverage.
    Uses VITS architecture for high-quality neural TTS.
    """

    def __init__(
        self, model_id: str = DEFAULT_MMS_TTS_MODEL, api_key: str | None = None
    ):
        super().__init__(model_id, api_key)

    async def process(self, text: str, language: str = "Hindi", **kwargs) -> bytes:
        """Generate speech from text using MMS-TTS."""
        # Facebook MMS-TTS models - ISO 639-3 language codes
        lang_models = {
            "Hindi": DEFAULT_MMS_TTS_MODEL,
            "hi": DEFAULT_MMS_TTS_MODEL,
            "Tamil": "facebook/mms-tts-tam",
            "ta": "facebook/mms-tts-tam",
            "Telugu": "facebook/mms-tts-tel",
            "te": "facebook/mms-tts-tel",
            "Bengali": "facebook/mms-tts-ben",
            "bn": "facebook/mms-tts-ben",
            "Marathi": "facebook/mms-tts-mar",
            "mr": "facebook/mms-tts-mar",
            "Gujarati": "facebook/mms-tts-guj",
            "gu": "facebook/mms-tts-guj",
            "Kannada": "facebook/mms-tts-kan",
            "kn": "facebook/mms-tts-kan",
            "Malayalam": "facebook/mms-tts-mal",
            "ml": "facebook/mms-tts-mal",
            "Punjabi": "facebook/mms-tts-pan",
            "pa": "facebook/mms-tts-pan",
            "Odia": "facebook/mms-tts-ory",
            "or": "facebook/mms-tts-ory",
            "English": "facebook/mms-tts-eng",
            "en": "facebook/mms-tts-eng",
        }

        model = lang_models.get(language, "facebook/mms-tts-hin")
        if model != self.model_id:
            self.model_id = model
            self.api_url = f"https://api-inference.huggingface.co/models/{model}"

        payload = {"inputs": text}

        try:
            await self.rate_limiter.wait_if_needed()
            response = await self.client.post(
                self.api_url, headers=self._get_headers(), json=payload
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.warning(f"TTS API failed: {e}")
            return b""


# Alias for backwards compatibility
IndicTTSAsyncClient = MMSTTSAsyncClient

__all__ = [
    "BERTAsyncClient",
    "BaseAsyncModelClient",
    "IndicTTSAsyncClient",  # Alias
    "IndicTrans2AsyncClient",
    "MMSTTSAsyncClient",
    "QwenAsyncClient",
    "RateLimiter",
    "cleanup_all_clients",
]
