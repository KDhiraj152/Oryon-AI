"""
Unified Inference Backend — Bridge to Existing Model Services
================================================================

Implements the InferenceBackend protocol by delegating to the
existing service layer (ai_core, rag, translate, speech, ocr).

This allows the new execution layer to work with existing models
while providing a clean migration path to fully isolated backends.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

from backend.infra.runtime.backends import (
    BackendCapabilities,
    BackendType,
    InferenceBackend,
    InferenceRequest,
    InferenceResult,
)
from backend.infra.telemetry import get_logger, get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)

class UnifiedBridgeBackend(InferenceBackend):
    """
    Bridge backend that delegates to existing service layer.

    Wraps the existing services (ai_core, rag, translate, tts, stt, ocr)
    behind the InferenceBackend protocol, enabling the new execution
    layer to use them without refactoring services immediately.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._loaded_models: set[str] = set()
        self._executor = None
        # Cached service instances (avoid per-request instantiation)
        self._translation_engine: Any = None
        self._speech_generator: Any = None
        self._speech_processor: Any = None
        self._ocr_service: Any = None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_type=BackendType.TORCH_MPS if self._device == "mps" else BackendType.TORCH_CPU,
            supports_streaming=True,
            supports_batching=True,
            supports_quantization=True,
            supports_fp16=self._device != "cpu",
            max_batch_size=64,
            max_sequence_length=32768,
            supported_model_types={"llm", "embedding", "reranker", "translation", "tts", "stt", "ocr"},
            device=self._device,
        )

    async def load_model(self, model_name: str, **kwargs: Any) -> None:
        """Models are loaded lazily by existing services."""
        self._loaded_models.add(model_name)
        logger.info("model_registered", model=model_name, device=self._device)

    async def unload_model(self, model_name: str) -> None:
        self._loaded_models.discard(model_name)

    async def is_model_loaded(self, model_name: str) -> bool:
        return model_name in self._loaded_models

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Route inference to the appropriate service."""
        start = time.monotonic()

        with tracer.span(
            f"backend.infer.{request.task}",
            attributes={"task": request.task, "device": self._device},
        ):
            try:
                output = await self._dispatch(request)
                latency = (time.monotonic() - start) * 1000

                return InferenceResult(
                    output=output,
                    latency_ms=latency,
                    model=request.inputs.get("model", request.task),
                    device=self._device,
                    input_tokens=request.inputs.get("input_tokens", 0),
                    output_tokens=request.inputs.get("output_tokens", 0),
                )

            except (RuntimeError, ValueError, OSError) as exc:
                logger.error("inference_failed", exc=exc, task=request.task)
                raise

    async def infer_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """Stream tokens from LLM generation."""
        if request.task != "generate":
            result = await self.infer(request)
            yield str(result.output)
            return

        try:
            from backend.ml.inference.unified_engine import get_inference_engine
            eng: Any = get_inference_engine()

            if hasattr(eng, "generate_stream"):
                async for token in eng.generate_stream(
                    prompt=request.inputs.get("prompt", ""),
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                ):
                    yield token
            elif hasattr(eng, "generate_async"):
                result = await eng.generate_async(
                    prompt=request.inputs.get("prompt", ""),
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                yield str(result)
            else:
                result = await asyncio.to_thread(
                    lambda: eng.generate(
                        prompt=request.inputs.get("prompt", ""),
                        max_tokens=request.max_tokens,
                    )
                )
                yield str(result)

        except ImportError:
            logger.warning("streaming_not_available", task=request.task)
            result = await self.infer(request)
            yield str(result.output)

    async def _dispatch(self, request: InferenceRequest) -> Any:
        """Dispatch to the correct service based on task type."""
        task = request.task
        inputs = request.inputs

        if task in ("generate", "chat", "reasoning", "code"):
            return await self._generate(inputs, request)
        elif task == "embed":
            return await self._embed(inputs)
        elif task == "rerank":
            return await self._rerank(inputs)
        elif task == "translate":
            return await self._translate(inputs)
        elif task == "tts":
            return await self._tts(inputs)
        elif task == "stt":
            return await self._stt(inputs)
        elif task == "ocr":
            return await self._ocr(inputs)
        else:
            raise ValueError(f"Unknown task: {task}")

    async def _generate(self, inputs: dict, request: InferenceRequest) -> Any:
        try:
            from backend.ml.inference.unified_engine import get_inference_engine
            eng: Any = get_inference_engine()
            if hasattr(eng, "generate_async"):
                return await eng.generate_async(
                    prompt=inputs.get("prompt", ""),
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    system_prompt=inputs.get("system_prompt"),
                )
            return await asyncio.to_thread(
                lambda: eng.generate(
                    prompt=inputs.get("prompt", ""),
                    max_tokens=request.max_tokens,
                )
            )
        except ImportError:
            from backend.services.chat.engine import get_ai_engine
            ai_eng: Any = get_ai_engine()
            return await ai_eng.generate(
                prompt=inputs.get("prompt", ""),
                context=inputs.get("context", []),
            )

    async def _embed(self, inputs: dict) -> Any:
        try:
            from backend.services.chat.rag import get_rag_service
            rag = get_rag_service()
            text = inputs.get("text", "")
            if isinstance(text, list):
                return await asyncio.to_thread(rag.embed_batch, text)  # type: ignore[attr-defined]
            return await asyncio.to_thread(rag.embed_text, text)  # type: ignore[attr-defined]
        except ImportError:
            raise RuntimeError("RAG service not available")

    async def _rerank(self, inputs: dict) -> Any:
        try:
            from backend.core.optimized.model_manager import get_model_manager
            mm = get_model_manager()
            return await asyncio.to_thread(
                mm.rerank,
                query=inputs.get("query", ""),
                documents=inputs.get("documents", []),
            )
        except ImportError:
            raise RuntimeError("Reranking service not available")

    async def _translate(self, inputs: dict) -> Any:
        try:
            if self._translation_engine is None:
                from backend.ml.translate.engine import TranslationEngine
                self._translation_engine = TranslationEngine()
            return await asyncio.to_thread(
                self._translation_engine.translate,
                inputs.get("text", ""),
                inputs.get("source_lang", "en"),
                inputs.get("target_lang", "hi"),
            )
        except ImportError:
            raise RuntimeError("Translation service not available")

    async def _tts(self, inputs: dict) -> Any:
        try:
            if self._speech_generator is None:
                from backend.ml.speech.speech_generator import SpeechGenerator
                self._speech_generator = SpeechGenerator()
            return await asyncio.to_thread(
                self._speech_generator.generate,
                inputs.get("text", ""),
            )
        except ImportError:
            raise RuntimeError("TTS service not available")

    async def _stt(self, inputs: dict) -> Any:
        try:
            if self._speech_processor is None:
                from backend.ml.speech.speech_processor import (
                    SpeechProcessor,  # type: ignore[attr-defined]
                )
                self._speech_processor = SpeechProcessor()
            return await asyncio.to_thread(
                self._speech_processor.transcribe,
                inputs.get("audio_path", ""),
            )
        except ImportError:
            raise RuntimeError("STT service not available")

    async def _ocr(self, inputs: dict) -> Any:
        try:
            if self._ocr_service is None:
                from backend.ml.ocr.ocr import OCRService
                self._ocr_service = OCRService()
            return await asyncio.to_thread(
                self._ocr_service.process_image,
                inputs.get("image_path", ""),
            )
        except ImportError:
            raise RuntimeError("OCR service not available")

    async def health_check(self) -> bool:
        return True

    async def shutdown(self) -> None:
        self._loaded_models.clear()
