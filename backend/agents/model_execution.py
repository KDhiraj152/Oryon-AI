"""
Model Execution Agent
=====================

Manages the full lifecycle of ML model inference:
- Routes inference requests to optimal hardware backends
- Manages model loading/unloading via MemoryCoordinator
- Tracks per-model latency, throughput, error rates
- Interfaces with GPU pipeline scheduler for batched execution
- Reports metrics to ResourceMonitor and Evaluation agents

Listens to: OrchestratorAgent (inference requests), HardwareOptimizer (config changes)
Emits to: ResourceMonitor (metrics), Evaluation (quality scores), Orchestrator (results)
"""

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .base import AgentMessage, BaseAgent, MessageType

logger = logging.getLogger(__name__)


class InferenceTask(str, Enum):
    """Supported inference task types."""
    GENERATE = "generate"
    GENERATE_STREAM = "generate_stream"
    EMBED = "embed"
    RERANK = "rerank"
    TRANSLATE = "translate"
    TTS = "tts"
    STT = "stt"
    OCR = "ocr"


@dataclass
class ModelStats:
    """Per-model performance statistics."""
    model_name: str
    total_calls: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    last_used: float = 0.0
    last_latency_ms: float = 0.0
    tokens_generated: int = 0
    # Rolling window for P50/P95
    recent_latencies: list[float] = field(default_factory=list)
    _max_window: int = 200

    def record(self, latency_ms: float, tokens: int = 0) -> None:
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        self.last_used = time.time()
        self.last_latency_ms = latency_ms
        self.tokens_generated += tokens
        self.recent_latencies.append(latency_ms)
        if len(self.recent_latencies) > self._max_window:
            self.recent_latencies = self.recent_latencies[-self._max_window:]

    def record_error(self) -> None:
        self.errors += 1

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.total_calls)

    @property
    def p50_ms(self) -> float:
        if not self.recent_latencies:
            return 0.0
        s = sorted(self.recent_latencies)
        return s[len(s) // 2]

    @property
    def p95_ms(self) -> float:
        if not self.recent_latencies:
            return 0.0
        s = sorted(self.recent_latencies)
        return s[int(len(s) * 0.95)]

    @property
    def error_rate(self) -> float:
        return self.errors / max(1, self.total_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "calls": self.total_calls,
            "avg_ms": round(self.avg_latency_ms, 1),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "errors": self.errors,
            "error_rate": round(self.error_rate, 4),
            "tokens": self.tokens_generated,
        }


# Map InferenceTask → model names used in MemoryCoordinator
TASK_TO_MODEL: dict[InferenceTask, str] = {
    InferenceTask.GENERATE: "llm",
    InferenceTask.GENERATE_STREAM: "llm",
    InferenceTask.EMBED: "embedder",
    InferenceTask.RERANK: "reranker",
    InferenceTask.TRANSLATE: "translation",
    InferenceTask.TTS: "tts",
    InferenceTask.STT: "stt",
    InferenceTask.OCR: "ocr",
}


class ModelExecutionAgent(BaseAgent):
    """
    Manages ML model inference with hardware-aware routing, lifecycle management,
    and performance tracking.

    Execution flow per request:
    1. Receive request (task type + payload)
    2. Route to optimal backend via DeviceRouter
    3. Ensure model is loaded via MemoryCoordinator
    4. Submit to GPU pipeline or execute directly
    5. Track latency/throughput/errors
    6. Emit metrics to ResourceMonitor
    7. Return result to caller
    """

    def __init__(self):
        super().__init__(name="model_execution")
        self._model_stats: dict[str, ModelStats] = {}
        self._active_requests: int = 0
        self._max_concurrent: int = 4
        self._request_semaphore = asyncio.Semaphore(4)
        self._request_log: OrderedDict[str, dict] = OrderedDict()
        self._max_log_size: int = 1000

    async def initialize(self) -> None:
        """Pre-warm connections to model infrastructure."""
        try:
            from backend.core.optimized.device_router import get_device_router
            self._device_router = get_device_router()
            logger.info(f"ModelExecution: device_router available, backends={self._device_router.get_optimal_backends()}")
        except Exception as e:
            self._device_router = None  # type: ignore[assignment]
            logger.warning(f"ModelExecution: device_router unavailable: {e}")

        try:
            from backend.core.optimized.memory_coordinator import get_memory_coordinator
            self._memory_coordinator = get_memory_coordinator()
        except Exception:
            self._memory_coordinator = None  # type: ignore[assignment]

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process inference requests and management commands."""
        if message.msg_type == MessageType.REQUEST:
            action = message.payload.get("action")

            if action == "infer":
                return await self._handle_inference(message)
            elif action == "get_stats":
                return self._handle_get_stats(message)
            elif action == "get_model_status":
                return self._handle_model_status(message)
            elif action == "preload":
                return await self._handle_preload(message)

        elif message.msg_type == MessageType.EVENT:
            action = message.payload.get("action")
            if action == "config_changed":
                return self._handle_config_change(message)

        return None

    async def _handle_inference(self, message: AgentMessage) -> AgentMessage:
        """Execute an inference request with full lifecycle management."""
        task_str = message.payload.get("task", "generate")
        payload = message.payload.get("payload", {})

        try:
            task = InferenceTask(task_str)
        except ValueError:
            return message.reply({"error": f"Unknown task: {task_str}"})

        model_name = TASK_TO_MODEL.get(task, task_str)
        stats = self._get_or_create_stats(model_name)

        async with self._request_semaphore:
            self._active_requests += 1
            t0 = time.perf_counter()
            try:
                result = await self._execute_task(task, payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                tokens = result.get("tokens_generated", 0) if isinstance(result, dict) else 0
                stats.record(elapsed_ms, tokens)

                # Emit metric to resource_monitor
                await self._emit_metric(model_name, elapsed_ms, tokens)

                # Touch model in coordinator to prevent eviction
                if self._memory_coordinator:
                    self._memory_coordinator.touch_model(model_name)

                self._log_request(message.correlation_id, task_str, elapsed_ms, True)

                return message.reply({
                    "result": result,
                    "latency_ms": round(elapsed_ms, 1),
                    "model": model_name,
                })

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                stats.record_error()
                self._log_request(message.correlation_id, task_str, elapsed_ms, False)
                logger.error(f"Inference failed for {task_str}: {e}", exc_info=True)
                return message.reply({"error": str(e), "task": task_str})
            finally:
                self._active_requests -= 1

    async def _execute_task(self, task: InferenceTask, payload: dict) -> Any:
        """Dispatch task to the appropriate model/engine."""
        if task == InferenceTask.GENERATE:
            return await self._exec_generate(payload)
        elif task == InferenceTask.GENERATE_STREAM:
            # Streaming not directly supported via agent messages;
            # return a collected result
            return await self._exec_generate(payload)
        elif task == InferenceTask.EMBED:
            return await self._exec_embed(payload)
        elif task == InferenceTask.RERANK:
            return await self._exec_rerank(payload)
        elif task == InferenceTask.TRANSLATE:
            return await self._exec_translate(payload)
        elif task == InferenceTask.TTS:
            return await self._exec_tts(payload)
        elif task == InferenceTask.STT:
            return await self._exec_stt(payload)
        elif task == InferenceTask.OCR:
            return await self._exec_ocr(payload)
        else:
            raise ValueError(f"Unhandled task: {task}")

    async def _exec_generate(self, payload: dict) -> dict:
        """Execute LLM generation via UnifiedInferenceEngine."""
        from backend.services.inference.unified_engine import get_inference_engine

        engine = get_inference_engine()
        prompt = payload.get("prompt", "")
        max_tokens = payload.get("max_tokens", 512)
        temperature = payload.get("temperature", 0.7)
        system_prompt = payload.get("system_prompt")

        result = await engine.generate_async(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        # Rough token estimate: 1 token ≈ 4 chars
        tokens_est = len(result) // 4 if isinstance(result, str) else 0
        return {"text": result, "tokens_generated": tokens_est}

    async def _exec_embed(self, payload: dict) -> dict:
        """Execute embedding via UnifiedInferenceEngine."""
        from backend.services.inference.unified_engine import get_inference_engine

        engine = get_inference_engine()
        texts = payload.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]

        embeddings = await engine.embed(texts)
        return {"shape": list(embeddings.shape), "dtype": str(embeddings.dtype)}

    async def _exec_rerank(self, payload: dict) -> dict:
        """Execute reranking via HighPerformanceModelManager."""
        loop = asyncio.get_running_loop()
        from backend.core.optimized.model_manager import get_model_manager

        manager = get_model_manager()
        query = payload.get("query", "")
        documents = payload.get("documents", [])
        top_k = payload.get("top_k")

        results = await loop.run_in_executor(
            None, lambda: manager.rerank(query, documents, top_k=top_k)
        )
        return {"rankings": results}

    async def _exec_translate(self, payload: dict) -> dict:
        """Execute translation via translate model."""
        loop = asyncio.get_running_loop()
        text = payload.get("text", "")
        source_lang = payload.get("source_lang", "en")
        target_lang = payload.get("target_lang", "hi")

        try:
            from backend.services.translate.engine import TranslationEngine
            engine = TranslationEngine()
            result = await loop.run_in_executor(
                None, lambda: engine.translate(text, source_lang, target_lang)
            )
            return {"translated_text": result}
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}")

    async def _exec_tts(self, payload: dict) -> dict:
        """Execute text-to-speech."""
        loop = asyncio.get_running_loop()
        from backend.core.optimized.model_manager import get_model_manager

        manager = get_model_manager()
        text = payload.get("text", "")

        audio, sr = await loop.run_in_executor(
            None, lambda: manager.synthesize_speech(text)
        )
        return {"sample_rate": sr, "duration_s": round(len(audio) / sr, 2)}

    async def _exec_stt(self, payload: dict) -> dict:
        """Execute speech-to-text."""
        loop = asyncio.get_running_loop()
        from backend.core.optimized.model_manager import get_model_manager

        manager = get_model_manager()
        audio = payload.get("audio")
        if audio is None:
            return {"text": "", "error": "No audio data provided"}

        text = await loop.run_in_executor(None, lambda: manager.transcribe(audio))
        return {"text": text}

    async def _exec_ocr(self, payload: dict) -> dict:
        """Execute OCR."""
        loop = asyncio.get_running_loop()
        image_path = payload.get("image_path", "")

        try:
            from backend.services.ocr import get_ocr_service
            ocr_service = get_ocr_service()
            result = await loop.run_in_executor(
                None, lambda: ocr_service.process_image(image_path)  # type: ignore[attr-defined]
            )
            return {"text": result}
        except Exception as e:
            raise RuntimeError(f"OCR failed: {e}")

    async def _emit_metric(self, model_name: str, latency_ms: float, tokens: int) -> None:
        """Send metrics to resource_monitor agent."""
        from .base import AgentRegistry
        registry = AgentRegistry()
        await registry.route_message(AgentMessage(
            msg_type=MessageType.METRIC,
            sender=self.name,
            recipient="resource_monitor",
            payload={
                "action": "record_inference",
                "model": model_name,
                "latency_ms": latency_ms,
                "tokens": tokens,
            },
        ))

    def _handle_get_stats(self, message: AgentMessage) -> AgentMessage:
        """Return aggregated model statistics."""
        return message.reply({
            "models": {k: v.to_dict() for k, v in self._model_stats.items()},
            "active_requests": self._active_requests,
        })

    def _handle_model_status(self, message: AgentMessage) -> AgentMessage:
        """Return current model load status."""
        status = {}
        if self._memory_coordinator:
            status = self._memory_coordinator.get_status()
        return message.reply({"coordinator_status": status})

    async def _handle_preload(self, message: AgentMessage) -> AgentMessage:
        """Preload specified models."""
        models = message.payload.get("models", [])
        results = {}
        for model_name in models:
            try:
                if self._memory_coordinator:
                    ok = await self._memory_coordinator.acquire(model_name)
                    results[model_name] = "acquired" if ok else "failed"
                else:
                    results[model_name] = "no_coordinator"
            except Exception as e:
                results[model_name] = f"error: {e}"
        return message.reply({"preload_results": results})

    def _handle_config_change(self, message: AgentMessage) -> AgentMessage | None:
        """Handle configuration changes from HardwareOptimizer."""
        changes = message.payload.get("changes", {})
        if "max_concurrent" in changes:
            new_val = changes["max_concurrent"]
            self._max_concurrent = new_val
            self._request_semaphore = asyncio.Semaphore(new_val)
            logger.info(f"ModelExecution: max_concurrent updated to {new_val}")
        return None

    def _get_or_create_stats(self, model_name: str) -> ModelStats:
        if model_name not in self._model_stats:
            self._model_stats[model_name] = ModelStats(model_name=model_name)
        return self._model_stats[model_name]

    def _log_request(self, correlation_id: str, task: str, latency_ms: float, success: bool) -> None:
        self._request_log[correlation_id] = {
            "task": task,
            "latency_ms": round(latency_ms, 1),
            "success": success,
            "timestamp": time.time(),
        }
        while len(self._request_log) > self._max_log_size:
            self._request_log.popitem(last=False)
