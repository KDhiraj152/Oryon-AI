"""
Middleware Orchestrator — Unified Facade
=========================================

Wires together all five phases into a single entry point:
  Phase 1 → RequestClassifier  (intent + complexity + routing)
  Phase 2 → AgentPipeline      (stage execution with contracts)
  Phase 3 → MemoryManager      (short/long/retrieval memory)
  Phase 4 → LatencyController   (budgets, early-exit, parallel inference)
  Phase 5 → SelfEvaluator       (quality, routing ledger, heuristic adjustment)

Integration with existing agent system:
  - Uses AgentRegistry to dispatch to ModelExecutionAgent
  - Bridges to existing OrchestratorAgent for backward compatibility
  - Exposes a simple `process()` async API for FastAPI routes
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .classifier import (
    ClassifiedRequest,
    ComplexityLevel,
    ModelTarget,
    RequestClassifier,
    TaskIntent,
)
from .evaluator import SelfEvaluator
from .latency import ExitReason, LatencyBudget, LatencyController
from .memory import MemoryManager
from .pipeline import (
    AgentPipeline,
    PipelineResult,
    PipelineStage,
    StageInput,
    StageOutput,
    StageStatus,
)

logger = logging.getLogger(__name__)

# ── Safe Math Evaluator ──────────────────────────────────────────────────────

def _safe_eval_ast(node: Any) -> float | int:
    """
    Recursively evaluate an AST expression tree using only safe arithmetic ops.

    Replaces eval() entirely — no code execution, only numeric computation.
    Raises ValueError on any unsupported node type.
    """
    import ast
    import operator
    from collections.abc import Callable

    _BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPS: dict[type, Callable[[Any], Any]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        # Prevent excessively large exponents
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        result: float | int = op_fn(left, right)
        return result
    elif isinstance(node, ast.UnaryOp):
        u_fn = _UNARY_OPS.get(type(node.op))
        if u_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        result = u_fn(_safe_eval_ast(node.operand))
        return result
    else:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

# ── Response Contract ────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MiddlewareResponse:
    """Unified response from the middleware layer."""

    request_id: str
    success: bool
    result: dict[str, Any]
    intent: str
    model_used: str
    total_latency_ms: float
    stage_latencies: dict[str, float] = field(default_factory=dict)
    classification: ClassifiedRequest | None = None
    evaluation_quality: str = ""
    from_cache: bool = False
    early_exit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

# ── Middleware Orchestrator ──────────────────────────────────────────────────

class MiddlewareOrchestrator:
    """
    High-performance orchestration layer coordinating models, hardware, and memory.

    Usage:
        orchestrator = MiddlewareOrchestrator()
        await orchestrator.initialize(agent_registry)
        response = await orchestrator.process("Explain photosynthesis in Hindi")
    """

    def __init__(
        self,
        classifier: RequestClassifier | None = None,
        memory: MemoryManager | None = None,
        latency: LatencyController | None = None,
        evaluator: SelfEvaluator | None = None,
    ):
        self.classifier = classifier or RequestClassifier()
        self.memory = memory or MemoryManager()
        self.latency = latency or LatencyController()
        self.evaluator = evaluator or SelfEvaluator()

        self._agent_registry: Any | None = None
        self._model_execution_agent: Any | None = None
        self._initialized = False

        # Stats
        self._total_requests = 0
        self._total_errors = 0
        self._total_cache_hits = 0

    # ── Lifecycle ────────────────────────────────────────────────────

    async def initialize(self, agent_registry: Any | None = None) -> None:
        """
        Initialize with optional reference to the existing AgentRegistry.

        If no registry is provided, the orchestrator works in standalone mode
        (useful for testing).
        """
        self._agent_registry = agent_registry
        if agent_registry:
            self._model_execution_agent = agent_registry.get_agent("model_execution")
        self._initialized = True
        logger.info("MiddlewareOrchestrator initialized")

    # ── Main Entry Point ─────────────────────────────────────────────

    async def process(
        self,
        prompt: str,
        session_id: str | None = None,
        payload: dict[str, Any] | None = None,
        timeout_ms: float | None = None,
        priority: int = 2,
    ) -> MiddlewareResponse:
        """
        Process a user request through the full middleware pipeline.

        Flow:
          1. Classify (intent, complexity, model target)
          2. Allocate latency budget
          3. Retrieve memory context (short-term + RAG if needed)
          4. Build and execute pipeline
          5. Record in memory
          6. Evaluate and record telemetry
        """
        request_id = str(uuid.uuid4())[:12]
        t0 = time.perf_counter()
        self._total_requests += 1

        try:
            # ── Phase 1: Classify ──────────────────────────────────
            classification = self.classifier.classify(prompt, request_id)
            logger.debug(
                f"[{request_id}] Classified: intent={classification.intent.value}, "
                f"complexity={classification.complexity.name}, "
                f"target={classification.model_target.value}"
            )

            # ── Phase 4: Latency budget ───────────────────────────
            budget = self.latency.create_budget(total_ms=timeout_ms)

            # ── Phase 3: Memory retrieval ─────────────────────────
            context_text = ""
            rag_context = ""
            from_cache = False

            if session_id:
                context_text = await self.memory.get_context_text(
                    session_id, last_n=10
                )

            if classification.needs_rag:
                retrieval = await self.memory.retrieve(
                    query=prompt, top_k=5, session_id=session_id
                )
                if retrieval.entries:
                    rag_parts = [e.content for e in retrieval.entries if e.content]
                    rag_context = "\n\n".join(rag_parts[:3])
                    from_cache = retrieval.from_cache
                    if from_cache:
                        self._total_cache_hits += 1
                    self.latency.record_latency("rag_retrieve", retrieval.latency_ms)

            # ── Phase 2: Build and execute pipeline ────────────────
            merged_payload = {
                "prompt": prompt,
                "context": context_text,
                "rag_context": rag_context,
                "intent": classification.intent.value,
                "model_target": classification.model_target.value,
                "estimated_tokens": classification.estimated_tokens,
                "needs_translation": classification.needs_translation,
                "needs_tts": classification.needs_tts,
                "target_language": classification.target_language,
                "source_language": classification.source_language,
                **(payload or {}),
            }

            pipeline = self._build_pipeline(classification)
            pipeline_result = await pipeline.execute(
                request_id=request_id,
                payload=merged_payload,
                context={"budget": budget, "classification": classification},
            )

            # Collect stage latencies
            stage_latencies = {
                s.stage_name: s.latency_ms
                for s in pipeline_result.stages
                if s.latency_ms > 0
            }
            for name, lat in stage_latencies.items():
                self.latency.record_latency(name, lat)

            # ── Phase 3: Record in short-term memory ──────────────
            if session_id:
                await self.memory.add_message(session_id, "user", prompt)
                response_text = pipeline_result.final_result.get("response_text", "")
                if response_text:
                    await self.memory.add_message(
                        session_id, "assistant", response_text
                    )

                # Check if session needs summarization
                if await self.memory.should_summarize(session_id):
                    logger.info("[%s] Session %s needs summarization", request_id, session_id)
                    # Could trigger async summarization here

            # ── Phase 5: Evaluate ─────────────────────────────────
            total_latency_ms = (time.perf_counter() - t0) * 1000
            model_used = pipeline_result.final_result.get(
                "model_used", classification.model_target.value
            )
            output_tokens = pipeline_result.final_result.get("output_tokens", 0)
            confidence = pipeline_result.final_result.get("confidence", 0.0)

            report = self.evaluator.evaluate(
                request_id=request_id,
                intent=classification.intent.value,
                model_target=classification.model_target.value,
                model_used=model_used,
                complexity_score=classification.complexity_score,
                total_latency_ms=total_latency_ms,
                confidence=confidence,
                output_tokens=output_tokens,
                has_error=not pipeline_result.success,
                was_early_exit=pipeline_result.final_result.get("early_exit", False),
                used_cache=from_cache,
                used_rag=classification.needs_rag,
                stage_latencies=stage_latencies,
            )

            return MiddlewareResponse(
                request_id=request_id,
                success=pipeline_result.success,
                result=pipeline_result.final_result,
                intent=classification.intent.value,
                model_used=model_used,
                total_latency_ms=round(total_latency_ms, 2),
                stage_latencies=stage_latencies,
                classification=classification,
                evaluation_quality=report.quality_signal.value,
                from_cache=from_cache,
                early_exit=pipeline_result.final_result.get("early_exit", False),
            )

        except Exception as e:
            self._total_errors += 1
            total_latency_ms = (time.perf_counter() - t0) * 1000
            logger.error("[%s] Middleware error: %s", request_id, e, exc_info=True)
            return MiddlewareResponse(
                request_id=request_id,
                success=False,
                result={"error": str(e)},
                intent="unknown",
                model_used="none",
                total_latency_ms=round(total_latency_ms, 2),
            )

    # ── Pipeline Construction ────────────────────────────────────────

    def _build_pipeline(self, classification: ClassifiedRequest) -> AgentPipeline:
        """Build a task-specific pipeline based on classification."""

        stages: list[PipelineStage] = []

        intent = classification.intent

        # Skip-LLM tasks (calculation)
        if classification.model_target == ModelTarget.SKIP_LLM:
            stages.append(PipelineStage(
                name="direct_compute",
                handler=self._handle_direct_compute,
                required_keys=("prompt",),
                output_keys=("response_text",),
                timeout_s=2.0,
            ))
            return AgentPipeline(stages)

        # Specialized tasks (translation, embedding, TTS, STT, OCR)
        if intent == TaskIntent.TRANSLATION:
            stages.append(PipelineStage(
                name="translate",
                handler=self._handle_translate,
                required_keys=("prompt",),
                output_keys=("response_text", "translated_text"),
                timeout_s=10.0,
            ))
            return AgentPipeline(stages)

        if intent in {TaskIntent.EMBEDDING, TaskIntent.RERANKING}:
            stages.append(PipelineStage(
                name=intent.value,
                handler=self._handle_specialized,
                required_keys=("prompt",),
                output_keys=("response_text",),
                timeout_s=5.0,
            ))
            return AgentPipeline(stages)

        if intent == TaskIntent.AUDIO:
            stages.append(PipelineStage(
                name="tts",
                handler=self._handle_tts,
                required_keys=("prompt",),
                output_keys=("audio_data",),
                timeout_s=10.0,
            ))
            return AgentPipeline(stages)

        if intent == TaskIntent.STT:
            stages.append(PipelineStage(
                name="stt",
                handler=self._handle_stt,
                required_keys=("prompt",),
                output_keys=("transcribed_text",),
                timeout_s=15.0,
            ))
            return AgentPipeline(stages)

        if intent == TaskIntent.OCR:
            stages.append(PipelineStage(
                name="ocr",
                handler=self._handle_ocr,
                required_keys=("prompt",),
                output_keys=("extracted_text",),
                timeout_s=10.0,
            ))
            return AgentPipeline(stages)

        # LLM-based tasks (chat, question, code, creative, etc.)
        stages.append(PipelineStage(
            name="generate",
            handler=self._handle_generate,
            required_keys=("prompt",),
            output_keys=("response_text",),
            timeout_s=30.0,
        ))

        # Post-generation: optional translation
        if classification.needs_translation:
            stages.append(PipelineStage(
                name="post_translate",
                handler=self._handle_post_translate,
                depends_on=("generate",),
                optional=True,
                output_keys=("translated_text",),
                timeout_s=10.0,
            ))

        # Post-generation: optional TTS
        if classification.needs_tts:
            stages.append(PipelineStage(
                name="post_tts",
                handler=self._handle_post_tts,
                depends_on=("generate",),
                optional=True,
                output_keys=("audio_data",),
                timeout_s=10.0,
            ))

        return AgentPipeline(stages)

    # ── Stage Handlers ───────────────────────────────────────────────

    async def _handle_direct_compute(self, inp: StageInput) -> StageOutput:
        """Handle computation tasks without LLM."""
        prompt = inp.payload.get("prompt", "")
        try:
            # Safe math evaluation — only allows numeric literals and operators
            import ast
            import operator
            import re
            expr = re.sub(r"[^0-9\+\-\*\/\.\(\)\s\%\^]", "", prompt)
            expr = expr.replace("^", "**")
            if expr.strip():
                tree = ast.parse(expr, mode="eval")
                result = _safe_eval_ast(tree.body)
                return StageOutput(
                    stage_name="direct_compute",
                    status=StageStatus.COMPLETED,
                    result={"response_text": str(result), "confidence": 0.95},
                )
        except (ValueError, SyntaxError, TypeError, ArithmeticError):
            pass

        return StageOutput(
            stage_name="direct_compute",
            status=StageStatus.COMPLETED,
            result={
                "response_text": "I couldn't compute that. Could you rephrase?",
                "confidence": 0.3,
            },
        )

    async def _handle_generate(self, inp: StageInput) -> StageOutput:
        """Route LLM generation through the existing agent system."""
        prompt = inp.payload.get("prompt", "")
        context = inp.payload.get("context", "")
        rag_context = inp.payload.get("rag_context", "")
        estimated_tokens = inp.payload.get("estimated_tokens", 2048)

        # Build full prompt with context
        full_prompt = prompt
        if rag_context:
            full_prompt = f"Context:\n{rag_context}\n\nUser: {prompt}"
        elif context:
            full_prompt = f"Conversation:\n{context}\n\nUser: {prompt}"

        # Route through existing ModelExecutionAgent
        if self._model_execution_agent:
            from ..agents.base import AgentMessage, MessageType

            msg = AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="middleware",
                recipient="model_execution",
                payload={
                    "action": "infer",
                    "task": "generate",
                    "payload": {
                        "prompt": full_prompt,
                        "system_prompt": inp.payload.get(
                            "system_prompt",
                            "You are Oryon, an AI education assistant.",
                        ),
                        "max_tokens": estimated_tokens,
                        "temperature": inp.payload.get("temperature", 0.7),
                    },
                },
            )
            response = await self._model_execution_agent.handle_message(msg)
            if response and "error" not in response.payload:
                result_data = response.payload.get("result", {})
                return StageOutput(
                    stage_name="generate",
                    status=StageStatus.COMPLETED,
                    result={
                        "response_text": result_data.get("text", ""),
                        "model_used": result_data.get("model", "unknown"),
                        "output_tokens": result_data.get("tokens_generated", 0),
                        "confidence": result_data.get("confidence", 0.7),
                    },
                )
            error = response.payload.get("error", "Unknown error") if response else "No response"
            return StageOutput(
                stage_name="generate",
                status=StageStatus.FAILED,
                error=error,
            )

        return StageOutput(
            stage_name="generate",
            status=StageStatus.FAILED,
            error="No model execution agent available",
        )

    async def _handle_translate(self, inp: StageInput) -> StageOutput:
        """Handle standalone translation."""
        if self._model_execution_agent:
            from ..agents.base import AgentMessage, MessageType

            msg = AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="middleware",
                recipient="model_execution",
                payload={
                    "action": "infer",
                    "task": "translate",
                    "payload": {
                        "text": inp.payload.get("prompt", ""),
                        "source_lang": inp.payload.get("source_language", "en"),
                        "target_lang": inp.payload.get("target_language", "hi"),
                    },
                },
            )
            response = await self._model_execution_agent.handle_message(msg)
            if response and "error" not in response.payload:
                result_data = response.payload.get("result", {})
                return StageOutput(
                    stage_name="translate",
                    status=StageStatus.COMPLETED,
                    result={
                        "response_text": result_data.get("translated_text", ""),
                        "translated_text": result_data.get("translated_text", ""),
                    },
                )

        return StageOutput(
            stage_name="translate",
            status=StageStatus.FAILED,
            error="Translation failed",
        )

    async def _handle_post_translate(self, inp: StageInput) -> StageOutput:
        """Translate LLM output as a post-processing step."""
        text = inp.context.get("response_text", "")
        if not text:
            return StageOutput(
                stage_name="post_translate",
                status=StageStatus.SKIPPED,
            )

        if self._model_execution_agent:
            from ..agents.base import AgentMessage, MessageType

            msg = AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="middleware",
                recipient="model_execution",
                payload={
                    "action": "infer",
                    "task": "translate",
                    "payload": {
                        "text": text,
                        "source_lang": "en",
                        "target_lang": inp.payload.get("target_language", "hi"),
                    },
                },
            )
            response = await self._model_execution_agent.handle_message(msg)
            if response and "error" not in response.payload:
                result_data = response.payload.get("result", {})
                return StageOutput(
                    stage_name="post_translate",
                    status=StageStatus.COMPLETED,
                    result={
                        "translated_text": result_data.get("translated_text", ""),
                    },
                )

        return StageOutput(
            stage_name="post_translate",
            status=StageStatus.FAILED,
            error="Post-translation failed",
        )

    async def _handle_tts(self, inp: StageInput) -> StageOutput:
        """Handle text-to-speech generation."""
        if self._model_execution_agent:
            from ..agents.base import AgentMessage, MessageType

            msg = AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="middleware",
                recipient="model_execution",
                payload={
                    "action": "infer",
                    "task": "tts",
                    "payload": {
                        "text": inp.payload.get("prompt", ""),
                    },
                },
            )
            response = await self._model_execution_agent.handle_message(msg)
            if response and "error" not in response.payload:
                return StageOutput(
                    stage_name="tts",
                    status=StageStatus.COMPLETED,
                    result={"audio_data": response.payload.get("result", {})},
                )

        return StageOutput(
            stage_name="tts",
            status=StageStatus.FAILED,
            error="TTS failed",
        )

    async def _handle_post_tts(self, inp: StageInput) -> StageOutput:
        """Generate TTS from LLM output as post-processing."""
        text = inp.context.get("response_text", "")
        if not text:
            return StageOutput(
                stage_name="post_tts",
                status=StageStatus.SKIPPED,
            )

        if self._model_execution_agent:
            from ..agents.base import AgentMessage, MessageType

            msg = AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="middleware",
                recipient="model_execution",
                payload={
                    "action": "infer",
                    "task": "tts",
                    "payload": {"text": text[:500]},
                },
            )
            response = await self._model_execution_agent.handle_message(msg)
            if response and "error" not in response.payload:
                return StageOutput(
                    stage_name="post_tts",
                    status=StageStatus.COMPLETED,
                    result={"audio_data": response.payload.get("result", {})},
                )

        return StageOutput(
            stage_name="post_tts",
            status=StageStatus.FAILED,
            error="Post-TTS failed",
        )

    async def _handle_stt(self, inp: StageInput) -> StageOutput:
        """Handle speech-to-text."""
        if self._model_execution_agent:
            from ..agents.base import AgentMessage, MessageType

            msg = AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="middleware",
                recipient="model_execution",
                payload={
                    "action": "infer",
                    "task": "stt",
                    "payload": inp.payload,
                },
            )
            response = await self._model_execution_agent.handle_message(msg)
            if response and "error" not in response.payload:
                return StageOutput(
                    stage_name="stt",
                    status=StageStatus.COMPLETED,
                    result=response.payload.get("result", {}),
                )

        return StageOutput(
            stage_name="stt", status=StageStatus.FAILED, error="STT failed"
        )

    async def _handle_ocr(self, inp: StageInput) -> StageOutput:
        """Handle OCR extraction."""
        if self._model_execution_agent:
            from ..agents.base import AgentMessage, MessageType

            msg = AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="middleware",
                recipient="model_execution",
                payload={
                    "action": "infer",
                    "task": "ocr",
                    "payload": inp.payload,
                },
            )
            response = await self._model_execution_agent.handle_message(msg)
            if response and "error" not in response.payload:
                return StageOutput(
                    stage_name="ocr",
                    status=StageStatus.COMPLETED,
                    result=response.payload.get("result", {}),
                )

        return StageOutput(
            stage_name="ocr", status=StageStatus.FAILED, error="OCR failed"
        )

    async def _handle_specialized(self, inp: StageInput) -> StageOutput:
        """Handle embedding/reranking."""
        intent = inp.payload.get("intent", "embedding")
        task = "embed" if intent == "embedding" else "rerank"

        if self._model_execution_agent:
            from ..agents.base import AgentMessage, MessageType

            msg = AgentMessage(
                msg_type=MessageType.REQUEST,
                sender="middleware",
                recipient="model_execution",
                payload={
                    "action": "infer",
                    "task": task,
                    "payload": inp.payload,
                },
            )
            response = await self._model_execution_agent.handle_message(msg)
            if response and "error" not in response.payload:
                return StageOutput(
                    stage_name=intent,
                    status=StageStatus.COMPLETED,
                    result=response.payload.get("result", {}),
                )

        return StageOutput(
            stage_name=intent,
            status=StageStatus.FAILED,
            error=f"{task} failed",
        )

    # ── Observability ────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive middleware statistics."""
        memory_stats = await self.memory.get_stats()
        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "total_cache_hits": self._total_cache_hits,
            "error_rate": round(
                self._total_errors / max(1, self._total_requests), 4
            ),
            "classifier": self.classifier.get_stats(),
            "latency": self.latency.get_stats(),
            "memory": memory_stats,
            "evaluation": self.evaluator.get_stats(),
        }

# ── Singleton ────────────────────────────────────────────────────────────────

_instance: MiddlewareOrchestrator | None = None

def get_middleware() -> MiddlewareOrchestrator:
    """Get or create the global middleware orchestrator."""
    global _instance
    if _instance is None:
        _instance = MiddlewareOrchestrator()
    return _instance

async def initialize_middleware(agent_registry: Any | None = None) -> MiddlewareOrchestrator:
    """Initialize the global middleware orchestrator."""
    mw = get_middleware()
    await mw.initialize(agent_registry)
    return mw
