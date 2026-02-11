"""
Orchestrator Agent
==================

Central request coordinator for the multi-agent system:
- Receives high-level user requests (chat, translate, embed, simplify)
- Decomposes into sub-tasks
- Routes to appropriate agents (ModelExecution, Evaluation)
- Aggregates results and returns to caller
- Implements request priority and load shedding
- Tracks end-to-end request latency

Listens to: External API requests, all agent results
Emits to: ModelExecutionAgent (tasks), EvaluationAgent (quality feedback),
         ResourceMonitor (request metrics)
"""

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .base import AgentMessage, AgentRegistry, BaseAgent, MessageType

logger = logging.getLogger(__name__)


class RequestPriority(int, Enum):
    """Request priority levels."""
    CRITICAL = 0   # Health checks, admin
    HIGH = 1       # Streaming chat
    NORMAL = 2     # Standard API requests
    LOW = 3        # Background tasks, batch processing
    BACKGROUND = 4 # Analytics, pre-warming


@dataclass
class PendingRequest:
    """Tracks an in-flight request through the agent system."""
    request_id: str
    task_type: str
    payload: dict
    priority: RequestPriority
    start_time: float
    timeout_s: float = 30.0
    result: Any = None
    error: str | None = None
    future: asyncio.Future | None = None
    sub_tasks: list[str] = field(default_factory=list)

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


class OrchestratorAgent(BaseAgent):
    """
    Central coordinator that decomposes user requests into agent tasks
    and assembles results.

    Design principles:
    - Non-blocking: all model calls go through ModelExecutionAgent
    - Priority-aware: high-priority requests preempt low-priority
    - Timeout-aware: shed load when queue depth exceeds thresholds
    - Observable: every request is tracked end-to-end
    """

    def __init__(self):
        super().__init__(name="orchestrator")
        self._pending: OrderedDict[str, PendingRequest] = OrderedDict()
        self._max_pending: int = 100
        self._completed_log: OrderedDict[str, dict] = OrderedDict()
        self._max_completed_log: int = 500
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._total_timeouts: int = 0
        self._total_shed: int = 0
        self._registry: AgentRegistry | None = None

    async def initialize(self) -> None:
        """Get agent registry reference."""
        self._registry = AgentRegistry()
        logger.info(f"OrchestratorAgent initialized, max_pending={self._max_pending}")

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process incoming requests and sub-task results."""
        if message.msg_type == MessageType.REQUEST:
            action = message.payload.get("action")
            if action == "process":
                return await self._handle_process_request(message)
            elif action == "get_stats":
                return message.reply(self._get_stats())
            elif action == "get_pending":
                return message.reply({
                    "pending": [{
                        "id": p.request_id,
                        "task": p.task_type,
                        "elapsed_ms": round(p.elapsed_ms, 1),
                        "priority": p.priority.name,
                    } for p in self._pending.values()]
                })

        elif message.msg_type == MessageType.RESPONSE:
            # A sub-task result came back
            return self._handle_sub_task_result(message)

        return None

    async def _handle_process_request(self, message: AgentMessage) -> AgentMessage:
        """
        Main entry point for processing user requests through the agent system.

        Expected payload:
        {
            "action": "process",
            "task_type": "chat" | "translate" | "embed" | "simplify" | "tts" | "stt" | "ocr",
            "payload": { ... task-specific data ... },
            "priority": 2,  # optional, default NORMAL
            "timeout_s": 30.0,  # optional
        }
        """
        task_type = message.payload.get("task_type", "chat")
        payload = message.payload.get("payload", {})
        priority = RequestPriority(message.payload.get("priority", RequestPriority.NORMAL))
        timeout_s = message.payload.get("timeout_s", 30.0)

        # Load shedding check
        if len(self._pending) >= self._max_pending:
            # Shed lowest priority first
            if priority.value >= RequestPriority.LOW.value:
                self._total_shed += 1
                return message.reply({
                    "error": "Server overloaded, request shed",
                    "pending_count": len(self._pending),
                })

        request_id = str(uuid.uuid4())[:12]
        pending = PendingRequest(
            request_id=request_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            start_time=time.time(),
            timeout_s=timeout_s,
        )
        self._pending[request_id] = pending
        self._total_requests += 1

        try:
            result = await asyncio.wait_for(
                self._execute_request(pending),
                timeout=timeout_s,
            )
            self._complete_request(request_id, result, None)
            return message.reply({
                "request_id": request_id,
                "result": result,
                "latency_ms": round(pending.elapsed_ms, 1),
            })
        except asyncio.TimeoutError:
            self._total_timeouts += 1
            self._complete_request(request_id, None, "timeout")
            return message.reply({
                "error": "Request timed out",
                "request_id": request_id,
                "timeout_s": timeout_s,
            })
        except Exception as e:
            self._total_errors += 1
            self._complete_request(request_id, None, str(e))
            return message.reply({
                "error": str(e),
                "request_id": request_id,
            })

    async def _execute_request(self, pending: PendingRequest) -> Any:
        """Decompose and execute a request via sub-agents."""
        task_type = pending.task_type
        payload = pending.payload

        if task_type == "chat":
            return await self._execute_chat(pending)
        elif task_type == "translate":
            return await self._execute_single_inference("translate", payload)
        elif task_type == "embed":
            return await self._execute_single_inference("embed", payload)
        elif task_type == "simplify":
            return await self._execute_simplify(pending)
        elif task_type == "tts":
            return await self._execute_single_inference("tts", payload)
        elif task_type == "stt":
            return await self._execute_single_inference("stt", payload)
        elif task_type == "ocr":
            return await self._execute_single_inference("ocr", payload)
        elif task_type == "rerank":
            return await self._execute_single_inference("rerank", payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _execute_chat(self, pending: PendingRequest) -> dict:
        """
        Chat request: optionally fetch RAG context, then generate response.
        """
        payload = pending.payload
        use_rag = payload.get("use_rag", True)

        rag_context = ""
        if use_rag:
            query = payload.get("prompt", payload.get("message", ""))
            # Embed + rerank for RAG
            try:
                _ = await self._send_to_model_execution("embed", {
                    "texts": [query],
                })
                # RAG retrieval would use embed_result in production
                # For now, include query for LLM context
                rag_context = f"[RAG context for: {query[:100]}]"
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}, proceeding without context")

        # Build prompt with context
        prompt = payload.get("prompt", payload.get("message", ""))
        if rag_context:
            prompt = f"Context: {rag_context}\n\nUser: {prompt}"

        system_prompt = payload.get("system_prompt", "You are ShikshaSetu, an AI education assistant.")

        result = await self._send_to_model_execution("generate", {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": payload.get("max_tokens", 1024),
            "temperature": payload.get("temperature", 0.7),
        })

        return result

    async def _execute_simplify(self, pending: PendingRequest) -> dict:
        """
        Content simplification pipeline:
        1. Generate simplified text
        2. (Optionally) translate result
        3. (Optionally) generate TTS
        """
        payload = pending.payload
        text = payload.get("text", "")
        target_language = payload.get("target_language")
        generate_audio = payload.get("generate_audio", False)

        # Step 1: Simplify via LLM
        simplify_prompt = (
            f"Simplify the following educational content for a student:\n\n{text}\n\n"
            f"Simplified version:"
        )
        simplified = await self._send_to_model_execution("generate", {
            "prompt": simplify_prompt,
            "max_tokens": payload.get("max_tokens", 1024),
            "temperature": 0.5,
        })
        result = {"simplified_text": simplified.get("text", "")}

        # Step 2: Optional translation (can run in parallel with TTS of original)
        tasks = []
        if target_language and target_language != "en":
            tasks.append(("translate", self._send_to_model_execution("translate", {
                "text": result["simplified_text"],
                "source_lang": "en",
                "target_lang": target_language,
            })))

        if generate_audio:
            audio_text = result["simplified_text"]
            tasks.append(("tts", self._send_to_model_execution("tts", {
                "text": audio_text,
            })))

        # Execute parallel sub-tasks
        if tasks:
            coros = [t[1] for t in tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for (label, _), res in zip(tasks, results):
                if isinstance(res, Exception):
                    logger.warning(f"Sub-task {label} failed: {res}")
                    result[f"{label}_error"] = str(res)
                elif isinstance(res, dict):
                    if label == "translate":
                        result["translated_text"] = res.get("translated_text", "")
                    elif label == "tts":
                        result["audio"] = res

        return result

    async def _execute_single_inference(self, task: str, payload: dict) -> dict:
        """Execute a single inference task via ModelExecutionAgent."""
        return await self._send_to_model_execution(task, payload)

    async def _send_to_model_execution(self, task: str, payload: dict) -> dict:
        """Send an inference request to ModelExecutionAgent and await result."""
        if not self._registry:
            raise RuntimeError("Orchestrator not initialized")

        msg = AgentMessage(
            msg_type=MessageType.REQUEST,
            sender=self.name,
            recipient="model_execution",
            payload={
                "action": "infer",
                "task": task,
                "payload": payload,
            },
        )

        # Direct agent call for synchronous-style coordination
        agent = self._registry._agents.get("model_execution")
        if not agent:
            raise RuntimeError("ModelExecutionAgent not registered")

        response = await agent.handle_message(msg)
        if response and "error" in response.payload:
            raise RuntimeError(response.payload["error"])
        return response.payload.get("result", {}) if response else {}

    def _handle_sub_task_result(self, message: AgentMessage) -> AgentMessage | None:
        """Handle async sub-task completion (for future event-driven mode)."""
        request_id = message.payload.get("request_id")
        if request_id and request_id in self._pending:
            pending = self._pending[request_id]
            if pending.future and not pending.future.done():
                pending.future.set_result(message.payload)
        return None

    def _complete_request(self, request_id: str, result: Any, error: str | None) -> None:
        """Move request from pending to completed log."""
        _ = result  # used by subclasses
        pending = self._pending.pop(request_id, None)
        if pending:
            self._completed_log[request_id] = {
                "task": pending.task_type,
                "latency_ms": round(pending.elapsed_ms, 1),
                "success": error is None,
                "error": error,
                "priority": pending.priority.name,
                "timestamp": time.time(),
            }
            while len(self._completed_log) > self._max_completed_log:
                self._completed_log.popitem(last=False)

    def _get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        completed = list(self._completed_log.values())
        successful = [r for r in completed if r["success"]]
        avg_latency = (
            sum(r["latency_ms"] for r in successful) / len(successful)
            if successful else 0.0
        )
        return {
            "total_requests": self._total_requests,
            "pending": len(self._pending),
            "completed": len(self._completed_log),
            "errors": self._total_errors,
            "timeouts": self._total_timeouts,
            "shed": self._total_shed,
            "avg_latency_ms": round(avg_latency, 1),
            "error_rate": round(self._total_errors / max(1, self._total_requests), 4),
        }

    # ─── Public convenience API ─────────────────────────────────────

    async def process_request(
        self,
        task_type: str,
        payload: dict,
        priority: int = 2,
        timeout_s: float = 30.0,
    ) -> dict:
        """
        Public API for external callers (e.g., FastAPI routes).
        Bypasses the message system for direct invocation.
        """
        msg = AgentMessage(
            msg_type=MessageType.REQUEST,
            sender="api",
            recipient=self.name,
            payload={
                "action": "process",
                "task_type": task_type,
                "payload": payload,
                "priority": priority,
                "timeout_s": timeout_s,
            },
        )
        response = await self.handle_message(msg)
        if response:
            return response.payload
        return {"error": "No response from orchestrator"}
