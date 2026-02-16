"""
Agent Pipeline — Phase 2
=========================

Defines agent roles, enforces strict input/output contracts,
avoids shared mutable state, and enables parallel execution when safe.

Design principles:
- Each stage is a pure async function: (StageInput) → StageOutput
- Stages declare dependencies to enable topological parallel execution
- No shared mutable state — stages communicate through immutable data
- Pipeline validates contracts at each boundary
- Failure at any stage propagates structured errors
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

logger = logging.getLogger(__name__)

# ── Stage Status ─────────────────────────────────────────────────────────────

class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

# ── Data Contracts ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class StageInput:
    """Immutable input to a pipeline stage."""

    request_id: str
    payload: dict[str, Any]
    context: dict[str, Any] = field(default_factory=dict)  # Read-only upstream data

@dataclass(frozen=True, slots=True)
class StageOutput:
    """Immutable output from a pipeline stage."""

    stage_name: str
    status: StageStatus
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Final aggregated result from the full pipeline."""

    request_id: str
    success: bool
    stages: tuple[StageOutput, ...]
    final_result: dict[str, Any]
    total_latency_ms: float
    parallel_stages_run: int = 0
    sequential_stages_run: int = 0

# ── Stage Definition ─────────────────────────────────────────────────────────

# Stage handler type: pure async function
StageHandler = Callable[[StageInput], Awaitable[StageOutput]]

@dataclass(frozen=True, slots=True)
class PipelineStage:
    """
    Declares a single pipeline stage.

    Attributes:
        name:           Unique stage name (used for dependency resolution)
        handler:        Async callable implementing the stage logic
        depends_on:     Names of stages that must complete before this one
        optional:       If True, failure doesn't abort the pipeline
        timeout_s:      Per-stage timeout
        retry_count:    How many times to retry on failure
        required_keys:  Keys that must exist in input payload (contract check)
        output_keys:    Keys the stage promises to produce (documentation)
    """

    name: str
    handler: StageHandler
    depends_on: tuple[str, ...] = ()
    optional: bool = False
    timeout_s: float = 15.0
    retry_count: int = 0
    required_keys: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()

# ── Pipeline Engine ──────────────────────────────────────────────────────────

class AgentPipeline:
    """
    Executes an ordered set of stages with dependency-aware parallelism.

    Stages with no mutual dependencies run concurrently.
    Stages with dependencies wait for their predecessors.

    Thread-safety: the pipeline itself is stateless per-execution;
    all mutable state lives in the _PipelineExecution context.
    """

    def __init__(self, stages: Sequence[PipelineStage]):
        self._stages = tuple(stages)
        self._by_name = {s.name: s for s in self._stages}
        self._validate_graph()

    # ── Public API ───────────────────────────────────────────────────

    async def execute(
        self,
        request_id: str,
        payload: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Run the full pipeline for a single request."""
        ctx = _PipelineExecution(
            request_id=request_id,
            payload=payload,
            context=context or {},
            stages=self._stages,
            by_name=self._by_name,
        )
        return await ctx.run()

    def describe(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable description of the pipeline."""
        return [
            {
                "name": s.name,
                "depends_on": list(s.depends_on),
                "optional": s.optional,
                "timeout_s": s.timeout_s,
                "required_keys": list(s.required_keys),
                "output_keys": list(s.output_keys),
            }
            for s in self._stages
        ]

    # ── Internal ─────────────────────────────────────────────────────

    def _validate_graph(self) -> None:
        """Ensure no cycles and all dependencies refer to known stages."""
        names = set(self._by_name.keys())
        for stage in self._stages:
            unknown = set(stage.depends_on) - names
            if unknown:
                raise ValueError(
                    f"Stage '{stage.name}' depends on unknown stages: {unknown}"
                )
        # Topological cycle check (Kahn's algorithm)
        in_deg = {s.name: len(s.depends_on) for s in self._stages}
        queue = [n for n, d in in_deg.items() if d == 0]
        visited = 0
        adj: dict[str, list[str]] = {s.name: [] for s in self._stages}
        for s in self._stages:
            for dep in s.depends_on:
                adj[dep].append(s.name)

        while queue:
            node = queue.pop(0)
            visited += 1
            for child in adj[node]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)

        if visited != len(self._stages):
            raise ValueError("Pipeline has a dependency cycle")

# ── Execution Context ────────────────────────────────────────────────────────

class _PipelineExecution:
    """Per-request mutable execution context. Not shared across requests."""

    def __init__(
        self,
        request_id: str,
        payload: dict[str, Any],
        context: dict[str, Any],
        stages: tuple[PipelineStage, ...],
        by_name: dict[str, PipelineStage],
    ):
        self.request_id = request_id
        self.payload = payload
        self.context = dict(context)  # shallow-copy — stages can't mutate original
        self.stages = stages
        self.by_name = by_name

        # Mutable execution state
        self.outputs: dict[str, StageOutput] = {}
        self.events: dict[str, asyncio.Event] = {s.name: asyncio.Event() for s in stages}
        self.parallel_count = 0
        self.sequential_count = 0

    async def run(self) -> PipelineResult:
        t0 = time.perf_counter()

        # Launch all stages as concurrent tasks
        tasks = [
            asyncio.create_task(self._run_stage(stage), name=f"stage-{stage.name}")
            for stage in self.stages
        ]

        # Await all tasks (each stage internally waits for its deps)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect outputs
        ordered_outputs: list[StageOutput] = []
        any_failure = False

        for stage, result in zip(self.stages, results, strict=False):
            if isinstance(result, Exception):
                output = StageOutput(
                    stage_name=stage.name,
                    status=StageStatus.FAILED,
                    error=str(result),
                )
                if not stage.optional:
                    any_failure = True
            else:
                output = self.outputs.get(stage.name, StageOutput(
                    stage_name=stage.name,
                    status=StageStatus.SKIPPED,
                ))
                if output.status == StageStatus.FAILED and not stage.optional:
                    any_failure = True
            ordered_outputs.append(output)

        # Aggregate final result from last successful non-optional stage
        final = {}
        for out in reversed(ordered_outputs):
            if out.status == StageStatus.COMPLETED:
                final = dict(out.result)
                break

        # Merge context contributions from all successful stages
        for out in ordered_outputs:
            if out.status == StageStatus.COMPLETED and out.result:
                for k, v in out.result.items():
                    if k not in final:
                        final[k] = v

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            request_id=self.request_id,
            success=not any_failure,
            stages=tuple(ordered_outputs),
            final_result=final,
            total_latency_ms=round(elapsed_ms, 2),
            parallel_stages_run=self.parallel_count,
            sequential_stages_run=self.sequential_count,
        )

    async def _run_stage(self, stage: PipelineStage) -> None:
        """Execute a single stage after its dependencies complete."""
        # Wait for dependencies
        for dep_name in stage.depends_on:
            await self.events[dep_name].wait()
            dep_output = self.outputs.get(dep_name)
            if (dep_output and dep_output.status == StageStatus.FAILED
                    and not self.by_name[dep_name].optional):
                # Required dependency failed — skip this stage
                self.outputs[stage.name] = StageOutput(
                    stage_name=stage.name,
                    status=StageStatus.SKIPPED,
                    error=f"Dependency '{dep_name}' failed",
                )
                self.events[stage.name].set()
                return

        # Track parallel vs sequential
        running_count = sum(
            1 for e in self.events.values()
            if not e.is_set()
        )
        if running_count > 1:
            self.parallel_count += 1
        else:
            self.sequential_count += 1

        # Validate input contract
        missing = [k for k in stage.required_keys if k not in self.payload and k not in self.context]
        if missing:
            self.outputs[stage.name] = StageOutput(
                stage_name=stage.name,
                status=StageStatus.FAILED,
                error=f"Missing required keys: {missing}",
            )
            self.events[stage.name].set()
            return

        # Build stage input — merge payload with upstream context
        stage_input = StageInput(
            request_id=self.request_id,
            payload=dict(self.payload),
            context=dict(self.context),
        )

        # Execute with retry + timeout
        last_error: str | None = None
        for attempt in range(1 + stage.retry_count):
            t0 = time.perf_counter()
            try:
                output = await asyncio.wait_for(
                    stage.handler(stage_input),
                    timeout=stage.timeout_s,
                )
                latency = (time.perf_counter() - t0) * 1000

                # Replace latency in output
                output = StageOutput(
                    stage_name=stage.name,
                    status=output.status,
                    result=output.result,
                    error=output.error,
                    latency_ms=round(latency, 2),
                    metadata=output.metadata,
                )
                self.outputs[stage.name] = output

                # Propagate output to context for downstream stages
                if output.status == StageStatus.COMPLETED and output.result:
                    self.context.update(output.result)

                self.events[stage.name].set()
                logger.debug(
                    f"Stage '{stage.name}' completed in {latency:.1f}ms "
                    f"(attempt {attempt + 1})"
                )
                return

            except TimeoutError:
                last_error = f"Timeout after {stage.timeout_s}s"
                logger.warning("Stage '%s' timed out (attempt %s)", stage.name, attempt + 1)
            except Exception as e:  # middleware handler
                last_error = str(e)
                logger.warning(
                    f"Stage '{stage.name}' failed (attempt {attempt + 1}): {e}"
                )

            # Brief delay before retry
            if attempt < stage.retry_count:
                await asyncio.sleep(0.1 * (attempt + 1))

        # All retries exhausted
        latency = (time.perf_counter() - t0) * 1000
        self.outputs[stage.name] = StageOutput(
            stage_name=stage.name,
            status=StageStatus.FAILED,
            error=last_error,
            latency_ms=round(latency, 2),
        )
        self.events[stage.name].set()

# ── Pipeline Builder Helpers ─────────────────────────────────────────────────

def build_chat_pipeline(
    classify_handler: StageHandler,
    rag_handler: StageHandler,
    generate_handler: StageHandler,
    format_handler: StageHandler | None = None,
) -> AgentPipeline:
    """Build a standard chat pipeline: classify → RAG → generate → format."""
    stages = [
        PipelineStage(
            name="classify",
            handler=classify_handler,
            required_keys=("prompt",),
            output_keys=("intent", "complexity", "model_target"),
            timeout_s=2.0,
        ),
        PipelineStage(
            name="rag_retrieve",
            handler=rag_handler,
            depends_on=("classify",),
            optional=True,            # Chat works without RAG
            output_keys=("rag_context",),
            timeout_s=5.0,
            retry_count=1,
        ),
        PipelineStage(
            name="generate",
            handler=generate_handler,
            depends_on=("classify", "rag_retrieve"),
            required_keys=("prompt",),
            output_keys=("response_text",),
            timeout_s=30.0,
        ),
    ]

    if format_handler:
        stages.append(
            PipelineStage(
                name="format",
                handler=format_handler,
                depends_on=("generate",),
                optional=True,
                output_keys=("formatted_response",),
                timeout_s=2.0,
            )
        )

    return AgentPipeline(stages)

def build_simplify_pipeline(
    simplify_handler: StageHandler,
    translate_handler: StageHandler,
    tts_handler: StageHandler,
) -> AgentPipeline:
    """Build simplify pipeline: simplify → (translate || TTS) in parallel."""
    return AgentPipeline([
        PipelineStage(
            name="simplify",
            handler=simplify_handler,
            required_keys=("text",),
            output_keys=("simplified_text",),
            timeout_s=15.0,
        ),
        PipelineStage(
            name="translate",
            handler=translate_handler,
            depends_on=("simplify",),
            optional=True,
            output_keys=("translated_text",),
            timeout_s=10.0,
        ),
        PipelineStage(
            name="tts",
            handler=tts_handler,
            depends_on=("simplify",),
            optional=True,
            output_keys=("audio_data",),
            timeout_s=10.0,
        ),
    ])

def build_embedding_pipeline(
    embed_handler: StageHandler,
    store_handler: StageHandler | None = None,
) -> AgentPipeline:
    """Build embedding pipeline: embed → optional store."""
    stages = [
        PipelineStage(
            name="embed",
            handler=embed_handler,
            required_keys=("texts",),
            output_keys=("embeddings",),
            timeout_s=10.0,
        ),
    ]
    if store_handler:
        stages.append(
            PipelineStage(
                name="store",
                handler=store_handler,
                depends_on=("embed",),
                optional=True,
                output_keys=("stored_ids",),
                timeout_s=5.0,
            )
        )
    return AgentPipeline(stages)
