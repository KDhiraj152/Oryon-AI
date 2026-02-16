"""
Middleware Orchestration Layer — Unit Tests
============================================

Tests all five phases:
  1. RequestClassifier (intent, complexity, routing)
  2. AgentPipeline (contracts, parallelism, failure handling)
  3. MemoryManager (short-term, long-term, retrieval, dedup)
  4. LatencyController (budgets, early-exit, adaptive timeout)
  5. SelfEvaluator (quality, routing ledger, heuristic adjustment)
  + Integration: MiddlewareOrchestrator end-to-end
"""

import asyncio
import time

import pytest

# ── Phase 1: Classifier Tests ───────────────────────────────────────────────

from backend.api.middleware.classifier import (
    ClassifiedRequest,
    ComplexityLevel,
    ModelTarget,
    RequestClassifier,
    TaskIntent,
)


class TestRequestClassifier:
    """Phase 1: Request classification tests."""

    def setup_method(self):
        self.classifier = RequestClassifier()

    def test_greeting_is_conversation(self):
        result = self.classifier.classify("hello")
        assert result.intent == TaskIntent.CONVERSATION
        assert result.complexity == ComplexityLevel.TRIVIAL
        assert result.model_target == ModelTarget.LIGHTWEIGHT

    def test_simple_question(self):
        result = self.classifier.classify("What is photosynthesis?")
        assert result.intent == TaskIntent.QUESTION
        assert result.complexity in (ComplexityLevel.TRIVIAL, ComplexityLevel.SIMPLE)

    def test_complex_question_uses_stronger_model(self):
        result = self.classifier.classify(
            "Analyze the step by step process of cellular respiration "
            "in detail, comparing aerobic and anaerobic pathways, "
            "and explain the biochemical mechanisms involved."
        )
        assert result.complexity.value >= ComplexityLevel.MODERATE.value
        assert result.model_target.value in ("standard", "strong")

    def test_translation_detected(self):
        result = self.classifier.classify("Translate hello world to hindi")
        assert result.intent == TaskIntent.TRANSLATION
        assert result.target_language == "hi"
        assert result.needs_translation is True
        assert result.model_target == ModelTarget.SPECIALIZED

    def test_code_intent(self):
        result = self.classifier.classify("Write a Python function to sort a list")
        assert result.intent == TaskIntent.CODE

    def test_calculation_skips_llm(self):
        result = self.classifier.classify("25 * 4 + 10")
        assert result.intent == TaskIntent.CALCULATION
        assert result.model_target == ModelTarget.SKIP_LLM

    def test_audio_tts(self):
        result = self.classifier.classify("Read this aloud: Hello world")
        assert result.intent == TaskIntent.AUDIO
        assert result.needs_tts is True
        assert result.model_target == ModelTarget.SPECIALIZED

    def test_summarization(self):
        result = self.classifier.classify("Summarize the theory of relativity")
        assert result.intent == TaskIntent.SUMMARIZATION

    def test_cache_hit(self):
        result1 = self.classifier.classify("hello")
        result2 = self.classifier.classify("hello")
        stats = self.classifier.get_stats()
        assert stats["cache_hits"] >= 1
        assert result1.intent == result2.intent

    def test_immutable_result(self):
        result = self.classifier.classify("hello")
        with pytest.raises(AttributeError):
            result.intent = TaskIntent.CODE  # noqa: test frozen

    def test_needs_rag_for_complex_questions(self):
        result = self.classifier.classify(
            "Explain the detailed mechanism of CRISPR gene editing"
        )
        # Moderate+ questions should need RAG
        if result.complexity >= ComplexityLevel.MODERATE:
            assert result.needs_rag is True

    def test_simplification_intent(self):
        result = self.classifier.classify("Simplify this concept for a child")
        assert result.intent == TaskIntent.SIMPLIFICATION

    def test_quiz_intent(self):
        result = self.classifier.classify("Give me a quiz on physics")
        assert result.intent == TaskIntent.QUIZ


# ── Phase 2: Pipeline Tests ─────────────────────────────────────────────────

from backend.api.middleware.pipeline import (
    AgentPipeline,
    PipelineResult,
    PipelineStage,
    StageInput,
    StageOutput,
    StageStatus,
)


class TestAgentPipeline:
    """Phase 2: Pipeline execution tests."""

    @pytest.mark.asyncio
    async def test_simple_pipeline(self):
        async def handler(inp: StageInput) -> StageOutput:
            return StageOutput(
                stage_name="test",
                status=StageStatus.COMPLETED,
                result={"value": 42},
            )

        pipeline = AgentPipeline([
            PipelineStage(name="test", handler=handler),
        ])
        result = await pipeline.execute("req-1", {"key": "val"})
        assert result.success is True
        assert result.final_result["value"] == 42

    @pytest.mark.asyncio
    async def test_dependency_chain(self):
        """Stages run in dependency order."""
        execution_order = []

        async def stage_a(inp: StageInput) -> StageOutput:
            execution_order.append("a")
            return StageOutput(
                stage_name="a",
                status=StageStatus.COMPLETED,
                result={"from_a": True},
            )

        async def stage_b(inp: StageInput) -> StageOutput:
            execution_order.append("b")
            assert inp.context.get("from_a") is True  # upstream context available
            return StageOutput(
                stage_name="b",
                status=StageStatus.COMPLETED,
                result={"from_b": True},
            )

        pipeline = AgentPipeline([
            PipelineStage(name="a", handler=stage_a),
            PipelineStage(name="b", handler=stage_b, depends_on=("a",)),
        ])
        result = await pipeline.execute("req-2", {})
        assert result.success is True
        assert execution_order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_parallel_stages(self):
        """Independent stages run concurrently."""
        timestamps = {}

        async def make_handler(name: str, delay: float):
            async def handler(inp: StageInput) -> StageOutput:
                timestamps[f"{name}_start"] = time.perf_counter()
                await asyncio.sleep(delay)
                timestamps[f"{name}_end"] = time.perf_counter()
                return StageOutput(
                    stage_name=name,
                    status=StageStatus.COMPLETED,
                    result={f"from_{name}": True},
                )
            return handler

        pipeline = AgentPipeline([
            PipelineStage(name="fast", handler=await make_handler("fast", 0.05)),
            PipelineStage(name="slow", handler=await make_handler("slow", 0.05)),
        ])
        result = await pipeline.execute("req-3", {})
        assert result.success is True

        # Both should have started nearly simultaneously
        start_diff = abs(timestamps["fast_start"] - timestamps["slow_start"])
        assert start_diff < 0.05  # Both started within 50ms

    @pytest.mark.asyncio
    async def test_optional_stage_failure(self):
        """Optional stage failure doesn't abort pipeline."""
        async def good(inp: StageInput) -> StageOutput:
            return StageOutput(
                stage_name="good",
                status=StageStatus.COMPLETED,
                result={"answer": "ok"},
            )

        async def bad(inp: StageInput) -> StageOutput:
            raise ValueError("boom")

        pipeline = AgentPipeline([
            PipelineStage(name="good", handler=good),
            PipelineStage(name="bad", handler=bad, optional=True),
        ])
        result = await pipeline.execute("req-4", {})
        assert result.success is True  # optional failure doesn't fail pipeline

    @pytest.mark.asyncio
    async def test_required_stage_failure_aborts(self):
        """Required stage failure marks pipeline as failed."""
        async def failing(inp: StageInput) -> StageOutput:
            raise RuntimeError("critical failure")

        pipeline = AgentPipeline([
            PipelineStage(name="critical", handler=failing, optional=False),
        ])
        result = await pipeline.execute("req-5", {})
        assert result.success is False

    def test_cycle_detection(self):
        """Pipeline rejects cyclic dependencies."""
        async def noop(inp: StageInput) -> StageOutput:
            return StageOutput(stage_name="a", status=StageStatus.COMPLETED)

        with pytest.raises(ValueError, match="cycle"):
            AgentPipeline([
                PipelineStage(name="a", handler=noop, depends_on=("b",)),
                PipelineStage(name="b", handler=noop, depends_on=("a",)),
            ])

    def test_unknown_dependency(self):
        """Pipeline rejects unknown dependencies."""
        async def noop(inp: StageInput) -> StageOutput:
            return StageOutput(stage_name="a", status=StageStatus.COMPLETED)

        with pytest.raises(ValueError, match="unknown"):
            AgentPipeline([
                PipelineStage(name="a", handler=noop, depends_on=("nonexistent",)),
            ])

    @pytest.mark.asyncio
    async def test_stage_timeout(self):
        """Stage respects timeout."""
        async def slow(inp: StageInput) -> StageOutput:
            await asyncio.sleep(10)  # Way too slow
            return StageOutput(stage_name="slow", status=StageStatus.COMPLETED)

        pipeline = AgentPipeline([
            PipelineStage(name="slow", handler=slow, timeout_s=0.1),
        ])
        result = await pipeline.execute("req-6", {})
        assert result.success is False
        assert result.stages[0].status == StageStatus.FAILED


# ── Phase 3: Memory Tests ───────────────────────────────────────────────────

from backend.api.middleware.memory import (
    LongTermMemory,
    MemoryManager,
    MemoryTier,
    RetrievalCache,
    RetrievalSource,
    ShortTermMemory,
)


class TestShortTermMemory:
    @pytest.mark.asyncio
    async def test_add_and_retrieve(self):
        stm = ShortTermMemory()
        await stm.add_message("s1", "user", "Hello")
        await stm.add_message("s1", "assistant", "Hi there!")

        context = await stm.get_context("s1")
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_max_messages_eviction(self):
        stm = ShortTermMemory(max_messages=5)
        for i in range(10):
            await stm.add_message("s1", "user", f"msg {i}")

        context = await stm.get_context("s1")
        assert len(context) == 5
        assert context[0]["content"] == "msg 5"  # Oldest kept

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        stm = ShortTermMemory()
        await stm.add_message("s1", "user", "Hello from s1")
        await stm.add_message("s2", "user", "Hello from s2")

        c1 = await stm.get_context("s1")
        c2 = await stm.get_context("s2")
        assert len(c1) == 1
        assert len(c2) == 1
        assert c1[0]["content"] != c2[0]["content"]

    @pytest.mark.asyncio
    async def test_summary_replacement(self):
        stm = ShortTermMemory(summarize_threshold=5)
        for i in range(8):
            await stm.add_message("s1", "user", f"msg {i}")

        assert await stm.should_summarize("s1") is True

        await stm.replace_with_summary("s1", "Discussed topics 0-7", keep_recent=3)
        context = await stm.get_context("s1")
        assert len(context) == 4  # 1 summary + 3 recent
        assert context[0]["metadata"].get("is_summary") is True


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        ltm = LongTermMemory()
        entry = await ltm.store("k1", "photosynthesis is the process...")
        assert entry.key == "k1"
        assert entry.tier == MemoryTier.LONG_TERM

        retrieved = await ltm.get("k1")
        assert retrieved is not None
        assert retrieved.content == "photosynthesis is the process..."

    @pytest.mark.asyncio
    async def test_deduplication(self):
        ltm = LongTermMemory()
        await ltm.store("k1", "identical content")
        await ltm.store("k2", "identical content")  # Same content, different key
        assert ltm.embedding_calls_saved >= 1

    @pytest.mark.asyncio
    async def test_text_search(self):
        ltm = LongTermMemory()
        await ltm.store("k1", "photosynthesis converts light energy")
        await ltm.store("k2", "mitosis is cell division")

        results = await ltm.search_by_text("photosynthesis")
        assert len(results) >= 1
        assert "photosynthesis" in results[0].content


class TestRetrievalCache:
    def test_cache_hit_miss(self):
        cache = RetrievalCache(max_entries=10, default_ttl_s=60)
        cache.put("abc", [{"id": "1", "text": "hello"}])

        result = cache.get("abc")
        assert result is not None
        assert len(result) == 1

        assert cache.get("xyz") is None
        assert cache.hit_rate > 0

    def test_ttl_expiry(self):
        cache = RetrievalCache(max_entries=10, default_ttl_s=0.01)
        cache.put("abc", [{"id": "1", "text": "hello"}])
        import time; time.sleep(0.02)
        assert cache.get("abc") is None


class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_add_and_get_context(self):
        mm = MemoryManager()
        await mm.add_message("s1", "user", "Hello")
        ctx = await mm.get_context("s1")
        assert len(ctx) == 1

    @pytest.mark.asyncio
    async def test_store_knowledge(self):
        mm = MemoryManager()
        entry = await mm.store_knowledge("k1", "Some knowledge")
        assert entry.key == "k1"

    @pytest.mark.asyncio
    async def test_retrieval_fallback(self):
        mm = MemoryManager()
        await mm.store_knowledge("k1", "quantum mechanics explains subatomic particles")

        result = await mm.retrieve("quantum mechanics")
        assert len(result.entries) >= 1
        assert result.source == RetrievalSource.DATABASE  # No vector provider

    @pytest.mark.asyncio
    async def test_stats(self):
        mm = MemoryManager()
        await mm.add_message("s1", "user", "test")
        stats = await mm.get_stats()
        assert "short_term_entries" in stats
        assert stats["short_term_entries"] >= 1


# ── Phase 4: Latency Tests ──────────────────────────────────────────────────

from backend.api.middleware.latency import (
    AdaptiveTimeout,
    ConfidenceConfig,
    ConfidenceGate,
    ExitReason,
    LatencyBudget,
    LatencyController,
    parallel_inference,
)


class TestLatencyBudget:
    def test_budget_tracking(self):
        budget = LatencyBudget(total_ms=5000)
        assert budget.remaining_ms <= 5000
        assert budget.remaining_ms > 4990  # Just created
        assert budget.is_expired is False

    def test_stage_budget(self):
        budget = LatencyBudget(
            total_ms=10000,
            stage_budgets={"generate": 5000, "translate": 3000},
        )
        assert budget.budget_for_stage("generate") == 5000
        assert budget.budget_for_stage("translate") == 3000
        assert budget.budget_for_stage("unknown") == 5000  # default


class TestConfidenceGate:
    def test_high_confidence_exits(self):
        gate = ConfidenceGate(ConfidenceConfig(threshold=0.85))
        budget = LatencyBudget(total_ms=10000)
        assert gate.should_early_exit(0.90, 100, budget) is True

    def test_low_confidence_continues(self):
        gate = ConfidenceGate(ConfidenceConfig(threshold=0.85))
        budget = LatencyBudget(total_ms=10000)
        assert gate.should_early_exit(0.50, 100, budget) is False

    def test_too_early_no_exit(self):
        gate = ConfidenceGate(ConfidenceConfig(threshold=0.85, min_latency_ms=200))
        budget = LatencyBudget(total_ms=10000)
        assert gate.should_early_exit(0.95, 10, budget) is False  # Too early


class TestAdaptiveTimeout:
    def test_insufficient_data_returns_default(self):
        at = AdaptiveTimeout()
        assert at.get_timeout_ms("unknown", default_ms=5000) == 5000

    def test_learns_from_history(self):
        at = AdaptiveTimeout(multiplier=1.5)
        for i in range(50):
            at.record("generate", 100 + i * 2)

        timeout = at.get_timeout_ms("generate")
        assert timeout > 100  # Should be above mean
        assert timeout < 30000  # Should be below max


class TestParallelInference:
    @pytest.mark.asyncio
    async def test_race_returns_first_winner(self):
        async def fast():
            await asyncio.sleep(0.01)
            return "fast_result", 0.9

        async def slow():
            await asyncio.sleep(1.0)
            return "slow_result", 0.95

        budget = LatencyBudget(total_ms=5000)
        result = await parallel_inference([fast, slow], budget, min_confidence=0.8)
        assert result.value == "fast_result"
        assert result.exit_reason == ExitReason.PARALLEL_RACE_WIN


class TestLatencyController:
    def test_create_budget(self):
        lc = LatencyController()
        budget = lc.create_budget(total_ms=10000)
        assert budget.total_ms == 10000
        assert "generate" in budget.stage_budgets

    def test_record_and_adapt(self):
        lc = LatencyController()
        for _ in range(50):
            lc.record_latency("generate", 150)

        budget = lc.create_budget()
        # Adaptive timeout should have learned from the recordings
        assert budget.stage_budgets.get("generate", 0) > 0


# ── Phase 5: Evaluator Tests ────────────────────────────────────────────────

from backend.api.middleware.evaluator import (
    EvaluationReport,
    HeuristicAdjuster,
    QualityEstimator,
    QualitySignal,
    RoutingLedger,
    RoutingOutcome,
    SelfEvaluator,
)


class TestQualityEstimator:
    def test_good_quality(self):
        qe = QualityEstimator()
        signal = qe.estimate("question", confidence=0.9, latency_ms=1000, output_tokens=200)
        assert signal == QualitySignal.GOOD

    def test_failed_on_error(self):
        qe = QualityEstimator()
        signal = qe.estimate("question", confidence=0.9, latency_ms=100, output_tokens=200, has_error=True)
        assert signal == QualitySignal.FAILED

    def test_degraded_on_slow(self):
        qe = QualityEstimator()
        signal = qe.estimate("question", confidence=0.3, latency_ms=15000, output_tokens=10)
        assert signal in (QualitySignal.DEGRADED, QualitySignal.ACCEPTABLE)


class TestRoutingLedger:
    def test_record_and_query(self):
        ledger = RoutingLedger()
        for _ in range(30):
            ledger.record(RoutingOutcome(
                intent="question",
                complexity_score=0.3,
                model_target="strong",
                quality_signal=QualitySignal.GOOD,
                latency_ms=500,
                confidence=0.9,
            ))

        quality_rate = ledger.get_quality_rate("question", "strong")
        assert quality_rate == 1.0

    def test_over_provisioning_detection(self):
        ledger = RoutingLedger()
        for _ in range(30):
            ledger.record(RoutingOutcome(
                intent="conversation",
                complexity_score=0.1,
                model_target="strong",
                quality_signal=QualitySignal.GOOD,
                latency_ms=200,
                confidence=0.95,
            ))

        issues = ledger.detect_over_provisioning()
        assert len(issues) >= 1
        assert issues[0]["intent"] == "conversation"

    def test_under_provisioning_detection(self):
        ledger = RoutingLedger()
        for _ in range(30):
            ledger.record(RoutingOutcome(
                intent="code",
                complexity_score=0.7,
                model_target="lightweight",
                quality_signal=QualitySignal.DEGRADED,
                latency_ms=5000,
                confidence=0.3,
            ))

        issues = ledger.detect_under_provisioning()
        assert len(issues) >= 1


class TestHeuristicAdjuster:
    def test_adjustment_bounded(self):
        adjuster = HeuristicAdjuster()
        old = adjuster.get_threshold("question")

        ledger = RoutingLedger()
        for _ in range(30):
            ledger.record(RoutingOutcome(
                intent="question",
                complexity_score=0.3,
                model_target="strong",
                quality_signal=QualitySignal.GOOD,
                latency_ms=500,
                confidence=0.9,
            ))

        adjustments = adjuster.adjust_from_ledger(ledger)
        new = adjuster.get_threshold("question")

        # Should have moved but within bounds
        if adjustments:
            assert new != old
            assert new >= 0.10
            assert new <= 0.80


class TestSelfEvaluator:
    def test_evaluate_records_report(self):
        se = SelfEvaluator()
        report = se.evaluate(
            request_id="test-1",
            intent="question",
            model_target="standard",
            model_used="qwen3-8b",
            complexity_score=0.4,
            total_latency_ms=2000,
            confidence=0.85,
            output_tokens=150,
        )
        assert report.quality_signal in QualitySignal
        assert report.request_id == "test-1"

    def test_auto_adjustment(self):
        se = SelfEvaluator(adjustment_interval=5)
        for i in range(10):
            se.evaluate(
                request_id=f"test-{i}",
                intent="conversation",
                model_target="strong",
                model_used="qwen3-14b",
                complexity_score=0.1,
                total_latency_ms=200,
                confidence=0.95,
                output_tokens=50,
            )

        stats = se.get_stats()
        assert stats["total_evaluations"] == 10

    def test_force_adjustment(self):
        se = SelfEvaluator()
        adjustments = se.force_adjustment()
        assert isinstance(adjustments, list)


# ── Integration: Orchestrator Tests ──────────────────────────────────────────

from backend.api.middleware.orchestrator import MiddlewareOrchestrator


class TestMiddlewareOrchestrator:
    @pytest.mark.asyncio
    async def test_process_calculation(self):
        """SKIP_LLM path — calculation without model agent."""
        mw = MiddlewareOrchestrator()
        await mw.initialize()

        response = await mw.process("25 * 4 + 10")
        assert response.success is True
        assert response.intent == "calculation"
        assert "110" in response.result.get("response_text", "")

    @pytest.mark.asyncio
    async def test_process_greeting(self):
        """Lightweight path — greeting classified correctly."""
        mw = MiddlewareOrchestrator()
        await mw.initialize()

        response = await mw.process("hello")
        assert response.intent == "conversation"
        # Without a model agent, generate will fail but intent is correct

    @pytest.mark.asyncio
    async def test_stats_available(self):
        mw = MiddlewareOrchestrator()
        await mw.initialize()
        stats = await mw.get_stats()
        assert "classifier" in stats
        assert "latency" in stats
        assert "memory" in stats
        assert "evaluation" in stats

    @pytest.mark.asyncio
    async def test_session_memory(self):
        mw = MiddlewareOrchestrator()
        await mw.initialize()

        # Process with session (even without agent, memory should work)
        await mw.process("25 + 25", session_id="test-session")
        ctx = await mw.memory.get_context("test-session")
        assert len(ctx) >= 1  # At least the user message

    @pytest.mark.asyncio
    async def test_classification_dry_run(self):
        mw = MiddlewareOrchestrator()
        result = mw.classifier.classify("Translate hello to tamil")
        assert result.intent == TaskIntent.TRANSLATION
        assert result.target_language == "ta"
        assert result.model_target == ModelTarget.SPECIALIZED
