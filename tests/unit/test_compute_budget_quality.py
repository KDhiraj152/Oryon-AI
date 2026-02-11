"""Quality invariance tests for compute budget system.

Verifies that no optimization degrades:
  - Factual accuracy (correct compute class for domain queries)
  - Logical coherence (budget fields are self-consistent)
  - Instruction adherence (safety is never skipped)
  - Edge-case handling (misclassification scenarios)
"""
import pytest
from backend.services.ai_core.compute_budget import (
    ComputeClass, ComputeBudget, classify_compute, get_budget, COMPUTE_BUDGETS,
)
from backend.services.ai_core.formatter import Intent
from backend.services.ai_core.router import ModelTier, TIER_ORDER


# ── Quality dimension 1: Factual accuracy ──────────────────────────────

class TestFactualAccuracy:
    """Queries that need factual precision never get under-computed."""

    @pytest.mark.parametrize("msg,intent", [
        ("What is the correct dosage for ibuprofen?", Intent.QUESTION),
        ("What are the symptoms of dengue fever?", Intent.QUESTION),
        ("Is radioactive decay random?", Intent.QUESTION),
        ("What is the treatment for snake bite?", Intent.QUESTION),
        ("What is the correct answer for board exam Q5?", Intent.QUESTION),
        ("name the chemical equation for rusting", Intent.QUESTION),
    ])
    def test_safety_critical_queries_are_critical(self, msg, intent):
        assert classify_compute(msg, intent) == ComputeClass.CRITICAL

    @pytest.mark.parametrize("msg,intent", [
        ("What is photosynthesis?", Intent.QUESTION),
        ("Explain the water cycle", Intent.EXPLANATION),
        ("What is gravity?", Intent.QUESTION),
        ("How does a battery work?", Intent.QUESTION),
        ("Translate to Hindi please", Intent.TRANSLATION),
    ])
    def test_standard_queries_get_standard(self, msg, intent):
        result = classify_compute(msg, intent)
        assert result in (ComputeClass.STANDARD, ComputeClass.COMPLEX), \
            f"Expected STANDARD or COMPLEX, got {result.value}"

    @pytest.mark.parametrize("msg,intent", [
        ("Is gravity a force?", Intent.QUESTION),
        ("Can water conduct electricity?", Intent.QUESTION),
        ("Does the earth revolve around the sun?", Intent.QUESTION),
    ])
    def test_short_yesno_questions_not_trivial(self, msg, intent):
        """Short yes/no questions about real topics must NOT be TRIVIAL."""
        result = classify_compute(msg, intent)
        assert result != ComputeClass.TRIVIAL, \
            f"'{msg}' was classified as TRIVIAL — under-computation failure"


# ── Quality dimension 2: Logical coherence ─────────────────────────────

class TestLogicalCoherence:
    """Budget fields are internally consistent."""

    def test_budgets_are_monotonically_escalating(self):
        """Higher compute class → higher or equal budget for all fields."""
        order = [ComputeClass.TRIVIAL, ComputeClass.STANDARD,
                 ComputeClass.COMPLEX, ComputeClass.CRITICAL]
        for i in range(len(order) - 1):
            lo = get_budget(order[i])
            hi = get_budget(order[i + 1])
            assert lo.max_tokens <= hi.max_tokens, \
                f"{lo.compute_class.value}.max_tokens > {hi.compute_class.value}"
            assert lo.max_latency_ms <= hi.max_latency_ms, \
                f"{lo.compute_class.value}.max_latency_ms > {hi.compute_class.value}"
            assert lo.max_reasoning_depth <= hi.max_reasoning_depth
            assert lo.model_count_ceiling <= hi.model_count_ceiling
            assert lo.acceptable_approximation_error >= hi.acceptable_approximation_error, \
                "Higher class should have LOWER acceptable error"

    def test_tier_ceilings_escalate(self):
        order = [ComputeClass.TRIVIAL, ComputeClass.STANDARD,
                 ComputeClass.COMPLEX, ComputeClass.CRITICAL]
        for i in range(len(order) - 1):
            lo = get_budget(order[i])
            hi = get_budget(order[i + 1])
            assert TIER_ORDER[lo.model_tier_ceiling] <= TIER_ORDER[hi.model_tier_ceiling]

    def test_all_budgets_frozen(self):
        """Budgets must be immutable."""
        for cls, budget in COMPUTE_BUDGETS.items():
            with pytest.raises(AttributeError):
                budget.max_tokens = 9999


# ── Quality dimension 3: Instruction adherence ────────────────────────

class TestInstructionAdherence:
    """Safety pipeline is never skipped — instruction from quality audit."""

    def test_all_compute_classes_run_safety(self):
        """Every compute class must have run_safety=True."""
        for cls, budget in COMPUTE_BUDGETS.items():
            assert budget.run_safety is True, \
                f"{cls.value} has run_safety=False — safety must never be skipped"

    def test_critical_forces_rag(self):
        """CRITICAL queries must force RAG for verified sources."""
        budget = get_budget(ComputeClass.CRITICAL)
        assert budget.use_rag is True

    def test_trivial_skips_rag(self):
        """TRIVIAL queries should skip RAG (greetings don't need documents)."""
        budget = get_budget(ComputeClass.TRIVIAL)
        assert budget.use_rag is False


# ── Quality dimension 4: Edge-case handling ────────────────────────────

class TestEdgeCases:
    """Intent misclassification and boundary scenarios."""

    @pytest.mark.parametrize("msg,intent", [
        ("hello can you explain quantum physics?", Intent.SMALL_TALK),
        ("hi what is the speed of light?", Intent.SMALL_TALK),
        ("hey tell me about World War 2", Intent.SMALL_TALK),
    ])
    def test_small_talk_misclass_with_question(self, msg, intent):
        """If detect_intent wrongly says SMALL_TALK but message has a question → NOT TRIVIAL."""
        result = classify_compute(msg, intent)
        assert result != ComputeClass.TRIVIAL, \
            f"'{msg}' with misclassified SMALL_TALK intent was TRIVIAL"

    @pytest.mark.parametrize("msg,intent", [
        ("hello", Intent.SMALL_TALK),
        ("hi", Intent.SMALL_TALK),
        ("thanks", Intent.SMALL_TALK),
        ("bye", Intent.SMALL_TALK),
        ("ok", Intent.SMALL_TALK),
        ("namaste", Intent.SMALL_TALK),
        ("good morning", Intent.SMALL_TALK),
    ])
    def test_genuine_greetings_are_trivial(self, msg, intent):
        assert classify_compute(msg, intent) == ComputeClass.TRIVIAL

    def test_empty_message(self):
        result = classify_compute("", Intent.UNKNOWN)
        assert result == ComputeClass.TRIVIAL

    def test_very_long_message_is_complex(self):
        msg = "explain " + "word " * 100
        result = classify_compute(msg, Intent.QUESTION)
        assert result == ComputeClass.COMPLEX

    def test_multiple_questions_are_complex(self):
        msg = "What is DNA? How does it replicate? What is RNA?"
        result = classify_compute(msg, Intent.QUESTION)
        assert result == ComputeClass.COMPLEX

    @pytest.mark.parametrize("msg", [
        "ok so what about acid in chemistry?",
        "hey so can you help me with math?",
        "sure but what is Newton's third law?",
    ])
    def test_greeting_prefix_with_question_not_trivial(self, msg):
        result = classify_compute(msg, Intent.QUESTION)
        assert result != ComputeClass.TRIVIAL
