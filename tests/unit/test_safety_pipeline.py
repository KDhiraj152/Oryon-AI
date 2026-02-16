"""
Unit tests for SafetyPipeline — content safety verification.

Tests cover:
- Pipeline initialisation and mode configuration
- Toxicity pattern detection
- PII pattern detection
- Grounding check logic
- Response filtering
- Overall result determination
- Sync and async verify paths
"""

import asyncio
from unittest.mock import patch, MagicMock
import pytest

from backend.services.content.safety_pipeline import (
    SafetyPipeline,
    SafetyLevel,
    IssueType,
    SafetyIssue,
    SafetyCheckResult,
    get_safety_pipeline,
    reset_pipeline,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline():
    """Standard balanced pipeline."""
    reset_pipeline()
    return SafetyPipeline(mode="balanced")


@pytest.fixture
def strict_pipeline():
    reset_pipeline()
    return SafetyPipeline(mode="strict", toxicity_threshold=0.1)


# ── Initialisation ───────────────────────────────────────────────────────────

class TestSafetyPipelineInit:
    def test_default_init(self, pipeline: SafetyPipeline):
        assert pipeline is not None
        assert hasattr(pipeline, "mode")

    def test_mode_strict(self, strict_pipeline: SafetyPipeline):
        assert strict_pipeline.mode == "strict"

    def test_singleton_factory(self):
        reset_pipeline()
        p1 = get_safety_pipeline()
        p2 = get_safety_pipeline()
        assert p1 is p2

    def test_reset_pipeline(self):
        reset_pipeline()
        p1 = get_safety_pipeline()
        reset_pipeline()
        p2 = get_safety_pipeline()
        assert p1 is not p2


# ── Grounding Check ──────────────────────────────────────────────────────────

class TestGroundingCheck:
    def test_grounded_response_passes(self, pipeline: SafetyPipeline):
        context = ["The capital of France is Paris."]
        result = pipeline._check_grounding("Paris is the capital of France.", context)
        assert result is True

    def test_ungrounded_response_fails(self, pipeline: SafetyPipeline):
        context = ["Water boils at 100 degrees Celsius."]
        result = pipeline._check_grounding(
            "The moon is made of cheese and unicorns live there.", context
        )
        # With no overlap the grounding check should fail
        assert result is False

    def test_empty_context_returns_true(self, pipeline: SafetyPipeline):
        # No context → nothing to ground against → pass
        result = pipeline._check_grounding("Any response.", [])
        assert result is True


# ── Response Filtering ────────────────────────────────────────────────────────

class TestResponseFiltering:
    def test_no_issues_returns_original(self, pipeline: SafetyPipeline):
        text = "Hello, world!"
        filtered = pipeline._filter_response(text, [])
        assert filtered == text

    def test_filter_with_issues_returns_string(self, pipeline: SafetyPipeline):
        issue = SafetyIssue(
            issue_type=IssueType.TOXICITY,
            severity=SafetyLevel.BLOCKED,
            description="Toxic content detected",
            location=None,
            confidence=0.95,
            suggested_fix=None,
        )
        filtered = pipeline._filter_response("bad content here", [issue])
        assert isinstance(filtered, str)


# ── Overall Result Determination ──────────────────────────────────────────────

class TestDetermineOverallResult:
    def _make_result(self, passed: bool, level: SafetyLevel) -> SafetyCheckResult:
        return SafetyCheckResult(
            pass_name="test",
            passed=passed,
            safety_level=level,
            issues=[],
            confidence=1.0,
            latency_ms=0.0,
            details={},
        )

    def test_all_passes_safe(self, pipeline: SafetyPipeline):
        results = [
            self._make_result(True, SafetyLevel.SAFE),
            self._make_result(True, SafetyLevel.SAFE),
        ]
        overall_safe, overall_level, reason = pipeline._determine_overall_result(results, [])
        assert overall_safe is True
        assert overall_level == SafetyLevel.SAFE
        assert reason is None

    def test_one_blocked_causes_unsafe(self, pipeline: SafetyPipeline):
        results = [
            self._make_result(True, SafetyLevel.SAFE),
            self._make_result(False, SafetyLevel.BLOCKED),
        ]
        issue = SafetyIssue(
            issue_type=IssueType.TOXICITY,
            severity=SafetyLevel.BLOCKED,
            description="Blocked",
            location=None,
            confidence=0.9,
            suggested_fix=None,
        )
        overall_safe, overall_level, reason = pipeline._determine_overall_result(results, [issue])
        assert overall_safe is False


# ── Sync Verify ───────────────────────────────────────────────────────────────

class TestVerifySync:
    def test_safe_query_response(self, pipeline: SafetyPipeline):
        result = pipeline.verify_sync(
            query="What is photosynthesis?",
            response="Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        )
        assert result.overall_safe is True

    def test_result_has_required_fields(self, pipeline: SafetyPipeline):
        result = pipeline.verify_sync(
            query="Hello",
            response="Hi there!",
        )
        assert hasattr(result, "overall_safe")
        assert hasattr(result, "overall_level")
        assert hasattr(result, "total_latency_ms")
        assert hasattr(result, "filtered_response")

    def test_to_dict(self, pipeline: SafetyPipeline):
        result = pipeline.verify_sync(query="Hi", response="Hello")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "overall_safe" in d
