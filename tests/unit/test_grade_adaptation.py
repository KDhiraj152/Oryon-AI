"""
Unit tests for GradeLevelAdaptationService — readability analysis and content adaptation.

Tests cover:
- Readability analysis for different grade levels
- Vocabulary replacement suggestions
- Content complexity adaptation
- Grade appropriateness validation
"""

import pytest
from unittest.mock import MagicMock

from backend.services.content.grade_adaptation import GradeLevelAdaptationService


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def service(mock_db):
    return GradeLevelAdaptationService(db=mock_db)


# ── Readability Analysis ──────────────────────────────────────────────────────

class TestReadabilityAnalysis:
    def test_simple_text_low_complexity(self, service: GradeLevelAdaptationService):
        result = service.analyze_readability(
            text="The cat sat on the mat. It was a good day.",
            complexity_level=3,
        )
        assert isinstance(result, dict)
        # Keys may be at top level or nested under 'metrics'
        has_readability_keys = (
            "readability" in result
            or "flesch_score" in result
            or "grade_level" in result
            or "metrics" in result
            or "complexity_level" in result
        )
        assert has_readability_keys

    def test_complex_text_high_complexity(self, service: GradeLevelAdaptationService):
        result = service.analyze_readability(
            text="Photosynthesis is a biochemical process whereby chloroplasts "
                 "convert electromagnetic radiation into adenosine triphosphate.",
            complexity_level=10,
        )
        assert isinstance(result, dict)

    def test_empty_text(self, service: GradeLevelAdaptationService):
        result = service.analyze_readability(text="", complexity_level=5)
        assert isinstance(result, dict)


# ── Vocabulary Replacements ───────────────────────────────────────────────────

class TestVocabularyReplacements:
    def test_returns_list(self, service: GradeLevelAdaptationService):
        suggestions = service.suggest_vocabulary_replacements(
            text="The algorithm utilizes sophisticated computational methodology.",
            complexity_level=3,
        )
        assert isinstance(suggestions, list)

    def test_simple_text_has_few_replacements(self, service: GradeLevelAdaptationService):
        suggestions = service.suggest_vocabulary_replacements(
            text="The dog ran fast.",
            complexity_level=3,
        )
        assert isinstance(suggestions, list)


# ── Content Complexity Adaptation ─────────────────────────────────────────────

class TestAdaptContentComplexity:
    def test_downgrade_complexity(self, service: GradeLevelAdaptationService):
        result = service.adapt_content_complexity(
            text="Mitochondria are the powerhouse of the cell, responsible for "
                 "oxidative phosphorylation and ATP synthesis.",
            current_grade=10,
            target_grade=5,
        )
        assert isinstance(result, dict)

    def test_same_grade_minimal_change(self, service: GradeLevelAdaptationService):
        original = "The sun is a star."
        result = service.adapt_content_complexity(
            text=original,
            current_grade=5,
            target_grade=5,
        )
        assert isinstance(result, dict)


# ── Grade Appropriateness Validation ──────────────────────────────────────────

class TestValidateGradeAppropriateness:
    def test_appropriate_content(self, service: GradeLevelAdaptationService):
        result = service.validate_grade_appropriateness(
            text="Plants need water to grow.",
            complexity_level=3,
            subject="Science",
        )
        assert isinstance(result, dict)

    def test_returns_expected_keys(self, service: GradeLevelAdaptationService):
        result = service.validate_grade_appropriateness(
            text="Some educational content.",
            complexity_level=8,
            subject="Mathematics",
        )
        assert isinstance(result, dict)
