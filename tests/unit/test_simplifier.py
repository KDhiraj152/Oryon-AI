"""
Unit tests for TextSimplifier — text complexity adaptation.

Tests cover:
- Complexity score calculation
- SimplifiedText dataclass
- Sync wrapper
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.content.simplifier import (
    TextSimplifier,
    SimplifiedText,
    BaseLLMClient,
    simplify_text_sync,
)


# ── Mock LLM Client ──────────────────────────────────────────────────────────

class MockLLMClient(BaseLLMClient):
    """Deterministic LLM client for testing — returns simplified echo."""

    async def generate(self, prompt: str, **kwargs) -> str:
        # Return a simplified version of whatever is asked
        return "Simple text for testing."


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    return MockLLMClient()


@pytest.fixture
def simplifier(mock_client):
    return TextSimplifier(client=mock_client, enable_refinement=False)


# ── Complexity Score ──────────────────────────────────────────────────────────

class TestComplexityScore:
    def test_simple_text_low_score(self, simplifier: TextSimplifier):
        score = simplifier.get_complexity_score("The cat sat on a mat.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 10.0

    def test_complex_text_higher_score(self, simplifier: TextSimplifier):
        simple_score = simplifier.get_complexity_score("The dog ran.")
        complex_score = simplifier.get_complexity_score(
            "Photosynthesis is a sophisticated biochemical process involving "
            "the conversion of electromagnetic radiation into chemical energy "
            "through a series of enzymatically catalysed reactions within "
            "chloroplast organelles."
        )
        assert complex_score >= simple_score

    def test_empty_text_returns_score(self, simplifier: TextSimplifier):
        score = simplifier.get_complexity_score("")
        assert isinstance(score, float)


# ── SimplifiedText Dataclass ──────────────────────────────────────────────────

class TestSimplifiedTextDataclass:
    def test_create(self):
        st = SimplifiedText(
            text="Hello",
            complexity_score=2.0,
            complexity_level=3,
            subject="General",
            metadata={},
            semantic_score=None,
            refinement_iterations=0,
            dimension_scores=None,
        )
        assert st.text == "Hello"
        assert st.complexity_score == 2.0


# ── Simplification ────────────────────────────────────────────────────────────

class TestSimplifyText:
    @pytest.mark.asyncio
    async def test_simplify_returns_simplified_text(self, simplifier: TextSimplifier):
        result = await simplifier.simplify_text(
            content="The mitochondria are responsible for cellular respiration.",
            complexity_level=3,
            subject="Science",
        )
        assert isinstance(result, SimplifiedText)
        assert len(result.text) > 0

    @pytest.mark.asyncio
    async def test_simplify_preserves_subject(self, simplifier: TextSimplifier):
        result = await simplifier.simplify_text(
            content="Solve the quadratic equation.",
            subject="Mathematics",
        )
        assert result.subject == "Mathematics"


# ── Sync Wrapper ──────────────────────────────────────────────────────────────

class TestSimplifyTextSync:
    def test_sync_wrapper_returns_result(self, simplifier: TextSimplifier):
        result = simplify_text_sync(
            content="A complex sentence for simplification.",
            complexity_level=5,
            subject="General",
            simplifier=simplifier,
            use_refinement=False,
        )
        assert isinstance(result, SimplifiedText)
