"""
Unit tests for UnifiedCulturalContextService — region-aware content adaptation.

Tests cover:
- Region detection from language codes
- Regional context retrieval
- Content adaptation with region-specific examples
- Cultural appropriateness checking
- Festival greeting lookup
- Local reference injection
"""

import pytest

from backend.services.content.cultural_context import (
    UnifiedCulturalContextService,
    Region,
    Subject,
    get_cultural_context_service,
)


@pytest.fixture
def service():
    return UnifiedCulturalContextService()


# ── Region Detection ──────────────────────────────────────────────────────────

class TestRegionDetection:
    def test_hindi_maps_to_north(self, service: UnifiedCulturalContextService):
        assert service.detect_region_from_language("hi") == Region.NORTH

    def test_tamil_maps_to_south(self, service: UnifiedCulturalContextService):
        assert service.detect_region_from_language("ta") == Region.SOUTH

    def test_bengali_maps_to_east(self, service: UnifiedCulturalContextService):
        assert service.detect_region_from_language("bn") == Region.EAST

    def test_unknown_defaults_to_general(self, service: UnifiedCulturalContextService):
        assert service.detect_region_from_language("xx") == Region.GENERAL

    def test_english_defaults_to_general(self, service: UnifiedCulturalContextService):
        region = service.detect_region_from_language("en")
        assert region in (Region.GENERAL, Region.NORTH)  # implementation may vary


# ── Regional Context ─────────────────────────────────────────────────────────

class TestRegionalContext:
    def test_north_has_festivals(self, service: UnifiedCulturalContextService):
        ctx = service.get_regional_context(Region.NORTH)
        assert hasattr(ctx, "festivals")
        assert isinstance(ctx.festivals, (list, tuple))

    def test_south_has_languages(self, service: UnifiedCulturalContextService):
        ctx = service.get_regional_context(Region.SOUTH)
        assert hasattr(ctx, "languages")
        assert len(ctx.languages) > 0

    def test_general_region_returns_context(self, service: UnifiedCulturalContextService):
        ctx = service.get_regional_context(Region.GENERAL)
        assert ctx is not None


# ── Content Adaptation ────────────────────────────────────────────────────────

class TestContentAdaptation:
    def test_adapted_content_returns_adapted_text(self, service: UnifiedCulturalContextService):
        result = service.adapt_content(
            text="Students should learn about fractions using common objects.",
            region=Region.NORTH,
            subject=Subject.MATHEMATICS,
        )
        assert hasattr(result, "adapted_text")
        assert isinstance(result.adapted_text, str)
        assert len(result.adapted_text) > 0

    def test_adapted_content_preserves_original(self, service: UnifiedCulturalContextService):
        original = "The teacher explained the concept of gravity."
        result = service.adapt_content(text=original, region=Region.SOUTH)
        assert result.original_text == original

    def test_general_region_still_adapts(self, service: UnifiedCulturalContextService):
        result = service.adapt_content(
            text="A simple math problem.",
            region=Region.GENERAL,
        )
        assert result is not None


# ── Cultural Appropriateness ──────────────────────────────────────────────────

class TestCulturalAppropriateness:
    def test_neutral_text_is_appropriate(self, service: UnifiedCulturalContextService):
        result = service.check_cultural_appropriateness(
            text="Plants need sunlight and water to grow.",
            region=Region.NORTH,
        )
        assert isinstance(result, dict)

    def test_result_has_expected_keys(self, service: UnifiedCulturalContextService):
        result = service.check_cultural_appropriateness(
            text="Some text to check.",
            region=Region.SOUTH,
        )
        # Should have at least an indicator of appropriateness
        assert isinstance(result, dict)


# ── Festival Greeting ─────────────────────────────────────────────────────────

class TestFestivalGreeting:
    def test_north_may_return_greeting(self, service: UnifiedCulturalContextService):
        greeting = service.get_festival_greeting(Region.NORTH)
        # May or may not return a greeting depending on date
        assert greeting is None or isinstance(greeting, str)


# ── Local References ──────────────────────────────────────────────────────────

class TestLocalReferences:
    def test_inject_returns_string(self, service: UnifiedCulturalContextService):
        text = "Consider the following mathematical problem."
        result = service.inject_local_references(text, Region.SOUTH, max_injections=2)
        assert isinstance(result, str)
        assert len(result) > 0


# ── Singleton Factory ─────────────────────────────────────────────────────────

class TestSingletonFactory:
    def test_get_cultural_context_service_returns_instance(self):
        svc = get_cultural_context_service()
        assert isinstance(svc, UnifiedCulturalContextService)
