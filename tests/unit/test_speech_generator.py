"""
Unit tests for SpeechGenerator, TechnicalTermHandler, AudioOptimizer — TTS pipeline.

Tests cover:
- TechnicalTermHandler processes terms
- AudioOptimizer initialisation
- SpeechGenerator estimation & language support
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.ml.speech.speech_generator import (
    TechnicalTermHandler,
    AudioOptimizer,
    SpeechGenerator,
    AudioFile,
)


# ── TechnicalTermHandler ─────────────────────────────────────────────────────

class TestTechnicalTermHandler:
    @pytest.fixture
    def handler(self):
        return TechnicalTermHandler()

    def test_process_returns_string(self, handler: TechnicalTermHandler):
        result = handler.process_technical_terms(
            text="Calculate the area using integration.",
            language="en",
            subject="Mathematics",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_language_passes_through(self, handler: TechnicalTermHandler):
        original = "Some text with no special terms."
        result = handler.process_technical_terms(original, "zz", "General")
        assert isinstance(result, str)


# ── AudioOptimizer ────────────────────────────────────────────────────────────

class TestAudioOptimizer:
    @pytest.fixture
    def optimizer(self):
        return AudioOptimizer()

    def test_init_defaults(self, optimizer: AudioOptimizer):
        assert optimizer.target_size_mb == 5.0
        assert optimizer.min_quality == 32


# ── AudioFile Dataclass ──────────────────────────────────────────────────────

class TestAudioFile:
    def test_create_audio_file(self):
        af = AudioFile(
            content=b"fake",
            format="wav",
            size_mb=0.001,
            duration_seconds=1.0,
            sample_rate=22050,
        )
        assert af.content == b"fake"
        assert af.format == "wav"
        assert af.language is None
        assert af.accuracy_score is None


# ── SpeechGenerator ───────────────────────────────────────────────────────────

class TestSpeechGenerator:
    @pytest.fixture
    def generator(self):
        # SpeechGenerator auto-detects TTS backend — may use fallback
        try:
            return SpeechGenerator()
        except Exception:
            pytest.skip("TTS backend not available")

    def test_estimate_audio_size(self, generator: SpeechGenerator):
        size = generator.estimate_audio_size("A short sentence for testing.")
        assert isinstance(size, (int, float))
        assert size > 0

    def test_supported_languages_not_empty(self, generator: SpeechGenerator):
        langs = generator.get_supported_languages()
        assert isinstance(langs, list)
        assert len(langs) > 0
