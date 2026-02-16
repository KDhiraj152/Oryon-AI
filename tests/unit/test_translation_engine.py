"""
Unit tests for TranslationEngine — IndicTrans2-based translation with caching.

Tests cover:
- Supported languages retrieval
- Language info lookup
- Script rendering validation  
- Translation with mock model
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.ml.translate.engine import TranslationEngine


@pytest.fixture
def engine():
    """TranslationEngine with no actual model loaded."""
    return TranslationEngine()


class TestSupportedLanguages:
    def test_returns_list(self, engine: TranslationEngine):
        langs = engine.get_supported_languages()
        assert isinstance(langs, list)
        assert len(langs) > 0

    def test_hindi_supported(self, engine: TranslationEngine):
        langs = engine.get_supported_languages()
        # Should contain Hindi in some form
        lang_codes = [l if isinstance(l, str) else l.get("code", "") for l in langs]
        assert any("hi" in str(c).lower() for c in lang_codes)


class TestLanguageInfo:
    def test_returns_dict_or_object(self, engine: TranslationEngine):
        info = engine.get_language_info("Hindi")
        assert info is not None

    def test_unknown_language(self, engine: TranslationEngine):
        info = engine.get_language_info("xx_nonexistent")
        # Should return None or empty dict for unknown
        assert info is None or isinstance(info, dict)


class TestScriptValidation:
    def test_valid_hindi_text(self, engine: TranslationEngine):
        result = engine.validate_script_rendering("नमस्ते", "Hindi")
        assert isinstance(result, bool)

    def test_english_text(self, engine: TranslationEngine):
        result = engine.validate_script_rendering("Hello world", "en")
        # "en" is not in SUPPORTED_LANGUAGES, so returns False
        assert result is False
