"""
Unit tests for backend services.
Tests core functionality of translation and content_domain validation services.

Note: Storage and VLLMClient tests removed - these modules have been superseded by:
- Storage: Direct file handling in API routes
- VLLMClient: MLX-based inference via core.optimized
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Translation Service Tests
from backend.ml.translate import TranslationService


class TestTranslationService:
    """Tests for TranslationService."""

    @pytest.fixture
    def translation_service(self):
        """Create TranslationService instance."""
        return TranslationService()

    @pytest.fixture
    def mock_translation_engine(self):
        """Mock TranslationEngine."""
        with patch("backend.ml.translate.TranslationEngine") as mock:
            instance = mock.return_value
            instance.translate.return_value = "translated text"
            yield instance

    @pytest.mark.asyncio
    async def test_translate_async_basic(
        self, translation_service, mock_translation_engine
    ):
        """Test async translation with basic text."""
        translation_service.engine = mock_translation_engine

        result = await translation_service.translate_async(
            text="Hello world", source_lang="en", target_lang="hi"
        )

        assert result == "translated text"
        # The engine is called with keyword arguments
        mock_translation_engine.translate.assert_called_once_with(
            text="Hello world",
            target_language="hi",
            subject="General",
            source_language="en",
        )

    @pytest.mark.asyncio
    async def test_translate_async_empty_text(
        self, translation_service, mock_translation_engine
    ):
        """Test async translation with empty text."""
        translation_service.engine = mock_translation_engine
        mock_translation_engine.translate.return_value = ""

        result = await translation_service.translate_async(
            text="", source_lang="en", target_lang="hi"
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_async_multiple_languages(
        self, translation_service, mock_translation_engine
    ):
        """Test async translation with various language pairs."""
        translation_service.engine = mock_translation_engine

        language_pairs = [
            ("en", "hi", "Hello", "नमस्ते"),
            ("hi", "en", "नमस्ते", "Hello"),
            ("en", "ta", "Welcome", "வரவேற்பு"),
        ]

        for source, target, input_text, expected in language_pairs:
            mock_translation_engine.translate.return_value = expected
            result = await translation_service.translate_async(
                text=input_text, source_lang=source, target_lang=target
            )
            assert result == expected


# ============================================================================
# Device Router Tests (Replaces Storage/VLLMClient)
# ============================================================================


class TestDeviceRouter:
    """Tests for optimized DeviceRouter."""

    @pytest.fixture
    def device_router(self):
        """Get DeviceRouter instance."""
        from backend.core.optimized import get_device_router

        return get_device_router()

    def test_device_router_initialization(self, device_router):
        """Test device router initializes correctly."""
        assert device_router is not None
        assert device_router.capabilities is not None
        # Check capabilities has expected attributes
        assert device_router.capabilities.memory_gb > 0

    def test_device_info(self, device_router):
        """Test device info retrieval."""
        info = device_router.get_info()

        assert (
            "chip" in info
            or "chip_name" in info
            or device_router.capabilities.chip_name
        )
        assert device_router.capabilities.memory_gb > 0

    def test_task_device_routing(self, device_router):
        """Test task-specific device routing."""
        from backend.core.optimized.device_router import TaskType

        tasks = [
            TaskType.EMBEDDING,
            TaskType.RERANKING,
            TaskType.TRANSLATION,
            TaskType.TTS,
            TaskType.STT,
            TaskType.LLM_INFERENCE,
        ]

        for task in tasks:
            decision = device_router.route(task)
            assert decision.backend is not None
            assert decision.device_str != ""


class TestModelManager:
    """Tests for optimized ModelManager."""

    @pytest.fixture
    def model_manager(self):
        """Get ModelManager instance."""
        from backend.core.optimized import get_model_manager

        return get_model_manager()

    def test_model_manager_initialization(self, model_manager):
        """Test model manager initializes correctly."""
        assert model_manager is not None
        # Should have stats method
        assert hasattr(model_manager, "get_stats")

    def test_model_types_defined(self, model_manager):
        """Test model types are properly defined."""
        from backend.core.optimized import ModelType

        expected_types = ["EMBEDDING", "LLM", "RERANKER", "TTS", "STT"]
        for model_type in expected_types:
            assert hasattr(ModelType, model_type)

    def test_model_manager_methods(self, model_manager):
        """Test model manager has expected methods."""
        expected_methods = [
            "get_embedding_model",
            "get_llm_model",
            "get_reranker_model",
            "get_tts_model",
            "get_stt_model",
            "is_loaded",
            "get_stats",
        ]

        for method in expected_methods:
            assert hasattr(model_manager, method)


class TestAsyncBatchProcessor:
    """Tests for async batch processor."""

    def test_batch_processor_initialization(self):
        """Test batch processor can be imported and has correct signature."""
        import inspect

        from backend.core.optimized import AsyncBatchProcessor

        # Check constructor signature
        sig = inspect.signature(AsyncBatchProcessor.__init__)
        params = list(sig.parameters.keys())

        assert "processor" in params
        assert "batch_size" in params

    @pytest.mark.asyncio
    async def test_batch_processor_with_mock(self):
        """Test batch processor with mock processor function."""
        from backend.core.optimized import AsyncBatchProcessor

        # Create mock processor
        async def mock_processor(items):
            return [f"processed_{i}" for i in items]

        proc = AsyncBatchProcessor(processor=mock_processor, batch_size=32)
        assert proc is not None


# Content Validator Tests
from backend.services.content.content_validation import ContentValidationService


class TestCurriculumValidator:
    """Tests for ContentValidationService."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""

        mock = MagicMock()
        mock.query.return_value.filter.return_value.all.return_value = []
        return mock

    @pytest.fixture
    def validator(self, mock_db):
        """Create ContentValidationService with mocked db."""
        return ContentValidationService(db=mock_db)

    def test_grade_ranges_defined(self, validator):
        """Test validator has core attributes."""
        assert validator.alignment_threshold == 0.70
        assert validator.validator is not None

    def test_subjects_defined(self, validator):
        """Test ContentValidator is initialized."""
        from backend.services.content.validate.content_domain import ContentValidator

        assert isinstance(validator.validator, ContentValidator)

    def test_model_initialization(self, mock_db):
        """Test validator can be initialized."""
        validator = ContentValidationService(db=mock_db)
        assert validator.alignment_threshold == 0.70
        assert validator.db is not None


# ============================================================================
# Unified Cache Tests
# ============================================================================


class TestUnifiedCache:
    """Tests for unified multi-tier cache."""

    @pytest.fixture
    def cache(self):
        """Get unified cache instance."""
        from backend.infra.cache import get_unified_cache

        return get_unified_cache()

    def test_cache_initialization(self, cache):
        """Test cache initializes correctly."""
        assert cache is not None

    def test_cache_stats(self, cache):
        """Test cache statistics."""
        stats = cache.get_stats()

        assert "l1_hits" in stats or "total_requests" in stats
        assert "l1_misses" in stats or "overall_hit_rate" in stats


class TestEmbeddingCache:
    """Tests for embedding-specific cache."""

    @pytest.fixture
    def embedding_cache(self):
        """Get embedding cache instance."""
        from backend.infra.cache.embedding_cache import get_embedding_cache

        return get_embedding_cache()

    def test_embedding_cache_initialization(self, embedding_cache):
        """Test embedding cache initializes correctly."""
        assert embedding_cache is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
