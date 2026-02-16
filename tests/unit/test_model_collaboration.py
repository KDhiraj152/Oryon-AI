"""
Tests for Model Collaboration System.

Tests multi-model communication and collaboration patterns.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestModelCollaboration:
    """Tests for the ModelCollaborator class."""

    def test_collaboration_imports(self):
        """Test that collaboration module imports correctly."""
        from backend.ml.pipeline.model_collaboration import (
            CollaborationConfig,
            CollaborationPattern,
            CollaborationResult,
            ModelCollaborator,
            ModelMessage,
            get_model_collaborator,
        )

        assert isinstance(ModelCollaborator, type)  # Verify class is importable
        assert len(CollaborationPattern) >= 5
        assert isinstance(CollaborationResult, type)  # Verify class is importable

    def test_collaboration_patterns(self):
        """Test all collaboration patterns are defined."""
        from backend.ml.pipeline.model_collaboration import CollaborationPattern

        patterns = [p.value for p in CollaborationPattern]

        assert "chain" in patterns
        assert "verify" in patterns
        assert "back_translate" in patterns
        assert "ensemble" in patterns
        assert "semantic_check" in patterns

    def test_collaborator_initialization(self):
        """Test ModelCollaborator initializes correctly."""
        from backend.ml.pipeline.model_collaboration import (
            CollaborationConfig,
            ModelCollaborator,
        )

        config = CollaborationConfig(
            min_confidence=0.8,
            semantic_threshold=0.85,
            max_iterations=3,
        )

        collaborator = ModelCollaborator(config)

        assert abs(collaborator.config.min_confidence - 0.8) < 1e-9
        assert abs(collaborator.config.semantic_threshold - 0.85) < 1e-9
        assert collaborator.config.max_iterations == 3

    def test_singleton_collaborator(self):
        """Test get_model_collaborator returns singleton."""
        from backend.ml.pipeline.model_collaboration import get_model_collaborator

        c1 = get_model_collaborator()
        c2 = get_model_collaborator()

        assert c1 is c2

    def test_collaboration_config_defaults(self):
        """Test CollaborationConfig has sensible defaults."""
        from backend.ml.pipeline.model_collaboration import CollaborationConfig

        config = CollaborationConfig()

        assert abs(config.min_confidence - 0.8) < 1e-9
        assert abs(config.semantic_threshold - 0.85) < 1e-9
        assert abs(config.consensus_threshold - 0.7) < 1e-9
        assert config.max_iterations == 3
        assert config.enable_back_translation is True
        assert config.enable_semantic_verification is True

    def test_model_message_creation(self):
        """Test ModelMessage dataclass."""
        from backend.ml.pipeline.model_collaboration import ModelMessage

        msg = ModelMessage(
            from_model="qwen",
            to_model="validator",
            content="Test content",
            context={"grade": 8},
        )

        assert msg.from_model == "qwen"
        assert msg.to_model == "validator"
        assert msg.content == "Test content"
        assert msg.context["grade"] == 8

        # Test with_context
        new_msg = msg.with_context(subject="Science")
        assert new_msg.context["grade"] == 8
        assert new_msg.context["subject"] == "Science"

    @pytest.mark.asyncio
    async def test_fallback_result(self):
        """Test fallback result when collaboration fails."""
        from backend.ml.pipeline.model_collaboration import (
            CollaborationPattern,
            ModelCollaborator,
        )

        collaborator = ModelCollaborator()

        result = collaborator._fallback_result(
            CollaborationPattern.VERIFY, "fallback text"
        )

        assert result.final_output == "fallback text"
        assert abs(result.confidence - 0.5) < 1e-9
        assert result.consensus is False
        assert result.metadata.get("fallback") is True

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from backend.ml.pipeline.model_collaboration import ModelCollaborator

        collaborator = ModelCollaborator()

        # Test identical vectors
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert collaborator._cosine_similarity(a, b) == pytest.approx(1.0)

        # Test orthogonal vectors
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert collaborator._cosine_similarity(a, b) == pytest.approx(0.0)

        # Test similar vectors
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        similarity = collaborator._cosine_similarity(a, b)
        assert 0.5 < similarity < 1.0


class TestPipelineCollaborationIntegration:
    """Test integration of collaboration with pipeline."""

    def test_processing_request_collaboration_fields(self):
        """Test ProcessingRequest has collaboration fields."""
        from backend.ml.pipeline import ProcessingRequest

        request = ProcessingRequest(
            text="Test",
            enable_collaboration=True,
            collaboration_pattern="verify",
            verify_translation=True,
        )

        assert request.enable_collaboration is True
        assert request.collaboration_pattern == "verify"
        assert request.verify_translation is True

    def test_processing_result_collaboration_fields(self):
        """Test ProcessingResult has collaboration fields."""
        from backend.ml.pipeline import ProcessingResult

        result = ProcessingResult(
            request_id="test-123",
            success=True,
            original_text="Test",
            collaboration_confidence=0.95,
            collaboration_consensus=True,
            models_used=["qwen3-8b", "bge-m3"],
            model_scores={"llm": 0.9, "semantic": 0.95},
        )

        assert abs(result.collaboration_confidence - 0.95) < 1e-9
        assert result.collaboration_consensus is True
        assert "qwen3-8b" in result.models_used
        assert abs(result.model_scores["llm"] - 0.9) < 1e-9

    def test_pipeline_exports_collaboration(self):
        """Test pipeline module exports collaboration classes."""
        from backend.ml.pipeline import (
            CollaborationPattern,
            CollaborationResult,
            ModelCollaborator,
            collaborate_and_simplify,
            collaborate_and_translate,
            ensemble_evaluate,
            get_model_collaborator,
        )

        assert ModelCollaborator is not None
        assert CollaborationPattern is not None
        assert callable(get_model_collaborator)
        assert callable(collaborate_and_simplify)
        assert callable(collaborate_and_translate)
        assert callable(ensemble_evaluate)


class TestCollaborationConvenienceFunctions:
    """Test convenience functions for collaboration."""

    @pytest.mark.asyncio
    async def test_collaborate_and_simplify_signature(self):
        """Test collaborate_and_simplify function signature."""
        import inspect

        from backend.ml.pipeline.model_collaboration import (
            collaborate_and_simplify,
        )

        sig = inspect.signature(collaborate_and_simplify)
        params = list(sig.parameters.keys())

        assert "text" in params
        assert "subject" in params
        assert "verify" in params

    @pytest.mark.asyncio
    async def test_collaborate_and_translate_signature(self):
        """Test collaborate_and_translate function signature."""
        import inspect

        from backend.ml.pipeline.model_collaboration import (
            collaborate_and_translate,
        )

        sig = inspect.signature(collaborate_and_translate)
        params = list(sig.parameters.keys())

        assert "text" in params
        assert "target_language" in params
        assert "verify_quality" in params

    @pytest.mark.asyncio
    async def test_ensemble_evaluate_signature(self):
        """Test ensemble_evaluate function signature."""
        import inspect

        from backend.ml.pipeline.model_collaboration import ensemble_evaluate

        sig = inspect.signature(ensemble_evaluate)
        params = list(sig.parameters.keys())

        assert "original_text" in params
        assert "processed_text" in params
        assert "subject" in params


class TestSevenModelCollaboration:
    """Tests for 7-model collaboration patterns."""

    def test_new_patterns_defined(self):
        """Test new collaboration patterns are defined."""
        from backend.ml.pipeline.model_collaboration import CollaborationPattern

        patterns = [p.value for p in CollaborationPattern]

        # Original patterns
        assert "chain" in patterns
        assert "verify" in patterns
        assert "back_translate" in patterns
        assert "ensemble" in patterns
        assert "semantic_check" in patterns
        assert "iterative" in patterns

        # NEW 8-model patterns
        assert "audio_verify" in patterns
        assert "document_chain" in patterns
        assert "rerank" in patterns

    def test_collaborator_has_all_model_accessors(self):
        """Test ModelCollaborator has accessors for all 7 models."""
        from backend.ml.pipeline.model_collaboration import ModelCollaborator

        collaborator = ModelCollaborator()

        # Check all 7 model accessors exist
        assert hasattr(collaborator, "_get_llm")
        assert hasattr(collaborator, "_get_translator")
        assert hasattr(collaborator, "_get_embedder")
        assert hasattr(collaborator, "_get_reranker")  # NEW
        assert hasattr(collaborator, "_get_validator")
        assert hasattr(collaborator, "_get_tts")
        assert hasattr(collaborator, "_get_stt")
        assert hasattr(collaborator, "_get_ocr")  # NEW

    def test_collaborator_metrics_include_new_patterns(self):
        """Test metrics track new collaboration patterns."""
        from backend.ml.pipeline.model_collaboration import ModelCollaborator

        collaborator = ModelCollaborator()
        metrics = collaborator.get_metrics()

        # Check new metric keys
        assert "audio_verifications" in metrics
        assert "ocr_chains" in metrics
        assert "reranking_used" in metrics

    @pytest.mark.asyncio
    async def test_verify_audio_output_signature(self):
        """Test verify_audio_output function signature."""
        import inspect

        from backend.ml.pipeline.model_collaboration import verify_audio_output

        sig = inspect.signature(verify_audio_output)
        params = list(sig.parameters.keys())

        assert "text" in params
        assert "target_language" in params
        assert "max_iterations" in params

    @pytest.mark.asyncio
    async def test_process_document_signature(self):
        """Test process_document function signature."""
        import inspect

        from backend.ml.pipeline.model_collaboration import process_document

        sig = inspect.signature(process_document)
        params = list(sig.parameters.keys())

        assert "image_path" in params
        assert "target_language" in params
        assert "generate_audio" in params

    @pytest.mark.asyncio
    async def test_generate_best_output_signature(self):
        """Test generate_best_output function signature."""
        import inspect

        from backend.ml.pipeline.model_collaboration import generate_best_output

        sig = inspect.signature(generate_best_output)
        params = list(sig.parameters.keys())

        assert "text" in params
        assert "task" in params
        assert "num_candidates" in params
        assert "subject" in params

    @pytest.mark.asyncio
    async def test_full_content_pipeline_signature(self):
        """Test full_content_pipeline function signature."""
        import inspect

        from backend.ml.pipeline.model_collaboration import (
            full_content_pipeline,
        )

        sig = inspect.signature(full_content_pipeline)
        params = list(sig.parameters.keys())

        assert "content" in params
        assert "source_language" in params
        assert "target_language" in params
        assert "subject" in params
        assert "generate_audio" in params
        assert "verify_all_steps" in params

    def test_get_reranker_getter(self):
        """Test get_reranker function exists in RAG module."""
        from backend.services.chat.rag import get_reranker

        assert callable(get_reranker)

    def test_get_embedder_getter(self):
        """Test get_embedder function exists in RAG module."""
        from backend.services.chat.rag import get_embedder

        assert callable(get_embedder)

    def test_get_got_ocr_service_getter(self):
        """Test get_got_ocr_service function exists in OCR module."""
        try:
            from backend.ml.ocr.ocr import get_got_ocr_service
            assert callable(get_got_ocr_service)
        except ImportError:
            pytest.skip("PIL/Pillow not installed")

    def test_new_functions_exported_from_pipeline(self):
        """Test new functions are exported from pipeline module."""
        from backend.ml.pipeline import (
            full_content_pipeline,
            generate_best_output,
            process_document,
            verify_audio_output,
        )

        assert callable(verify_audio_output)
        assert callable(process_document)
        assert callable(generate_best_output)
        assert callable(full_content_pipeline)
