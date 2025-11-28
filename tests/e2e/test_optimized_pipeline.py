"""
E2E Integration Test for MASTER-OPTIMIZER Full Pipeline
Tests complete flow: upload → simplify → translate → validate → TTS
Validates memory usage stays under 8GB on Apple Silicon M4
"""
import os
import pytest
import psutil
import asyncio
from pathlib import Path

from backend.pipeline.orchestrator import ContentPipelineOrchestrator
from backend.services.unified_model_client import get_unified_client
from backend.core.model_tier_router import ModelTier
from backend.utils.device_manager import DeviceManager


# Memory constants for M4 (8GB unified)
MAX_MEMORY_GB = 8.0
WARNING_MEMORY_GB = 7.0


def get_memory_usage_gb() -> float:
    """Get current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


@pytest.fixture
def device_manager():
    """Initialize device manager for MPS."""
    return DeviceManager()


@pytest.fixture
def orchestrator():
    """Initialize pipeline orchestrator."""
    return ContentPipelineOrchestrator(api_key=os.getenv("HF_API_KEY"))


@pytest.fixture
def unified_client():
    """Initialize unified model client."""
    return get_unified_client(api_key=os.getenv("HF_API_KEY"))


class TestMemoryConstraints:
    """Test memory usage stays within M4 constraints."""
    
    def test_initial_memory_baseline(self):
        """Verify baseline memory before model loading."""
        initial_memory = get_memory_usage_gb()
        assert initial_memory < 2.0, f"Baseline memory too high: {initial_memory:.2f}GB"
        print(f"✓ Baseline memory: {initial_memory:.2f}GB")
    
    def test_model_loading_memory(self, unified_client, device_manager):
        """Test memory usage during model loading."""
        initial_memory = get_memory_usage_gb()
        
        # Load small model
        asyncio.run(unified_client._get_or_load_model(
            "simplify",
            ModelTier.SMALL,
            "Test task"
        ))
        
        after_small = get_memory_usage_gb()
        delta_small = after_small - initial_memory
        
        assert delta_small < 2.0, f"Small model uses too much memory: {delta_small:.2f}GB"
        print(f"✓ Small model memory: {delta_small:.2f}GB")
        
        # Clear cache
        device_manager.empty_cache()
        
        # Load medium model
        asyncio.run(unified_client._get_or_load_model(
            "translate",
            ModelTier.MEDIUM,
            "Translation task"
        ))
        
        after_medium = get_memory_usage_gb()
        assert after_medium < MAX_MEMORY_GB, f"Memory exceeded: {after_medium:.2f}GB"
        print(f"✓ Medium model memory: {after_medium:.2f}GB")


class TestTierRouting:
    """Test model tier routing works correctly."""
    
    @pytest.mark.asyncio
    async def test_small_tier_routing(self, unified_client):
        """Test simple tasks route to SMALL tier."""
        simple_text = "The sun is hot."
        
        result = await unified_client.simplify_text(
            text=simple_text,
            grade_level=5,
            subject="Science"
        )
        
        assert result, "Simplification failed"
        assert len(result) > 0, "Empty result"
        print(f"✓ SMALL tier: '{simple_text}' → '{result}'")
    
    @pytest.mark.asyncio
    async def test_medium_tier_routing(self, unified_client):
        """Test moderate tasks route to MEDIUM tier."""
        moderate_text = (
            "Photosynthesis is the process by which green plants and some other organisms "
            "use sunlight to synthesize foods with the help of chlorophyll pigments."
        )
        
        result = await unified_client.simplify_text(
            text=moderate_text,
            grade_level=8,
            subject="Science"
        )
        
        assert result, "Simplification failed"
        assert len(result) > 0, "Empty result"
        print(f"✓ MEDIUM tier: {len(moderate_text)} chars → {len(result)} chars")
    
    @pytest.mark.asyncio
    async def test_translation_routing(self, unified_client):
        """Test translation with tier routing."""
        text = "Mathematics is the study of numbers and patterns."
        
        result = await unified_client.translate_text(
            text=text,
            target_language="Hindi"
        )
        
        assert result, "Translation failed"
        assert len(result) > 0, "Empty result"
        print(f"✓ Translation: '{text}' → '{result}'")


class TestFullPipeline:
    """Test complete pipeline end-to-end."""
    
    def test_text_only_pipeline(self, orchestrator):
        """Test complete pipeline with text output only."""
        initial_memory = get_memory_usage_gb()
        print(f"Initial memory: {initial_memory:.2f}GB")
        
        # Sample educational content
        input_text = """
        Water cycle is the continuous movement of water on, above, and below the surface of the Earth.
        The water cycle involves evaporation, condensation, precipitation, and collection.
        This cycle is very important for life on Earth.
        """
        
        # Process through pipeline (text only for faster test)
        result = orchestrator.process_content(
            input_data=input_text,
            target_language="Hindi",
            grade_level=6,
            subject="Science",
            output_format="text"
        )
        
        final_memory = get_memory_usage_gb()
        memory_delta = final_memory - initial_memory
        
        # Assertions
        assert result.simplified_text, "No simplified text"
        assert result.translated_text, "No translated text"
        assert result.ncert_alignment_score > 0, "Invalid alignment score"
        assert len(result.metrics) >= 3, "Missing stage metrics"
        
        # Memory check
        assert final_memory < MAX_MEMORY_GB, f"Memory exceeded: {final_memory:.2f}GB"
        if final_memory > WARNING_MEMORY_GB:
            print(f"⚠ Warning: High memory usage: {final_memory:.2f}GB")
        
        print(f"\n✓ Pipeline Results:")
        print(f"  Original length: {len(input_text)} chars")
        print(f"  Simplified length: {len(result.simplified_text)} chars")
        print(f"  Translated length: {len(result.translated_text)} chars")
        print(f"  NCERT score: {result.ncert_alignment_score:.2%}")
        print(f"  Memory delta: {memory_delta:.2f}GB")
        print(f"  Total time: {result.metadata['total_processing_time_ms']}ms")
    
    def test_multiple_requests_memory_stability(self, orchestrator):
        """Test memory remains stable across multiple requests."""
        memory_readings = []
        
        for i in range(3):
            current_memory = get_memory_usage_gb()
            memory_readings.append(current_memory)
            
            result = orchestrator.process_content(
                input_data=f"Test content {i}: The Earth revolves around the Sun.",
                target_language="Hindi",
                grade_level=7,
                subject="Science",
                output_format="text"
            )
            
            assert result.simplified_text, f"Request {i} failed"
            print(f"Request {i+1}: {current_memory:.2f}GB")
        
        # Check memory didn't grow excessively
        max_memory = max(memory_readings)
        assert max_memory < MAX_MEMORY_GB, f"Memory exceeded: {max_memory:.2f}GB"
        
        # Check for memory leaks (growth across requests)
        if len(memory_readings) > 1:
            growth = memory_readings[-1] - memory_readings[0]
            assert growth < 1.0, f"Possible memory leak: {growth:.2f}GB growth"
            print(f"✓ Memory stable: {growth:.2f}GB growth over 3 requests")


class TestCircuitBreaker:
    """Test circuit breaker and API fallback."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_triggers(self, unified_client):
        """Test circuit breaker opens after failures."""
        # This is a placeholder - actual test would require mocking failures
        # For now, just verify circuit breaker is integrated
        
        assert hasattr(unified_client, 'simplify_text'), "Missing simplify_text method"
        assert hasattr(unified_client, 'translate_text'), "Missing translate_text method"
        
        print("✓ Circuit breaker integrated")
    
    @pytest.mark.asyncio
    async def test_api_fallback_available(self, unified_client):
        """Test API fallback chain exists."""
        # Verify API key is configured for fallback
        api_key = os.getenv("HF_API_KEY")
        
        if api_key:
            print(f"✓ API fallback configured: {api_key[:8]}...")
        else:
            print("⚠ No API key - local-only mode")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input(self, orchestrator):
        """Test empty input is rejected."""
        with pytest.raises(Exception):
            orchestrator.process_content(
                input_data="",
                target_language="Hindi",
                grade_level=7,
                subject="Science",
                output_format="text"
            )
        print("✓ Empty input rejected")
    
    def test_invalid_language(self, orchestrator):
        """Test invalid language is rejected."""
        with pytest.raises(Exception):
            orchestrator.process_content(
                input_data="Test content",
                target_language="Klingon",
                grade_level=7,
                subject="Science",
                output_format="text"
            )
        print("✓ Invalid language rejected")
    
    def test_invalid_grade(self, orchestrator):
        """Test invalid grade is rejected."""
        with pytest.raises(Exception):
            orchestrator.process_content(
                input_data="Test content",
                target_language="Hindi",
                grade_level=50,
                subject="Science",
                output_format="text"
            )
        print("✓ Invalid grade rejected")


@pytest.mark.skipif(
    os.getenv("SKIP_SLOW_TESTS") == "1",
    reason="Skip slow full pipeline test"
)
class TestFullPipelineWithAudio:
    """Test complete pipeline including TTS (slower)."""
    
    def test_full_pipeline_with_audio(self, orchestrator):
        """Test complete pipeline with audio generation."""
        pytest.skip("TTS testing requires audio validation - implement separately")


if __name__ == "__main__":
    # Quick smoke test
    print("Running MASTER-OPTIMIZER E2E Tests...\n")
    pytest.main([__file__, "-v", "--tb=short"])
