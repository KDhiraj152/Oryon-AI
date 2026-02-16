"""
Collaboration Types and Constants
=================================

Core enums, dataclasses, and constants for model collaboration.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

# =========================================================================
# MODEL CONSTANTS
# =========================================================================

MODEL_QWEN3 = "Qwen3-8B"
MODEL_INDICTRANS2 = "IndicTrans2-1B"
MODEL_BGE_M3 = "BGE-M3"
MODEL_BGE_RERANKER = "BGE-Reranker"
MODEL_WHISPER = "Whisper-V3"
MODEL_MMS_TTS = "MMS-TTS"
MODEL_GOT_OCR2 = "GOT-OCR2"

# Legacy aliases for backward compatibility
MODEL_QWEN25 = MODEL_QWEN3
MODEL_GEMMA2 = MODEL_QWEN3  # Validation now uses main Qwen3-8B LLM

ALL_MODELS = [
    MODEL_QWEN3,
    MODEL_INDICTRANS2,
    MODEL_BGE_M3,
    MODEL_BGE_RERANKER,
    MODEL_WHISPER,
    MODEL_MMS_TTS,
    MODEL_GOT_OCR2,
]

# =========================================================================
# ENUMS
# =========================================================================

class CollaborationPattern(Enum):
    """Patterns for multi-model collaboration."""

    CHAIN = "chain"  # Sequential processing: Model A → Model B → Model C
    VERIFY = "verify"  # Generator + Validator: Model A generates, Model B verifies
    BACK_TRANSLATE = "back_translate"  # A→B→A verification (e.g., EN→HI→EN)
    ENSEMBLE = "ensemble"  # Multiple models vote/average
    DEBATE = "debate"  # Models discuss to reach consensus
    ITERATIVE = "iterative"  # Refine until quality threshold
    SEMANTIC_CHECK = "semantic_check"  # Embedding similarity validation
    AUDIO_VERIFY = "audio_verify"  # TTS → STT verification loop
    DOCUMENT_CHAIN = "document_chain"  # OCR → Simplify → Translate → Audio
    RERANK = "rerank"  # Generate candidates → Rerank → Select best

class ModelRole(StrEnum):
    """Roles models can play in collaboration."""

    GENERATOR = "generator"  # Creates content
    VALIDATOR = "validator"  # Checks quality
    TRANSLATOR = "translator"  # Converts languages
    SEMANTIC_JUDGE = "semantic"  # Measures meaning preservation
    REFINER = "refiner"  # Improves content
    ORCHESTRATOR = "orchestrator"  # Coordinates others

# =========================================================================
# DATACLASSES
# =========================================================================

@dataclass
class ModelMessage:
    """Message passed between models during collaboration."""

    from_model: str
    to_model: str
    content: str
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def with_context(self, **kwargs) -> "ModelMessage":
        """Add context to message."""
        new_context = {**self.context, **kwargs}
        return ModelMessage(
            from_model=self.from_model,
            to_model=self.to_model,
            content=self.content,
            context=new_context,
            metadata=self.metadata,
            timestamp=self.timestamp,
        )

@dataclass
class CollaborationResult:
    """Result from a model collaboration session."""

    pattern: CollaborationPattern
    final_output: str
    confidence: float  # 0-1 confidence score
    consensus: bool  # Did models agree?
    iterations: int
    participating_models: list[str]
    messages: list[ModelMessage]
    scores: dict[str, float]  # Scores from each validator
    processing_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class CollaborationConfig:
    """Configuration for model collaboration."""

    # Quality thresholds
    min_confidence: float = 0.8
    semantic_threshold: float = 0.85  # Embedding similarity threshold
    consensus_threshold: float = 0.7  # % of models that must agree

    # Iteration limits
    max_iterations: int = 3
    max_debate_rounds: int = 2

    # Timeouts (seconds)
    model_timeout: float = 30.0
    total_timeout: float = 120.0

    # Feature flags
    enable_back_translation: bool = True
    enable_semantic_verification: bool = True
    enable_ensemble_voting: bool = True
