"""
Model Collaboration System - Inter-Model Communication & Orchestration
=======================================================================

v2.1.0 - Refactored into modular package (December 2025)

This module provides backward compatibility for imports.
The actual implementations are now in separate modules within this package.

MODELS PARTICIPATING (ALL 7):
- Qwen3-8B (Orchestrator + Validator): Main LLM, coordinates, generates, and validates
- IndicTrans2 (Translator): Translation with back-translation verification
- BGE-M3 (Semantic Judge): Embedding similarity for semantic preservation
- BGE-Reranker (Quality Ranker): Ranks multiple outputs to select best
- MMS-TTS (Audio Generator): Text-to-Speech for audio content
- Whisper (Audio Verifier): STT to verify TTS output accuracy
- GOT-OCR2 (Document Reader): OCR for images and documents
"""

# Import from types module - the canonical source
# Import from orchestrator - main class
from .orchestrator import (
    ModelCollaborator,
)
from .types import (
    ALL_MODELS,
    MODEL_BGE_M3,
    MODEL_BGE_RERANKER,
    MODEL_GEMMA2,
    MODEL_GOT_OCR2,
    MODEL_INDICTRANS2,
    MODEL_MMS_TTS,
    MODEL_QWEN25,
    MODEL_QWEN3,
    MODEL_WHISPER,
    CollaborationConfig,
    CollaborationPattern,
    CollaborationResult,
    ModelMessage,
    ModelRole,
)


# Singleton getter
def get_model_collaborator() -> ModelCollaborator:
    """Get singleton ModelCollaborator instance."""
    from .convenience import get_model_collaborator as _get

    return _get()


# Import convenience functions
from .convenience import (
    collaborate_and_simplify,
    collaborate_and_translate,
    ensemble_evaluate,
    full_educational_pipeline,
    generate_best_output,
    process_document,
    verify_audio_output,
)

__all__ = [
    "ALL_MODELS",
    "MODEL_BGE_M3",
    "MODEL_BGE_RERANKER",
    "MODEL_GEMMA2",
    "MODEL_GOT_OCR2",
    "MODEL_INDICTRANS2",
    "MODEL_MMS_TTS",
    "MODEL_QWEN25",
    "MODEL_QWEN3",
    "MODEL_WHISPER",
    "CollaborationConfig",
    "CollaborationPattern",
    "CollaborationResult",
    "ModelCollaborator",
    "ModelMessage",
    "ModelRole",
    "collaborate_and_simplify",
    "collaborate_and_translate",
    "ensemble_evaluate",
    "full_educational_pipeline",
    "generate_best_output",
    "get_model_collaborator",
    "process_document",
    "verify_audio_output",
]
