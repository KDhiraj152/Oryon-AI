"""
Canonical Type Definitions
===========================

Single source of truth for shared types used across the codebase.
All modules should import shared enums and base types from here.

This module defines:
- ModelTier: Model capability tiers for routing
- TaskType: Task types for model routing
- ModelType: Hardware-level model categories
- ModelRole: Role a model plays in collaboration

Other type definitions (dataclasses, configs) remain in their
domain modules but use these shared enums.
"""

from enum import Enum, StrEnum

__all__ = [
    "ModelTier",
    "ModelType",
    "TaskType",
]

class ModelTier(StrEnum):
    """Model capability tiers for routing decisions.

    Used by the model router to match request complexity
    to appropriate model capability levels.
    """

    LIGHTWEIGHT = "lightweight"  # Fast, for simple tasks
    STANDARD = "standard"  # Balanced, for general chat
    STRONG = "strong"  # Powerful, for complex reasoning
    SPECIALIZED = "specialized"  # Domain-specific models

class TaskType(StrEnum):
    """Task types for model routing and scheduling.

    Determines which model and configuration to use for a request.
    """

    CHAT = "chat"
    REASONING = "reasoning"
    CODE = "code"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    VALIDATION = "validation"
    SYSTEM = "system"

class ModelType(StrEnum):
    """Hardware-level model categories for the model manager.

    Used by the hardware optimization layer to manage model
    loading, memory allocation, and device placement.
    """

    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    TTS = "tts"
    STT = "stt"
    TRANSLATION = "translation"
    OCR = "ocr"
