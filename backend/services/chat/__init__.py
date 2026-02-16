"""
AI Core Module - Production ChatGPT/Gemini-quality Local Assistant
===================================================================

This module implements the central AI orchestration layer with:
- Intelligent Intent: Dynamic intent classification without hardcoded patterns
- Formatting & UX: Consistent, deterministic output with headings, steps, code fences
- Context & Memory: Session context management with summarization fallback
- Retrieval & Docs: Vector DB + retrieval + reranking with citations
- Model Routing: Configurable LLM routing (local, strong, lightweight)
- Tools & Sandboxing: Safe, auditable code execution
- Developer Features: Codebase analysis, PR diffs, test generation
- Reliability & Security: Hallucination detection, secret redaction
- Observability: Metrics, structured logs, graceful degradation

Version: 1.1.0
"""

from .citations import Citation, CitationManager, SourceType
from .context import ContextManager, ConversationContext
from .engine import AIEngine, get_ai_engine
from .explainer import ExplainabilityReport, Explainer
from .exports import ExportFormat, ExportService
from .formatter import OutputFormat, ResponseFormatter
from .intent import (
    IntelligentIntentClassifier,
    Intent,
    IntentAnalysis,
    analyze_intent,
    get_intent_classifier,
)
from .prompts import PromptRegistry, PromptVersion
from .router import ModelRouter, ModelTier
from .safety import SafetyGuard, SecretScanner
from .sandbox import ToolSandbox, calculate, execute_code, get_sandbox

__all__ = [
    # Core Engine
    "AIEngine",
    "Citation",
    # Citations
    "CitationManager",
    "ContextManager",
    # Context Management
    "ConversationContext",
    "ExplainabilityReport",
    # Explainability
    "Explainer",
    "ExportFormat",
    # Exports
    "ExportService",
    # Intent Classification
    "IntelligentIntentClassifier",
    "Intent",
    "IntentAnalysis",
    # Model Routing
    "ModelRouter",
    "ModelTier",
    "OutputFormat",
    # Prompts
    "PromptRegistry",
    "PromptVersion",
    # Formatting
    "ResponseFormatter",
    # Safety
    "SafetyGuard",
    "SecretScanner",
    "SourceType",
    # Sandbox
    "ToolSandbox",
    "analyze_intent",
    "calculate",
    "execute_code",
    "get_ai_engine",
    "get_intent_classifier",
    "get_sandbox",
]
