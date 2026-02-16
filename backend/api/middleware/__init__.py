"""
Middleware & Orchestration Layer
================================

High-performance orchestration coordinating models, hardware, and memory.

Modules:
- classifier:   Intent classification & request complexity scoring
- pipeline:      Agent pipeline with strict I/O contracts
- memory:        Short-term / long-term / retrieval memory manager
- latency:       Latency control, early-exit, confidence-based stopping
- evaluator:     Self-evaluation loop with routing heuristic adjustment
- orchestrator:  Unified facade wiring everything together
"""

from .classifier import ClassifiedRequest, ComplexityLevel, RequestClassifier
from .evaluator import EvaluationReport, SelfEvaluator
from .latency import LatencyBudget, LatencyController
from .memory import MemoryManager, MemoryTier
from .orchestrator import MiddlewareOrchestrator
from .pipeline import AgentPipeline, PipelineResult, PipelineStage

__all__ = [
    "AgentPipeline",
    "ClassifiedRequest",
    "ComplexityLevel",
    "EvaluationReport",
    "LatencyBudget",
    "LatencyController",
    "MemoryManager",
    "MemoryTier",
    "MiddlewareOrchestrator",
    "PipelineResult",
    "PipelineStage",
    "RequestClassifier",
    "SelfEvaluator",
]
