"""
Evaluation Module
=================

Semantic accuracy evaluation and refinement for AI-generated content.

Components:
- semantic_evaluator: Multi-dimensional content quality scoring
- refinement_pipeline: Iterative content improvement pipeline

Usage:
    from backend.ml.evaluation import SemanticAccuracyEvaluator, SemanticRefinementPipeline
"""

from .refinement_pipeline import (
    IterationResult,
    RefinementConfig,
    RefinementResult,
    RefinementTask,
    SemanticRefinementPipeline,
)
from .semantic_evaluator import (
    DimensionScore,
    EvaluationConfig,
    EvaluationDimension,
    EvaluationResult,
    SemanticAccuracyEvaluator,
)


def get_semantic_evaluator():
    """Get a semantic evaluator instance."""
    return SemanticAccuracyEvaluator()

def get_refinement_pipeline(config: RefinementConfig = None):
    """Get a refinement pipeline instance."""
    return SemanticRefinementPipeline(config or RefinementConfig())

__all__ = [
    "DimensionScore",
    "EvaluationConfig",
    "EvaluationDimension",
    "EvaluationResult",
    "IterationResult",
    "RefinementConfig",
    "RefinementResult",
    "RefinementTask",
    "SemanticAccuracyEvaluator",
    "SemanticRefinementPipeline",
    "get_refinement_pipeline",
    "get_semantic_evaluator",
]
