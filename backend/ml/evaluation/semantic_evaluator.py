"""
Semantic Accuracy Evaluator
============================

Evaluates the semantic accuracy of AI-generated content.
Target: 8.2+ on a 10-point scale.

Evaluation Criteria:
1. Factual Accuracy - Content is correct
2. Semantic Preservation - Original meaning preserved
3. Educational Clarity - Easy to understand at target level
4. Cultural Appropriateness - Suitable for Indian context
5. Completeness - All key concepts covered

Uses embedding similarity + LLM-based evaluation for robust scoring.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

class EvaluationDimension(StrEnum):
    """Dimensions of content quality evaluation."""

    FACTUAL_ACCURACY = "factual_accuracy"
    SEMANTIC_PRESERVATION = "semantic_preservation"
    CONTENT_CLARITY = "content_clarity"
    CULTURAL_APPROPRIATENESS = "cultural_appropriateness"
    COMPLETENESS = "completeness"

@dataclass
class EvaluationConfig:
    """Configuration for semantic evaluation.

    M4-Optimized Targets (2025):
    - min_acceptable: 7.5 (quality gate)
    - target: 9.0 (achievable with RAG + validation)
    - excellent: 9.5 (peak performance)
    """

    # Weights for each dimension (must sum to 1.0)
    weights: dict[str, float] = field(
        default_factory=lambda: {
            EvaluationDimension.FACTUAL_ACCURACY: 0.30,
            EvaluationDimension.SEMANTIC_PRESERVATION: 0.25,
            EvaluationDimension.CONTENT_CLARITY: 0.20,
            EvaluationDimension.CULTURAL_APPROPRIATENESS: 0.10,
            EvaluationDimension.COMPLETENESS: 0.15,
        }
    )

    # Thresholds - M4 optimized targets
    min_acceptable_score: float = 7.5  # Raised from 7.0
    target_score: float = 9.0  # Raised from 8.2 (achievable with optimizations)
    excellent_score: float = 9.5  # Peak performance target

    # Embedding similarity threshold for semantic preservation
    similarity_threshold: float = 0.80  # Raised from 0.75 for stricter matching

    # Use LLM for detailed evaluation
    use_llm_evaluation: bool = True

    # Cache evaluations
    cache_results: bool = True

@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    dimension: str
    score: float  # 0-10 scale
    confidence: float  # 0-1
    feedback: str
    details: dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    overall_score: float  # 0-10 scale
    dimension_scores: dict[str, DimensionScore]

    # Quick access
    passes_minimum: bool = False
    meets_target: bool = False
    is_excellent: bool = False

    # Details
    feedback: str = ""
    suggestions: list[str] = field(default_factory=list)

    # Performance
    evaluation_time_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 2),
            "passes_minimum": self.passes_minimum,
            "meets_target": self.meets_target,
            "is_excellent": self.is_excellent,
            "feedback": self.feedback,
            "suggestions": self.suggestions,
            "dimension_scores": {
                k: {
                    "score": round(v.score, 2),
                    "confidence": round(v.confidence, 2),
                    "feedback": v.feedback,
                }
                for k, v in self.dimension_scores.items()
            },
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
        }

class SemanticAccuracyEvaluator:
    """
    Evaluates semantic accuracy of AI-generated content.

    Uses a combination of:
    1. Embedding similarity for semantic preservation
    2. LLM-based evaluation for AIal quality
    3. Rule-based checks for cultural sensitivity

    Target: 8.2+ overall score
    """

    # LLM evaluation prompt
    EVALUATION_PROMPT = """You are an expert content evaluator for users.

Evaluate this AI-generated content for a Grade {complexity_level} {subject} student.

ORIGINAL TEXT:
{original_text}

SIMPLIFIED/PROCESSED TEXT:
{processed_text}

Rate each dimension from 0-10:

1. FACTUAL_ACCURACY (Is the content factually correct?):
   Score: [0-10]

2. SEMANTIC_PRESERVATION (Does it preserve the original meaning?):
   Score: [0-10]

3. CONTENT_CLARITY (Is it clear and easy to understand?):
   Score: [0-10]

4. CULTURAL_APPROPRIATENESS (Is it suitable for users?):
   Score: [0-10]

5. COMPLETENESS (Are all key concepts covered?):
   Score: [0-10]

Provide your response in this EXACT format:
FACTUAL_ACCURACY: [score]
SEMANTIC_PRESERVATION: [score]
CONTENT_CLARITY: [score]
CULTURAL_APPROPRIATENESS: [score]
COMPLETENESS: [score]
FEEDBACK: [one sentence overall feedback]
SUGGESTIONS: [comma-separated improvement suggestions if score < 8]"""

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

        # Lazy-loaded components
        self._inference_engine = None

    def _get_inference_engine(self):
        """Get inference engine (lazy loaded)."""
        if self._inference_engine is None:
            from ..inference import get_inference_engine

            self._inference_engine = get_inference_engine(auto_load=False)
        return self._inference_engine

    async def evaluate(
        self,
        original_text: str,
        processed_text: str,
        complexity_level: int = 8,
        subject: str = "General",
    ) -> EvaluationResult:
        """
        Evaluate processed content against original.

        Args:
            original_text: Original input text
            processed_text: AI-processed output
            complexity_level: Target complexity level (1-12)
            subject: Subject area

        Returns:
            EvaluationResult with scores and feedback
        """
        start = time.perf_counter()

        dimension_scores: dict[str, DimensionScore] = {}

        # 1. Evaluate semantic preservation using embeddings
        semantic_score = await self._evaluate_semantic_similarity(
            original_text, processed_text
        )
        dimension_scores[EvaluationDimension.SEMANTIC_PRESERVATION.value] = semantic_score

        # 2. LLM-based evaluation for other dimensions
        if self.config.use_llm_evaluation:
            llm_scores = await self._evaluate_with_llm(
                original_text, processed_text, complexity_level, subject
            )
            dimension_scores.update(llm_scores)
        else:
            # Fallback to heuristic evaluation
            heuristic_scores = self._evaluate_heuristic(
                original_text, processed_text, complexity_level
            )
            dimension_scores.update(heuristic_scores)

        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)

        # Generate feedback and suggestions
        feedback, suggestions = self._generate_feedback(dimension_scores, overall_score)

        elapsed = (time.perf_counter() - start) * 1000

        return EvaluationResult(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            passes_minimum=overall_score >= self.config.min_acceptable_score,
            meets_target=overall_score >= self.config.target_score,
            is_excellent=overall_score >= self.config.excellent_score,
            feedback=feedback,
            suggestions=suggestions,
            evaluation_time_ms=elapsed,
        )

    async def _evaluate_semantic_similarity(
        self,
        original: str,
        processed: str,
    ) -> DimensionScore:
        """Evaluate semantic preservation using embedding similarity."""
        try:
            engine = self._get_inference_engine()

            # Get embeddings
            embeddings = await engine.embed([original, processed])
            orig_emb = embeddings[0]
            proc_emb = embeddings[1]

            # SIMD-optimized cosine similarity
            try:
                from backend.core.optimized.simd_ops import cosine_similarity_single

                similarity = cosine_similarity_single(orig_emb, proc_emb)
            except ImportError:
                similarity = np.dot(orig_emb, proc_emb) / (
                    np.linalg.norm(orig_emb) * np.linalg.norm(proc_emb)
                )

            # Map similarity to 0-10 scale
            # 0.5 similarity -> 5, 1.0 similarity -> 10
            score = min(10.0, max(0.0, (similarity - 0.5) * 10))

            # Calculate confidence based on text length
            confidence = min(1.0, (len(original) + len(processed)) / 1000)

            if similarity >= 0.90:
                feedback = "Excellent semantic preservation"
            elif similarity >= 0.80:
                feedback = "Good semantic preservation with minor variations"
            elif similarity >= 0.70:
                feedback = "Adequate semantic preservation, some meaning may be lost"
            else:
                feedback = "Significant semantic drift detected"

            return DimensionScore(
                dimension=EvaluationDimension.SEMANTIC_PRESERVATION,
                score=score,
                confidence=confidence,
                feedback=feedback,
                details={"similarity": float(similarity)},
            )

        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Embedding evaluation failed: %s", e)
            return DimensionScore(
                dimension=EvaluationDimension.SEMANTIC_PRESERVATION,
                score=7.0,  # Default neutral score
                confidence=0.3,
                feedback="Could not compute embedding similarity",
            )

    async def _evaluate_with_llm(
        self,
        original: str,
        processed: str,
        complexity_level: int,
        subject: str,
    ) -> dict[str, DimensionScore]:
        """Use LLM for detailed evaluation."""
        try:
            engine = self._get_inference_engine()

            prompt = self.EVALUATION_PROMPT.format(
                complexity_level=complexity_level,
                subject=subject,
                original_text=original[:3000],  # Increased context window
                processed_text=processed[:3000],
            )

            from ..inference.unified_engine import GenerationConfig

            config = GenerationConfig(
                max_tokens=1024,  # Increased for detailed feedback
                temperature=0.1,  # Low temperature for consistent evaluation
            )

            response = await engine.generate(prompt, config)

            # Parse scores from response
            scores = self._parse_llm_response(response)
            return scores

        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("LLM evaluation failed: %s", e)
            return {}

    def _parse_llm_response(self, response: str) -> dict[str, DimensionScore]:
        """Parse LLM evaluation response."""
        scores = {}

        # Extract scores using regex
        patterns = {
            EvaluationDimension.FACTUAL_ACCURACY: r"FACTUAL_ACCURACY:\s*(\d+(?:\.\d+)?)",
            EvaluationDimension.CONTENT_CLARITY: r"CONTENT_CLARITY:\s*(\d+(?:\.\d+)?)",
            EvaluationDimension.CULTURAL_APPROPRIATENESS: r"CULTURAL_APPROPRIATENESS:\s*(\d+(?:\.\d+)?)",
            EvaluationDimension.COMPLETENESS: r"COMPLETENESS:\s*(\d+(?:\.\d+)?)",
        }

        for dimension, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                score = min(10.0, max(0.0, score))

                scores[dimension.value] = DimensionScore(
                    dimension=dimension,
                    score=score,
                    confidence=0.8,
                    feedback=f"LLM evaluation: {score}/10",
                )

        # Extract feedback
        feedback_match = re.search(
            r"FEEDBACK:\s*(.+)(?:SUGGESTIONS:|$)", response, re.IGNORECASE | re.DOTALL
        )
        if feedback_match:
            feedback = feedback_match.group(1).strip()
            for dim_score in scores.values():
                dim_score.feedback = feedback

        return scores

    def _evaluate_heuristic(
        self,
        original: str,
        processed: str,
        complexity_level: int,
    ) -> dict[str, DimensionScore]:
        """Heuristic evaluation (fallback when LLM unavailable)."""
        scores = {}

        # Factual accuracy: Assume preserved unless text is drastically shorter
        length_ratio = len(processed) / max(1, len(original))
        factual_score = 8.0 if 0.3 <= length_ratio <= 1.5 else 6.0

        scores[EvaluationDimension.FACTUAL_ACCURACY.value] = DimensionScore(
            dimension=EvaluationDimension.FACTUAL_ACCURACY,
            score=factual_score,
            confidence=0.5,
            feedback="Heuristic evaluation based on length preservation",
        )

        # Educational clarity: Based on sentence length
        avg_sentence_len = len(processed.split()) / max(1, processed.count(".") + 1)
        target_len = 8 + complexity_level  # Longer sentences for higher grades

        if abs(avg_sentence_len - target_len) <= 5:
            clarity_score = 8.5
        elif abs(avg_sentence_len - target_len) <= 10:
            clarity_score = 7.0
        else:
            clarity_score = 6.0

        scores[EvaluationDimension.CONTENT_CLARITY.value] = DimensionScore(
            dimension=EvaluationDimension.CONTENT_CLARITY,
            score=clarity_score,
            confidence=0.4,
            feedback=f"Average sentence length: {avg_sentence_len:.1f} words",
        )

        # Cultural appropriateness: Default to good
        scores[EvaluationDimension.CULTURAL_APPROPRIATENESS.value] = DimensionScore(
            dimension=EvaluationDimension.CULTURAL_APPROPRIATENESS,
            score=8.0,
            confidence=0.3,
            feedback="Default cultural appropriateness score",
        )

        # Completeness: Based on key term preservation
        original_words = set(original.lower().split())
        processed_words = set(processed.lower().split())
        overlap = len(original_words & processed_words) / max(1, len(original_words))
        completeness_score = 5.0 + (overlap * 5.0)

        scores[EvaluationDimension.COMPLETENESS.value] = DimensionScore(
            dimension=EvaluationDimension.COMPLETENESS,
            score=completeness_score,
            confidence=0.5,
            feedback=f"Term overlap: {overlap:.0%}",
        )

        return scores

    def _calculate_overall_score(
        self,
        dimension_scores: dict[str, DimensionScore],
    ) -> float:
        """Calculate weighted overall score."""
        total_weight = 0.0
        weighted_sum = 0.0

        for dimension, weight in self.config.weights.items():
            if dimension in dimension_scores:
                score = dimension_scores[dimension].score
                confidence = dimension_scores[dimension].confidence

                # Weight by both dimension weight and confidence
                effective_weight = weight * confidence
                weighted_sum += score * effective_weight
                total_weight += effective_weight

        if total_weight == 0:
            return 7.0  # Default neutral score

        return weighted_sum / total_weight

    def _generate_feedback(
        self,
        dimension_scores: dict[str, DimensionScore],
        overall_score: float,
    ) -> tuple[str, list[str]]:
        """Generate feedback and improvement suggestions."""
        suggestions = []

        # Generate feedback based on overall score
        if overall_score >= 9.0:
            feedback = "Excellent content quality. Ready for use."
        elif overall_score >= 8.2:
            feedback = "Good content quality. Meets target standards."
        elif overall_score >= 7.0:
            feedback = "Acceptable quality but has room for improvement."
        else:
            feedback = "Content needs improvement before use."

        # Generate suggestions for low scores
        for dimension, score_obj in dimension_scores.items():
            if score_obj.score < 7.5:
                if dimension == EvaluationDimension.FACTUAL_ACCURACY:
                    suggestions.append("Review factual content for accuracy")
                elif dimension == EvaluationDimension.SEMANTIC_PRESERVATION:
                    suggestions.append("Check if key meaning is preserved")
                elif dimension == EvaluationDimension.CONTENT_CLARITY:
                    suggestions.append("Simplify complex sentences")
                elif dimension == EvaluationDimension.CULTURAL_APPROPRIATENESS:
                    suggestions.append("Add culturally relevant examples")
                elif dimension == EvaluationDimension.COMPLETENESS:
                    suggestions.append("Ensure all key concepts are covered")

        return feedback, suggestions

    def get_config(self) -> dict[str, Any]:
        """Get evaluator configuration."""
        return {
            "weights": self.config.weights,
            "min_acceptable_score": self.config.min_acceptable_score,
            "target_score": self.config.target_score,
            "excellent_score": self.config.excellent_score,
            "use_llm_evaluation": self.config.use_llm_evaluation,
        }

# ==================== Global Instance ====================

_evaluator: SemanticAccuracyEvaluator | None = None

def get_semantic_evaluator(
    config: EvaluationConfig | None = None,
) -> SemanticAccuracyEvaluator:
    """Get global semantic evaluator."""
    global _evaluator

    if _evaluator is None:
        _evaluator = SemanticAccuracyEvaluator(config)

    return _evaluator
