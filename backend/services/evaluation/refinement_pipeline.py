"""
Semantic Accuracy Refinement Pipeline
======================================

Implements iterative refinement to achieve 8.2+ semantic accuracy.

Key improvements over single-pass:
1. Feedback loop - re-generate if score < target
2. Separate validator model bias mitigation
3. Adaptive weights based on task type
4. Maximum 3 refinement iterations
5. Detailed issue tracking for each iteration

Target: 8.2+ semantic accuracy on 10-point scale
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RefinementTask(str, Enum):
    """Types of content refinement tasks."""

    SIMPLIFICATION = "simplification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXPLANATION = "explanation"


@dataclass
class RefinementConfig:
    """Configuration for refinement pipeline.

    Note: With 5-phase M4 optimization + RAG + validation, scores of 9.0+
    are achievable. The target_score is a minimum quality gate - outputs
    achieving higher scores are kept as-is.
    """

    target_score: float = 9.0  # M4-optimized: aim for 9.0+ with RAG/validation
    min_acceptable: float = 7.5  # Raised from 7.0 for quality assurance
    max_iterations: int = 3

    # Task-specific weight adjustments
    task_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            RefinementTask.SIMPLIFICATION: {
                "factual_accuracy": 0.35,
                "educational_clarity": 0.35,  # High - main goal
                "semantic_preservation": 0.15,  # Lower - simplification changes text
                "completeness": 0.10,
                "cultural_appropriateness": 0.05,
            },
            RefinementTask.TRANSLATION: {
                "factual_accuracy": 0.30,
                "semantic_preservation": 0.35,  # High - preserve meaning
                "educational_clarity": 0.15,
                "completeness": 0.15,
                "cultural_appropriateness": 0.05,
            },
            RefinementTask.EXPLANATION: {
                "factual_accuracy": 0.40,  # Critical for explanations
                "educational_clarity": 0.30,
                "completeness": 0.20,
                "semantic_preservation": 0.05,
                "cultural_appropriateness": 0.05,
            },
        }
    )

    # Issue-specific refinement prompts
    refinement_prompts: dict[str, str] = field(
        default_factory=lambda: {
            "low_factual_accuracy": "Ensure all facts and information are accurate. Double-check numbers, dates, and scientific terms.",
            "low_educational_clarity": "Use simpler words and shorter sentences. Add examples where helpful.",
            "low_semantic_preservation": "Maintain the core meaning while simplifying. Don't remove key concepts.",
            "low_completeness": "Include all important points from the original. Don't skip key concepts.",
            "low_cultural_appropriateness": "Use examples and context relevant to Indian students.",
        }
    )


@dataclass
class IterationResult:
    """Result of a single refinement iteration."""

    iteration: int
    text: str
    score: float
    dimension_scores: dict[str, float]
    issues: list[str]
    refinement_applied: str
    time_ms: float


@dataclass
class RefinementResult:
    """Complete refinement pipeline result."""

    final_text: str
    final_score: float
    achieved_target: bool
    iterations_used: int
    iteration_history: list[IterationResult]
    total_time_ms: float
    improvement: float  # Score improvement from iteration 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": round(self.final_score, 2),
            "achieved_target": self.achieved_target,
            "iterations_used": self.iterations_used,
            "improvement": round(self.improvement, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "history": [
                {
                    "iteration": r.iteration,
                    "score": round(r.score, 2),
                    "issues": r.issues,
                }
                for r in self.iteration_history
            ],
        }


class SemanticRefinementPipeline:
    """
    Iterative refinement pipeline for semantic accuracy.

    Process:
    1. Generate initial output
    2. Evaluate against all dimensions
    3. If score < target and iterations < max:
       a. Identify lowest-scoring dimension
       b. Generate refinement prompt
       c. Re-generate with feedback
       d. Repeat evaluation
    4. Return best result
    """

    def __init__(self, config: RefinementConfig | None = None):
        self.config = config or RefinementConfig()
        self._inference_engine = None
        self._evaluator = None

    def _get_inference_engine(self):
        """Lazy load inference engine."""
        if self._inference_engine is None:
            from ..inference import get_inference_engine

            self._inference_engine = get_inference_engine()
        return self._inference_engine

    def _get_evaluator(self):
        """Lazy load semantic evaluator."""
        if self._evaluator is None:
            from .semantic_evaluator import EvaluationConfig, SemanticAccuracyEvaluator

            # Configure evaluator with task-aware settings
            eval_config = EvaluationConfig(
                target_score=self.config.target_score,
                min_acceptable_score=self.config.min_acceptable,
                use_llm_evaluation=True,
            )
            self._evaluator = SemanticAccuracyEvaluator(eval_config)
        return self._evaluator

    async def refine(
        self,
        original_text: str,
        initial_output: str,
        task: RefinementTask = RefinementTask.SIMPLIFICATION,
        grade_level: int = 8,
        subject: str = "General",
    ) -> RefinementResult:
        """
        Refine content to achieve target semantic accuracy.

        Args:
            original_text: Original source text
            initial_output: First-pass generated output
            task: Type of task (affects weight distribution)
            grade_level: Target grade level
            subject: Subject area

        Returns:
            RefinementResult with best output and history
        """
        start_time = time.perf_counter()

        evaluator = self._get_evaluator()

        # Update evaluator weights based on task
        if task in self.config.task_weights:
            evaluator.config.weights = self.config.task_weights[task]

        iterations: list[IterationResult] = []
        current_text = initial_output
        best_text = initial_output
        best_score = 0.0

        for i in range(self.config.max_iterations):
            iter_start = time.perf_counter()

            # Evaluate current output
            eval_result = await evaluator.evaluate(
                original_text=original_text,
                processed_text=current_text,
                grade_level=grade_level,
                subject=subject,
            )

            score = eval_result.overall_score
            dimension_scores = {
                k: v.score for k, v in eval_result.dimension_scores.items()
            }

            # Track best result
            if score > best_score:
                best_score = score
                best_text = current_text

            # Identify issues (dimensions below 8.0)
            issues = [f"low_{dim}" for dim, s in dimension_scores.items() if s < 8.0]

            iter_time = (time.perf_counter() - iter_start) * 1000

            # Determine refinement applied text
            refinement_text = ""
            if i > 0:
                issue_text = issues[0] if issues else "general"
                refinement_text = f"Addressed: {issue_text}"

            # Record iteration
            iterations.append(
                IterationResult(
                    iteration=i + 1,
                    text=current_text,
                    score=score,
                    dimension_scores=dimension_scores,
                    issues=issues,
                    refinement_applied=refinement_text,
                    time_ms=iter_time,
                )
            )

            # Check if target achieved
            if score >= self.config.target_score:
                logger.info(f"Target achieved at iteration {i + 1}: {score:.2f}")
                break

            # If more iterations available, refine
            if i < self.config.max_iterations - 1 and issues:
                # Find lowest scoring dimension
                lowest_dim = min(dimension_scores, key=dimension_scores.get)
                lowest_issue = f"low_{lowest_dim}"

                # Generate refinement prompt
                refinement_prompt = self._build_refinement_prompt(
                    original_text=original_text,
                    current_output=current_text,
                    issue=lowest_issue,
                    score=score,
                    dimension_scores=dimension_scores,
                    task=task,
                    grade_level=grade_level,
                )

                # Re-generate with feedback
                current_text = await self._refine_output(refinement_prompt)

                logger.info(
                    f"Iteration {i + 1}: score={score:.2f}, refining for {lowest_issue}"
                )

        total_time = (time.perf_counter() - start_time) * 1000

        return RefinementResult(
            final_text=best_text,
            final_score=best_score,
            achieved_target=best_score >= self.config.target_score,
            iterations_used=len(iterations),
            iteration_history=iterations,
            total_time_ms=total_time,
            improvement=best_score - iterations[0].score if iterations else 0,
        )

    def _build_refinement_prompt(
        self,
        original_text: str,
        current_output: str,
        issue: str,
        score: float,
        dimension_scores: dict[str, float],
        task: RefinementTask,
        grade_level: int,
    ) -> str:
        """Build prompt for refinement iteration."""

        # Get issue-specific guidance
        guidance = self.config.refinement_prompts.get(
            issue, "Improve overall quality and accuracy."
        )

        # Format dimension feedback
        dim_feedback = "\n".join(
            [f"- {dim}: {s:.1f}/10" for dim, s in dimension_scores.items()]
        )

        prompt = f"""You are refining educational content for Grade {grade_level} Indian students.

TASK: {task.value}

ORIGINAL TEXT:
{original_text[:1500]}

CURRENT OUTPUT (Score: {score:.1f}/10):
{current_output[:1500]}

EVALUATION SCORES:
{dim_feedback}

ISSUE TO ADDRESS: {issue}
GUIDANCE: {guidance}

Please provide an improved version that addresses the identified issue while maintaining quality in other dimensions.

IMPROVED OUTPUT:"""

        return prompt

    async def _refine_output(self, prompt: str) -> str:
        """Generate refined output."""
        try:
            engine = self._get_inference_engine()

            from ..inference.unified_engine import GenerationConfig

            config = GenerationConfig(
                max_tokens=4096,  # Increased for complete refined output
                temperature=0.3,  # Slightly creative for refinement
            )

            response = await engine.generate(prompt, config)

            # Clean up response
            response = response.strip()
            if response.startswith("IMPROVED OUTPUT:"):
                response = response[16:].strip()

            return response

        except Exception as e:
            logger.error(f"Refinement generation failed: {e}")
            raise


# ==================== Convenience Functions ====================


async def refine_for_accuracy(
    original_text: str,
    processed_text: str,
    task: str = "simplification",
    grade_level: int = 8,
    target_score: float = 8.2,
) -> RefinementResult:
    """
    Convenience function to refine content for semantic accuracy.

    Usage:
        result = await refine_for_accuracy(
            original_text="Complex physics explanation...",
            processed_text="Simplified version...",
            task="simplification",
            grade_level=8,
            target_score=8.2,
        )

        if result.achieved_target:
            use(result.final_text)
        else:
            review(result.iteration_history)
    """
    config = RefinementConfig(target_score=target_score)
    pipeline = SemanticRefinementPipeline(config)

    task_enum = (
        RefinementTask(task)
        if task in [t.value for t in RefinementTask]
        else RefinementTask.SIMPLIFICATION
    )

    return await pipeline.refine(
        original_text=original_text,
        initial_output=processed_text,
        task=task_enum,
        grade_level=grade_level,
    )


def get_refinement_pipeline(
    target_score: float = 8.2,
    max_iterations: int = 3,
) -> SemanticRefinementPipeline:
    """Get configured refinement pipeline."""
    config = RefinementConfig(
        target_score=target_score,
        max_iterations=max_iterations,
    )
    return SemanticRefinementPipeline(config)
