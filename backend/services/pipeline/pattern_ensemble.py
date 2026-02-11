"""
Ensemble Collaboration Patterns
===============================

Contains ensemble and iterative patterns:
- Ensemble: Multiple models vote/average for evaluation
- Iterative: Refine until quality threshold
- Debate: Models discuss to reach consensus (simplified)
"""

import asyncio
import logging
from typing import Any, Dict

from ..types import (
    CollaborationPattern,
    CollaborationResult,
)

logger = logging.getLogger(__name__)


class EnsemblePatternsMixin:
    """
    Mixin providing ensemble collaboration patterns.

    Requires ModelAccessorsMixin and CollaborationHelpersMixin to be mixed in.
    """

    def _ensemble_collect_scores(
        self,
        results: list[Any],
        models_used: list[str],
        scores: dict[str, float],
    ) -> None:
        """Collect valid scores from parallel evaluation results."""
        score_mapping = [
            ("llm_score", "qwen3-8b", None),
            ("semantic_score", "bge-m3", "semantic_checks"),
            ("validator_score", "qwen3-8b", None),
        ]
        for idx, (key, model, metric_key) in enumerate(score_mapping):
            result = results[idx]
            if isinstance(result, Exception) or result is None:
                continue
            scores[key] = result
            models_used.append(model)
            if metric_key:
                self._metrics[metric_key] = self._metrics.get(metric_key, 0) + 1  # type: ignore

    @staticmethod
    def _ensemble_weighted_average(
        scores: dict[str, float],
        base_weights: dict[str, float],
    ) -> float:
        """Compute weighted average confidence from ensemble scores."""
        total_weight = 0.0
        weighted_sum = 0.0
        for key, score in scores.items():
            weight = base_weights.get(key, 0.33)
            weighted_sum += score * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    async def _ensemble_collaboration(
        self,
        _task: str,  # Reserved for task-specific ensemble strategies
        input_text: str,
        context: dict[str, Any],
    ) -> CollaborationResult:
        """
        Ensemble pattern: Multiple models evaluate, consensus wins.

        Each model (LLM, Embedder, Validator) provides a score.
        Final decision based on weighted voting.
        """
        models_used: list[str] = []
        scores: dict[str, float] = {}

        grade_level = context.get("grade_level", 8)
        subject = context.get("subject", "General")
        processed_text = context.get("processed_text", input_text)

        base_weights = {"llm_score": 0.3, "semantic_score": 0.4, "validator_score": 0.3}

        # Task 1: LLM evaluates
        async def llm_evaluate() -> float | None:
            llm = self._get_llm()  # type: ignore
            if not llm:
                return None
            prompt = f"""Rate the following educational content on a scale of 0-10.

Original:
{input_text}

Processed (for Grade {grade_level} {subject}):
{processed_text}

Consider: accuracy, clarity, age-appropriateness, completeness.

Respond with just the score (0-10):"""
            response = await llm.generate_async(prompt, max_tokens=10)
            try:
                score = float(response.strip().split()[0])
                return min(max(score, 0), 10) / 10.0
            except (ValueError, IndexError):
                return 0.7

        # Task 2: Semantic similarity
        async def semantic_evaluate() -> float | None:
            embedder = self._get_embedder()  # type: ignore
            if not embedder:
                return None
            orig_emb = await self._get_embedding(input_text)  # type: ignore
            proc_emb = await self._get_embedding(processed_text)  # type: ignore
            if orig_emb is not None and proc_emb is not None:
                return self._cosine_similarity(orig_emb, proc_emb)  # type: ignore
            return None

        # Task 3: Validator evaluates
        async def validator_evaluate() -> float | None:
            validator = self._get_validator()  # type: ignore
            if not validator:
                return None
            result = await validator.evaluate(
                original_text=input_text, processed_text=processed_text,
                grade_level=grade_level, subject=subject,
            )
            return float(result.overall_score / 10.0)

        # Run all evaluations in parallel
        results = await asyncio.gather(
            llm_evaluate(), semantic_evaluate(), validator_evaluate(),
            return_exceptions=True,
        )

        self._ensemble_collect_scores(list(results), models_used, scores)
        confidence = self._ensemble_weighted_average(scores, base_weights)

        consensus = confidence >= self.config.consensus_threshold  # type: ignore
        if consensus:
            self._metrics["successful_consensus"] = (  # type: ignore
                self._metrics.get("successful_consensus", 0) + 1  # type: ignore
            )

        return CollaborationResult(
            pattern=CollaborationPattern.ENSEMBLE,
            final_output=processed_text,
            confidence=confidence,
            consensus=consensus,
            iterations=1,
            participating_models=models_used,
            messages=[],
            scores=scores,
            processing_time_ms=0,
        )

    async def _iterative_collaboration(
        self,
        task: str,
        input_text: str,
        context: dict[str, Any],
    ) -> CollaborationResult:
        """
        Iterative pattern: Models take turns improving output.

        LLM generates, Validator scores, LLM refines, repeat.
        Each iteration builds on the previous.
        """
        # This delegates to verify pattern with iteration logic
        return await self._verify_collaboration(task, input_text, context)  # type: ignore

    async def _debate_collaboration(
        self,
        task: str,
        input_text: str,
        context: dict[str, Any],
    ) -> CollaborationResult:
        """
        Debate pattern: Models discuss to reach consensus.

        Currently simplified to ensemble evaluation.
        Future: implement actual multi-model debate.
        """
        # Delegate to ensemble for now
        return await self._ensemble_collaboration(task, input_text, context)
