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
        models_used = []
        scores = {}

        grade_level = context.get("grade_level", 8)
        subject = context.get("subject", "General")
        processed_text = context.get("processed_text", input_text)

        # Weight for each model's vote
        weights = {
            "llm_score": 0.3,
            "semantic_score": 0.4,
            "validator_score": 0.3,
        }

        # Task 1: LLM evaluates
        async def llm_evaluate():
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
                return min(max(score, 0), 10) / 10.0  # Normalize to 0-1
            except (ValueError, IndexError):
                return 0.7

        # Task 2: Semantic similarity
        async def semantic_evaluate():
            embedder = self._get_embedder()  # type: ignore
            if not embedder:
                return None

            orig_emb = await self._get_embedding(input_text)  # type: ignore
            proc_emb = await self._get_embedding(processed_text)  # type: ignore

            if orig_emb is not None and proc_emb is not None:
                return self._cosine_similarity(orig_emb, proc_emb)  # type: ignore
            return None

        # Task 3: Validator evaluates
        async def validator_evaluate():
            validator = self._get_validator()  # type: ignore
            if not validator:
                return None

            result = await validator.evaluate(
                original_text=input_text,
                processed_text=processed_text,
                grade_level=grade_level,
                subject=subject,
            )
            return result.overall_score / 10.0  # Normalize to 0-1

        # Run all evaluations in parallel
        results = await asyncio.gather(
            llm_evaluate(),
            semantic_evaluate(),
            validator_evaluate(),
            return_exceptions=True,
        )

        # Collect valid scores
        llm_score = (
            results[0]
            if not isinstance(results[0], Exception) and results[0] is not None
            else None
        )
        semantic_score = (
            results[1]
            if not isinstance(results[1], Exception) and results[1] is not None
            else None
        )
        validator_score = (
            results[2]
            if not isinstance(results[2], Exception) and results[2] is not None
            else None
        )

        if llm_score is not None:
            scores["llm_score"] = llm_score
            models_used.append("qwen-3b")

        if semantic_score is not None:
            scores["semantic_score"] = semantic_score
            models_used.append("bge-m3")
            self._metrics["semantic_checks"] = (
                self._metrics.get("semantic_checks", 0) + 1
            )  # type: ignore

        if validator_score is not None:
            scores["validator_score"] = validator_score
            models_used.append("gemma-2b")

        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for key, score in scores.items():
            weight = weights.get(key, 0.33)
            weighted_sum += score * weight
            total_weight += weight

        confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Consensus: all models agree above threshold
        consensus = all(
            s >= self.config.consensus_threshold  # type: ignore
            for s in scores.values()
        )

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
