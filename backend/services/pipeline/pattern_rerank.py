"""
Rerank Collaboration Patterns
=============================

Contains reranking pattern:
- Rerank: Generate multiple candidates → BGE-Reranker selects best
"""

import logging
from typing import Any, Dict, Optional

from ..types import (
    MODEL_QWEN25,
    CollaborationPattern,
    CollaborationResult,
)

logger = logging.getLogger(__name__)


class RerankPatternsMixin:
    """
    Mixin providing reranking collaboration patterns.

    Requires ModelAccessorsMixin and CollaborationHelpersMixin to be mixed in.
    """

    async def _rerank_collaboration(
        self,
        task: str,
        input_text: str,
        context: dict[str, Any],
    ) -> CollaborationResult:
        """
        Rerank pattern: Generate multiple → BGE-Reranker selects best

        For high-quality output:
        1. LLM generates multiple candidate outputs (3-5)
        2. BGE-Reranker scores each candidate against requirements
        3. Best candidate is selected
        4. Optional: Semantic check for verification

        Best for critical content where quality matters most.
        """
        self._metrics["reranking_used"] = (  # type: ignore
            self._metrics.get("reranking_used", 0) + 1  # type: ignore
        )
        models_used = []
        scores = {}

        llm = self._get_llm()  # type: ignore
        reranker = self._get_reranker()  # type: ignore

        if not llm:
            return self._fallback_result(CollaborationPattern.RERANK, input_text)  # type: ignore

        num_candidates = context.get("num_candidates", 3)
        complexity_level = context.get("complexity_level", 8)
        # subject available in context for task-specific ranking

        # Step 1: Generate multiple candidates
        candidates = []

        for i in range(num_candidates):
            try:
                # Vary temperature/parameters for diversity
                temperature = 0.5 + (i * 0.15)  # 0.5, 0.65, 0.8

                if task == "simplify":
                    candidate = await self._simplify_text(  # type: ignore
                        input_text,
                        complexity_level,
                        {**context, "temperature": temperature},
                    )
                else:
                    # Generic LLM generation
                    candidate = await self._generate_with_llm(
                        task,
                        input_text,
                        {**context, "temperature": temperature},
                    )

                if candidate and candidate not in candidates:
                    candidates.append(candidate)

            except Exception as e:
                logger.warning(f"[Rerank] Candidate {i + 1} generation failed: {e}")

        if not candidates:
            return self._fallback_result(CollaborationPattern.RERANK, input_text)  # type: ignore

        models_used.append(MODEL_QWEN25)

        self._log_message(  # type: ignore
            MODEL_QWEN25,
            "bge-reranker",
            f"Generated {len(candidates)} candidates",
            {"num_candidates": len(candidates)},
        )

        # Step 2: Rerank candidates
        best_candidate = candidates[0]
        best_score = 0.0

        if reranker and len(candidates) > 1:
            try:
                # Create query for reranking based on task requirements
                query = self._create_rerank_query(task, input_text, context)

                if hasattr(reranker, "rerank_async"):
                    rerank_result = await reranker.rerank_async(
                        query=query,
                        documents=candidates,
                    )
                elif hasattr(reranker, "rerank"):
                    rerank_result = reranker.rerank(
                        query=query,
                        documents=candidates,
                    )
                elif hasattr(reranker, "compute_score"):
                    # BGE-Reranker style interface
                    rerank_scores = []
                    for candidate in candidates:
                        score = reranker.compute_score([query, candidate])
                        rerank_scores.append(
                            {
                                "text": candidate,
                                "score": float(score)
                                if hasattr(score, "__float__")
                                else score,
                            }
                        )
                    rerank_result = sorted(
                        rerank_scores, key=lambda x: x["score"], reverse=True
                    )
                else:
                    rerank_result = [{"text": c, "score": 0.5} for c in candidates]

                models_used.append("bge-reranker")

                # Extract best candidate
                if isinstance(rerank_result, list) and len(rerank_result) > 0:
                    top_result = rerank_result[0]
                    if isinstance(top_result, dict):
                        best_candidate = top_result.get(
                            "text", top_result.get("document", candidates[0])
                        )
                        best_score = top_result.get("score", 0.8)
                    else:
                        best_candidate = str(top_result)
                        best_score = 0.8

                # Store all scores
                for i, result in enumerate(rerank_result[: len(candidates)]):
                    if isinstance(result, dict):
                        scores[f"candidate_{i + 1}"] = result.get("score", 0.0)

                self._log_message(  # type: ignore
                    "bge-reranker",
                    "semantic_check",
                    f"Best candidate score: {best_score:.3f}",
                    {"num_ranked": len(rerank_result)},
                )

            except Exception as e:
                logger.warning(f"[Rerank] Reranking failed: {e}, using first candidate")
                best_score = 0.7
        else:
            best_score = 0.7  # Default if no reranker

        # Step 3: Semantic verification of best candidate
        embedder = self._get_embedder()  # type: ignore
        semantic_score = 0.0

        if embedder:
            orig_emb = await self._get_embedding(input_text)  # type: ignore
            best_emb = await self._get_embedding(best_candidate)  # type: ignore

            if orig_emb and best_emb:
                semantic_score = self._cosine_similarity(orig_emb, best_emb)  # type: ignore
                scores["semantic_preservation"] = semantic_score

                if "bge-m3" not in models_used:
                    models_used.append("bge-m3")

        # Combined confidence
        overall_confidence = (
            (best_score * 0.6 + semantic_score * 0.4)
            if semantic_score > 0
            else best_score
        )
        scores["overall"] = overall_confidence

        return CollaborationResult(
            pattern=CollaborationPattern.RERANK,
            final_output=best_candidate,
            confidence=overall_confidence,
            consensus=overall_confidence >= 0.75,
            iterations=len(candidates),
            participating_models=models_used,
            messages=[],
            scores=scores,
            processing_time_ms=0,
            metadata={
                "candidates_generated": len(candidates),
                "best_rerank_score": best_score,
            },
        )

    def _create_rerank_query(
        self,
        task: str,
        original_text: str,
        context: dict[str, Any],
    ) -> str:
        """Create a query string for reranking based on task requirements."""
        complexity_level = context.get("complexity_level", 8)
        subject = context.get("subject", "General")

        if task == "simplify":
            return f"""Find the best simplified version of content that:
- Is appropriate for complexity level {complexity_level} users
- Preserves all key concepts from: {original_text[:200]}...
- Uses clear, simple language
- Is engaging and easy to understand for {subject} learning"""

        elif task == "translate":
            target_lang = context.get("target_language", "Hindi")
            return f"""Find the best translation that:
- Accurately conveys the meaning of: {original_text[:200]}...
- Is natural and fluent in {target_lang}
- Preserves semantic context and terminology
- Is appropriate for users"""

        else:
            return f"""Find the best version that:
- Is high quality and accurate
- Based on: {original_text[:200]}...
- Is clear and well-structured"""

    async def _generate_with_llm(
        self,
        task: str,
        input_text: str,
        context: dict[str, Any],
    ) -> str | None:
        """Generic LLM generation with task context."""
        llm = self._get_llm()  # type: ignore
        if not llm:
            return None

        temperature = context.get("temperature", 0.7)

        try:
            prompt = f"""Task: {task}
Input: {input_text}
Context: {context}

Generate a high-quality output for the above task."""

            if hasattr(llm, "generate_async"):
                result = await llm.generate_async(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=4096,
                )
            elif hasattr(llm, "generate"):
                result = llm.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=4096,
                )
            elif callable(llm):
                result = llm(prompt, temperature=temperature)
            else:
                return None

            if isinstance(result, dict):
                return result.get("text", result.get("output", str(result)))
            return str(result)

        except Exception as e:
            logger.warning(f"[Collaborator] LLM generation failed: {e}")
            return None
