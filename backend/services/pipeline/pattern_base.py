"""
Base Collaboration Patterns
===========================

Contains fundamental patterns:
- Chain: Sequential A → B → C processing
- Verify: Generator → Validator → Refine loop
- Semantic Check: Embedding-based similarity verification
"""

import logging
from typing import Any, Dict

from ..types import (
    CollaborationPattern,
    CollaborationResult,
)

logger = logging.getLogger(__name__)


class BasePatternsMixin:
    """
    Mixin providing base collaboration patterns.

    Requires ModelAccessorsMixin and CollaborationHelpersMixin to be mixed in.
    """

    async def _chain_collaboration(
        self,
        _task: str,  # Reserved for future task-specific chains
        input_text: str,
        context: dict[str, Any],
    ) -> CollaborationResult:
        """
        Chain pattern: A → B → C

        Each model processes in sequence, passing enriched context.
        Example: Simplify → Translate → Validate
        """
        models_used = []
        current_text = input_text
        scores = {}

        grade_level = context.get("grade_level", 8)
        subject = context.get("subject", "General")
        target_language = context.get("target_language", "Hindi")

        # Step 1: LLM simplifies
        llm = self._get_llm()  # type: ignore
        if llm:
            try:
                prompt = f"""Simplify this text for Grade {grade_level} {subject} students:

{current_text}

Simplified version (keep key facts, use simple words):"""

                simplified = await llm.generate_async(prompt, max_tokens=4096)

                self._log_message("qwen", "chain", simplified, {"step": "simplify"})  # type: ignore
                current_text = simplified
                models_used.append("qwen-3b")
                scores["simplification"] = 1.0
            except Exception as e:
                logger.warning(f"[Chain] LLM step failed: {e}")

        # Step 2: Translate if needed
        if target_language.lower() != "english":
            translator = self._get_translator()  # type: ignore
            if translator:
                try:
                    translated = translator.translate(current_text, target_language)

                    self._log_message(  # type: ignore
                        "indictrans2",
                        "chain",
                        translated.translated_text,
                        {"step": "translate", "language": target_language},
                    )
                    current_text = translated.translated_text
                    models_used.append("indictrans2")
                    scores["translation"] = (
                        translated.confidence
                        if hasattr(translated, "confidence")
                        else 0.9
                    )
                except Exception as e:
                    logger.warning(f"[Chain] Translation step failed: {e}")

        # Step 3: Validate with semantic check
        embedder = self._get_embedder()  # type: ignore
        if embedder and self.config.enable_semantic_verification:  # type: ignore
            try:
                orig_emb = await self._get_embedding(input_text)  # type: ignore
                curr_emb = await self._get_embedding(current_text)  # type: ignore

                if orig_emb is not None and curr_emb is not None:
                    similarity = self._cosine_similarity(orig_emb, curr_emb)  # type: ignore
                    scores["semantic_preservation"] = similarity
                    models_used.append("bge-m3")

                    self._log_message(  # type: ignore
                        "bge-m3",
                        "chain",
                        f"similarity={similarity:.3f}",
                        {"step": "semantic_check"},
                    )
            except Exception as e:
                logger.warning(f"[Chain] Semantic check failed: {e}")

        # Calculate overall confidence
        confidence = sum(scores.values()) / len(scores) if scores else 0.5

        return CollaborationResult(
            pattern=CollaborationPattern.CHAIN,
            final_output=current_text,
            confidence=confidence,
            consensus=True,
            iterations=len(models_used),
            participating_models=models_used,
            messages=[],
            scores=scores,
            processing_time_ms=0,
        )

    async def _verify_collaboration(
        self,
        task: str,
        input_text: str,
        context: dict[str, Any],
    ) -> CollaborationResult:
        """
        Verify pattern: Generator → Validator → Refine if needed

        One model generates, another validates, generator refines based on feedback.
        """
        models_used = []
        scores = {}
        iterations = 0

        grade_level = context.get("grade_level", 8)
        subject = context.get("subject", "General")

        llm = self._get_llm()  # type: ignore
        validator = self._get_validator()  # type: ignore

        if not llm:
            return self._fallback_result(CollaborationPattern.VERIFY, input_text)  # type: ignore

        current_text = input_text
        best_text = input_text
        best_score = 0.0

        for i in range(self.config.max_iterations):  # type: ignore
            iterations += 1

            # Step 1: Generate/refine
            if i == 0:
                # Initial generation
                prompt = f"""You are an expert educational content creator.

Task: {task}
Grade Level: {grade_level}
Subject: {subject}

Input:
{input_text}

Provide high-quality output suitable for the specified grade level:"""

                current_text = await llm.generate_async(prompt, max_tokens=4096)
                models_used.append("qwen-3b")
                self._log_message(  # type: ignore
                    "qwen",
                    "validator",
                    current_text,
                    {"iteration": i + 1, "action": "generate"},
                )

            # Step 2: Validate
            if validator:
                try:
                    eval_result = await validator.evaluate(
                        original_text=input_text,
                        processed_text=current_text,
                        grade_level=grade_level,
                        subject=subject,
                    )

                    score = eval_result.overall_score
                    scores[f"iteration_{i + 1}"] = score

                    if "gemma" not in models_used:
                        models_used.append("gemma-2b")

                    self._log_message(  # type: ignore
                        "gemma",
                        "qwen",
                        f"score={score:.2f}",
                        {"iteration": i + 1, "action": "validate"},
                    )

                    # Track best
                    if score > best_score:
                        best_score = score
                        best_text = current_text

                    # Check if good enough
                    if score >= self.config.min_confidence * 10:  # type: ignore
                        logger.info(
                            f"[Verify] Target reached at iteration {i + 1}: {score:.2f}"
                        )
                        break

                    # Step 3: Refine based on feedback
                    if i < self.config.max_iterations - 1:  # type: ignore
                        # Get dimension feedback
                        weak_dims = [
                            dim
                            for dim, ds in eval_result.dimension_scores.items()
                            if ds.score < 8.0
                        ]

                        if weak_dims:
                            feedback_prompt = f"""The following content needs improvement.

Current output:
{current_text}

Issues to address:
{", ".join(weak_dims)}

Current score: {score:.1f}/10

Please improve the content to address these issues while maintaining accuracy:"""

                            current_text = await llm.generate_async(
                                feedback_prompt, max_tokens=4096
                            )
                            self._log_message(  # type: ignore
                                "qwen",
                                "validator",
                                current_text,
                                {"iteration": i + 1, "action": "refine"},
                            )
                except Exception as e:
                    logger.warning(f"[Verify] Validation failed: {e}")
                    scores[f"iteration_{i + 1}"] = 7.0  # Default score
                    break
            else:
                # No validator, just return generated output
                break

        confidence = best_score / 10.0  # Normalize to 0-1

        return CollaborationResult(
            pattern=CollaborationPattern.VERIFY,
            final_output=best_text,
            confidence=confidence,
            consensus=confidence >= self.config.min_confidence,  # type: ignore
            iterations=iterations,
            participating_models=models_used,
            messages=[],
            scores=scores,
            processing_time_ms=0,
        )

    async def _semantic_check_collaboration(
        self,
        input_text: str,
        processed_text: str,
        context: dict[str, Any],
    ) -> CollaborationResult:
        """
        Semantic check pattern: Verify meaning preservation via embeddings.

        Uses BGE-M3 to compare original and processed text embeddings.
        """
        embedder = self._get_embedder()  # type: ignore

        if not embedder:
            return self._fallback_result(  # type: ignore
                CollaborationPattern.SEMANTIC_CHECK, processed_text
            )

        self._metrics["semantic_check_used"] += 1  # type: ignore

        try:
            orig_emb = await self._get_embedding(input_text)  # type: ignore
            proc_emb = await self._get_embedding(processed_text)  # type: ignore

            if orig_emb is None or proc_emb is None:
                return self._fallback_result(  # type: ignore
                    CollaborationPattern.SEMANTIC_CHECK, processed_text
                )

            similarity = self._cosine_similarity(orig_emb, proc_emb)  # type: ignore

            threshold = context.get("similarity_threshold", 0.75)
            consensus = similarity >= threshold

            self._log_message(  # type: ignore
                "bge-m3",
                "semantic_check",
                f"similarity={similarity:.3f}, threshold={threshold}",
                {"consensus": consensus},
            )

            return CollaborationResult(
                pattern=CollaborationPattern.SEMANTIC_CHECK,
                final_output=processed_text,
                confidence=similarity,
                consensus=consensus,
                iterations=1,
                participating_models=["bge-m3"],
                messages=[],
                scores={"semantic_similarity": similarity},
                processing_time_ms=0,
            )

        except Exception as e:
            logger.warning(f"[SemanticCheck] Failed: {e}")
            return self._fallback_result(  # type: ignore
                CollaborationPattern.SEMANTIC_CHECK, processed_text
            )
