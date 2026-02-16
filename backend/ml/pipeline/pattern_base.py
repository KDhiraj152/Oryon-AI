"""
Base Collaboration Patterns
===========================

Contains fundamental patterns:
- Chain: Sequential A → B → C processing
- Verify: Generator → Validator → Refine loop
- Semantic Check: Embedding-based similarity verification
"""

import logging
from typing import Any

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

    async def _chain_step_simplify(
        self,
        text: str,
        complexity_level: int,
        subject: str,
    ) -> tuple[str, list[str], dict[str, float]]:
        """Chain step: LLM simplification."""
        llm = self._get_llm()  # type: ignore
        if not llm:
            return text, [], {}

        try:
            prompt = f"""Simplify this text for Grade {complexity_level} {subject} users:

{text}

Simplified version (keep key facts, use simple words):"""
            simplified = await llm.generate_async(prompt, max_tokens=4096)
            self._log_message("qwen", "chain", simplified, {"step": "simplify"})  # type: ignore
            return simplified, ["qwen-3b"], {"simplification": 1.0}
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("[Chain] LLM step failed: %s", e)
            return text, [], {}

    async def _chain_step_translate(
        self,
        text: str,
        target_language: str,
    ) -> tuple[str, list[str], dict[str, float]]:
        """Chain step: Translation."""
        import asyncio

        if target_language.lower() == "english":
            return text, [], {}

        translator = self._get_translator()  # type: ignore
        if not translator:
            return text, [], {}

        try:
            translated = translator.translate(text, target_language)
            await asyncio.sleep(0)  # yield to event loop after sync call
            self._log_message(  # type: ignore
                "indictrans2", "chain", translated.translated_text,
                {"step": "translate", "language": target_language},
            )
            conf = translated.confidence if hasattr(translated, "confidence") else 0.9
            return translated.translated_text, ["indictrans2"], {"translation": conf}
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("[Chain] Translation step failed: %s", e)
            return text, [], {}

    async def _chain_step_semantic(
        self,
        current_text: str,
        orig_emb_future: Any,
    ) -> tuple[list[str], dict[str, float]]:
        """Chain step: Semantic verification."""
        try:
            orig_emb = await orig_emb_future if orig_emb_future else None
            curr_emb = await self._get_embedding(current_text)  # type: ignore

            if orig_emb is not None and curr_emb is not None:
                similarity = self._cosine_similarity(orig_emb, curr_emb)  # type: ignore
                self._log_message(  # type: ignore
                    "bge-m3", "chain", f"similarity={similarity:.3f}",
                    {"step": "semantic_check"},
                )
                return ["bge-m3"], {"semantic_preservation": similarity}
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("[Chain] Semantic check failed: %s", e)
        return [], {}

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
        import asyncio as _aio

        models_used: list[str] = []
        scores: dict[str, float] = {}
        complexity_level = context.get("complexity_level", 8)
        subject = context.get("subject", "General")
        target_language = context.get("target_language", "Hindi")

        # Pre-compute embedding of original text as background task
        orig_emb_future: _aio.Task | None = None
        embedder = self._get_embedder()  # type: ignore
        if embedder and self.config.enable_semantic_verification:  # type: ignore
            orig_emb_future = _aio.ensure_future(self._get_embedding(input_text))  # type: ignore

        # Step 1: Simplify
        current_text, step_models, step_scores = await self._chain_step_simplify(
            input_text, complexity_level, subject
        )
        models_used.extend(step_models)
        scores.update(step_scores)

        # Step 2: Translate
        current_text, step_models, step_scores = await self._chain_step_translate(
            current_text, target_language
        )
        models_used.extend(step_models)
        scores.update(step_scores)

        # Step 3: Semantic check
        if embedder and self.config.enable_semantic_verification:  # type: ignore
            step_models, step_scores = await self._chain_step_semantic(
                current_text, orig_emb_future
            )
            models_used.extend(step_models)
            scores.update(step_scores)

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

    async def _verify_initial_generate(
        self, llm: Any, input_text: str, task: str, complexity_level: int, subject: str,
    ) -> str:
        """Generate initial output for verify pattern."""
        prompt = f"""You are an expert content creator.

Task: {task}
Complexity Level: {complexity_level}
Subject: {subject}

Input:
{input_text}

Provide high-quality output suitable for the specified complexity level:"""
        result = await llm.generate_async(prompt, max_tokens=4096)
        self._log_message("qwen", "validator", result, {"iteration": 1, "action": "generate"})  # type: ignore
        return str(result)

    async def _verify_refine(
        self, llm: Any, current_text: str, eval_result: Any, score: float, iteration: int,
    ) -> str | None:
        """Refine text based on validation feedback. Returns None if no refinement needed."""
        weak_dims = [
            dim for dim, ds in eval_result.dimension_scores.items() if ds.score < 8.0
        ]
        if not weak_dims:
            return None

        feedback_prompt = f"""The following content needs improvement.

Current output:
{current_text}

Issues to address:
{", ".join(weak_dims)}

Current score: {score:.1f}/10

Please improve the content to address these issues while maintaining accuracy:"""
        refined = await llm.generate_async(feedback_prompt, max_tokens=4096)
        self._log_message(  # type: ignore
            "qwen", "validator", refined, {"iteration": iteration, "action": "refine"},
        )
        return str(refined)

    async def _verify_evaluate_step(
        self,
        validator: Any,
        llm: Any,
        input_text: str,
        current_text: str,
        models_used: list[str],
        scores: dict[str, float],
        iteration_idx: int,
        best_score: float,
        best_text: str,
        complexity_level: int,
        subject: str,
    ) -> tuple[float, str, str, bool]:
        """Run one verify-refine iteration. Returns (best_score, best_text, current_text, should_stop)."""
        eval_result = await validator.evaluate(
            original_text=input_text, processed_text=current_text,
            complexity_level=complexity_level, subject=subject,
        )
        score = eval_result.overall_score
        scores[f"iteration_{iteration_idx + 1}"] = score

        if "qwen3" not in str(models_used):
            models_used.append("qwen3-8b")

        self._log_message(  # type: ignore
            "qwen3", "qwen", f"score={score:.2f}",
            {"iteration": iteration_idx + 1, "action": "validate"},
        )

        if score > best_score:
            best_score = score
            best_text = current_text

        if score >= self.config.min_confidence * 10:  # type: ignore
            logger.info("[Verify] Target reached at iteration %s: %.2f", iteration_idx + 1, score)
            return best_score, best_text, current_text, True

        # Refine if not last iteration
        if iteration_idx < self.config.max_iterations - 1:  # type: ignore
            refined = await self._verify_refine(llm, current_text, eval_result, score, iteration_idx + 1)
            if refined:
                current_text = refined

        return best_score, best_text, current_text, False

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
        models_used: list[str] = []
        scores: dict[str, float] = {}
        iterations = 0

        complexity_level = context.get("complexity_level", 8)
        subject = context.get("subject", "General")

        llm = self._get_llm()  # type: ignore
        validator = self._get_validator()  # type: ignore

        if not llm:
            return self._fallback_result(CollaborationPattern.VERIFY, input_text)  # type: ignore

        best_text = input_text
        best_score = 0.0

        # Initial generation
        current_text = await self._verify_initial_generate(llm, input_text, task, complexity_level, subject)
        models_used.append("qwen-3b")

        for i in range(self.config.max_iterations):  # type: ignore
            iterations += 1

            if not validator:
                break

            try:
                best_score, best_text, current_text, should_stop = (
                    await self._verify_evaluate_step(
                        validator, llm, input_text, current_text,
                        models_used, scores, i, best_score, best_text,
                        complexity_level, subject,
                    )
                )
                if should_stop:
                    break
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("[Verify] Validation failed: %s", e)
                scores[f"iteration_{i + 1}"] = 7.0
                break

        confidence = best_score / 10.0

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

        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("[SemanticCheck] Failed: %s", e)
            return self._fallback_result(  # type: ignore
                CollaborationPattern.SEMANTIC_CHECK, processed_text
            )
