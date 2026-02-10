"""
Translation Collaboration Patterns
==================================

Contains translation-related patterns:
- Back-translate: Forward → Backward translation verification
"""

import logging
from typing import Any, Dict

from ..types import (
    CollaborationPattern,
    CollaborationResult,
)

logger = logging.getLogger(__name__)


class TranslationPatternsMixin:
    """
    Mixin providing translation collaboration patterns.

    Requires ModelAccessorsMixin and CollaborationHelpersMixin to be mixed in.
    """

    async def _back_translate_collaboration(
        self,
        input_text: str,
        context: dict[str, Any],
    ) -> CollaborationResult:
        """
        Back-translation pattern: Translate → Back-translate → Compare

        Translates to target language, then back to source, compares for accuracy.
        This validates translation quality without human reference.
        """
        models_used = []
        scores = {}

        target_language = context.get("target_language", "Hindi")
        source_language = context.get("source_language", "English")

        translator = self._get_translator()  # type: ignore
        embedder = self._get_embedder()  # type: ignore

        if not translator:
            return self._fallback_result(  # type: ignore
                CollaborationPattern.BACK_TRANSLATE, input_text
            )

        self._metrics["back_translation_used"] += 1  # type: ignore

        try:
            # Step 1: Forward translation
            forward_result = translator.translate(input_text, target_language)
            translated = forward_result.translated_text
            models_used.append("indictrans2")

            self._log_message(  # type: ignore
                "indictrans2",
                "back_translate",
                translated,
                {"step": "forward", "to": target_language},
            )

            # Step 2: Back translation
            back_result = translator.translate(translated, source_language)
            back_translated = back_result.translated_text

            self._log_message(  # type: ignore
                "indictrans2",
                "back_translate",
                back_translated,
                {"step": "backward", "to": source_language},
            )

            # Step 3: Compare original and back-translated using embeddings
            semantic_score = 0.85  # Default
            orig_emb = None

            if embedder:
                orig_emb = await self._get_embedding(input_text)  # type: ignore
                back_emb = await self._get_embedding(back_translated)  # type: ignore

                if orig_emb is not None and back_emb is not None:
                    semantic_score = self._cosine_similarity(orig_emb, back_emb)  # type: ignore
                    models_used.append("bge-m3")

                    self._log_message(  # type: ignore
                        "bge-m3",
                        "back_translate",
                        f"similarity={semantic_score:.3f}",
                        {"step": "compare"},
                    )

            scores["forward_translation"] = 1.0
            scores["back_translation"] = 1.0
            scores["semantic_preservation"] = semantic_score

            # High similarity = good translation
            confidence = semantic_score
            consensus = semantic_score >= self.config.semantic_threshold  # type: ignore

            # If translation quality is poor, try to improve
            final_output = translated
            if not consensus and self.config.max_iterations > 1:  # type: ignore
                # Ask LLM to improve translation
                llm = self._get_llm()  # type: ignore
                if llm:
                    improve_prompt = f"""The following translation may have lost some meaning.

Original ({source_language}):
{input_text}

Translation ({target_language}):
{translated}

Back-translation (for comparison):
{back_translated}

Please provide an improved {target_language} translation that better preserves the original meaning:"""

                    improved = await llm.generate_async(improve_prompt, max_tokens=4096)

                    # Verify improvement
                    improved_back = translator.translate(
                        improved, source_language
                    ).translated_text
                    improved_emb = await self._get_embedding(improved_back)  # type: ignore

                    if improved_emb is not None and orig_emb is not None:
                        improved_score = self._cosine_similarity(  # type: ignore
                            orig_emb, improved_emb
                        )
                        if improved_score > semantic_score:
                            final_output = improved
                            confidence = improved_score
                            consensus = improved_score >= self.config.semantic_threshold  # type: ignore
                            scores["improved_translation"] = improved_score
                            models_used.append("qwen-3b")

            return CollaborationResult(
                pattern=CollaborationPattern.BACK_TRANSLATE,
                final_output=final_output,
                confidence=confidence,
                consensus=consensus,
                iterations=2 if "qwen-3b" in models_used else 1,
                participating_models=models_used,
                messages=[],
                scores=scores,
                processing_time_ms=0,
                metadata={
                    "original": input_text,
                    "back_translated": back_translated,
                    "target_language": target_language,
                },
            )

        except Exception as e:
            logger.error(f"[BackTranslate] Failed: {e}")
            return self._fallback_result(  # type: ignore
                CollaborationPattern.BACK_TRANSLATE, input_text
            )
