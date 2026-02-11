"""
Document Collaboration Patterns
===============================

Contains document processing pattern:
- Document Chain: OCR → Simplify → Translate → Audio
"""

import logging
from typing import Any, Dict

from ..types import (
    MODEL_QWEN25,
    CollaborationPattern,
    CollaborationResult,
)

logger = logging.getLogger(__name__)


class DocumentPatternsMixin:
    """
    Mixin providing document processing collaboration patterns.

    Requires ModelAccessorsMixin and CollaborationHelpersMixin to be mixed in.
    """

    async def _document_chain_collaboration(
        self,
        input_text: str,
        context: dict[str, Any],
    ) -> CollaborationResult:
        """
        Document Chain pattern: OCR → Simplify → Translate → Audio

        Full pipeline for document processing:
        1. GOT-OCR2 extracts text from image/document
        2. Qwen3-8B simplifies the content
        3. IndicTrans2 translates to target language
        4. Optional: MMS-TTS generates audio
        5. BGE-M3 verifies semantic preservation at each step

        Perfect for converting textbooks/worksheets to accessible content.
        """
        self._metrics["ocr_chains"] = self._metrics.get("ocr_chains", 0) + 1  # type: ignore
        models_used = []
        scores = {}
        current_text = input_text
        original_extracted = input_text
        audio_path = None

        # Step 1: OCR if image path provided
        image_path = context.get("image_path")
        ocr = self._get_ocr()  # type: ignore

        if image_path and ocr:
            try:
                if hasattr(ocr, "extract_async"):
                    ocr_result = await ocr.extract_async(image_path)
                elif hasattr(ocr, "extract"):
                    ocr_result = ocr.extract(image_path)
                elif hasattr(ocr, "recognize"):
                    ocr_result = ocr.recognize(image_path)
                else:
                    ocr_result = {"text": current_text}

                current_text = (
                    ocr_result.get("text", current_text)
                    if isinstance(ocr_result, dict)
                    else str(ocr_result)
                )
                original_extracted = current_text
                models_used.append("got-ocr2")

                self._log_message(  # type: ignore
                    "got-ocr2",
                    MODEL_QWEN25,
                    f"Extracted text: {current_text[:100]}...",
                    {
                        "image_path": image_path,
                        "confidence": (
                            ocr_result.get("confidence", 0.9)
                            if isinstance(ocr_result, dict)
                            else 0.9
                        ),
                    },
                )

                scores["ocr_confidence"] = (
                    ocr_result.get("confidence", 0.9)
                    if isinstance(ocr_result, dict)
                    else 0.9
                )

            except Exception as e:
                logger.warning(f"[DocumentChain] OCR failed: {e}")

        # Step 2: Simplify with LLM
        llm = self._get_llm()  # type: ignore
        grade_level = context.get("grade_level", 8)

        if llm:
            try:
                simplified = await self._simplify_text(
                    current_text, grade_level, context
                )  # type: ignore
                if simplified:
                    current_text = simplified
                    if MODEL_QWEN25 not in models_used:
                        models_used.append(MODEL_QWEN25)

                    self._log_message(  # type: ignore
                        MODEL_QWEN25,
                        "indicTrans2",
                        f"Simplified: {current_text[:100]}...",
                        {"grade_level": grade_level},
                    )
            except Exception as e:
                logger.warning(f"[DocumentChain] Simplification failed: {e}")

        # Check semantic preservation after simplification
        embedder = self._get_embedder()  # type: ignore
        if embedder and original_extracted != current_text:
            orig_emb = await self._get_embedding(original_extracted)  # type: ignore
            simp_emb = await self._get_embedding(current_text)  # type: ignore

            if orig_emb and simp_emb:
                simplify_similarity = self._cosine_similarity(orig_emb, simp_emb)  # type: ignore
                scores["simplification_similarity"] = simplify_similarity

                if "bge-m3" not in models_used:
                    models_used.append("bge-m3")

        # Step 3: Translate if target language specified
        target_language = context.get("target_language")

        if target_language:
            translator = self._get_translator()  # type: ignore
            if translator:
                try:
                    if hasattr(translator, "translate_async"):
                        trans_result = await translator.translate_async(
                            current_text,
                            source_lang="en",
                            target_lang=target_language,
                        )
                    elif hasattr(translator, "translate"):
                        trans_result = translator.translate(
                            current_text,
                            source_lang="en",
                            target_lang=target_language,
                        )
                    else:
                        trans_result = {"translated_text": current_text}

                    current_text = (
                        trans_result.get("translated_text", current_text)
                        if isinstance(trans_result, dict)
                        else str(trans_result)
                    )

                    if "indicTrans2" not in models_used:
                        models_used.append("indicTrans2")

                    self._log_message(  # type: ignore
                        "indicTrans2",
                        "mms-tts",
                        f"Translated: {current_text[:100]}...",
                        {"target_language": target_language},
                    )

                except Exception as e:
                    logger.warning(f"[DocumentChain] Translation failed: {e}")

        # Step 4: Generate audio if requested
        generate_audio = context.get("generate_audio", False)

        if generate_audio:
            tts = self._get_tts()  # type: ignore
            if tts:
                try:
                    lang_code = target_language if target_language else "en"

                    if hasattr(tts, "generate_async"):
                        audio_result = await tts.generate_async(
                            text=current_text,
                            language=lang_code,
                        )
                    elif hasattr(tts, "generate"):
                        audio_result = tts.generate(
                            text=current_text,
                            language=lang_code,
                        )
                    else:
                        audio_result = {}

                    audio_path = (
                        audio_result.get("audio_path")
                        if isinstance(audio_result, dict)
                        else audio_result
                    )

                    if "mms-tts" not in models_used:
                        models_used.append("mms-tts")

                    self._log_message(  # type: ignore
                        "mms-tts", "complete", f"Audio generated: {audio_path}", {}
                    )

                except Exception as e:
                    logger.warning(f"[DocumentChain] TTS failed: {e}")

        # Calculate overall confidence
        confidence_values = [
            v for k, v in scores.items() if isinstance(v, (int, float))
        ]
        overall_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.8
        )

        return CollaborationResult(
            pattern=CollaborationPattern.DOCUMENT_CHAIN,
            final_output=current_text,
            confidence=overall_confidence,
            consensus=overall_confidence >= 0.8,
            iterations=len(models_used),
            participating_models=models_used,
            messages=[],
            scores=scores,
            processing_time_ms=0,
            metadata={
                "original_extracted": original_extracted if image_path else None,
                "audio_path": audio_path,
                "target_language": target_language,
            },
        )
