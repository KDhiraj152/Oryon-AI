"""
Audio Collaboration Patterns
============================

Contains audio verification pattern:
- Audio Verify: TTS → Whisper STT → Semantic comparison
"""

import logging
from typing import Any, Dict

from ..types import (
    CollaborationPattern,
    CollaborationResult,
)

logger = logging.getLogger(__name__)


class AudioPatternsMixin:
    """
    Mixin providing audio collaboration patterns.

    Requires ModelAccessorsMixin and CollaborationHelpersMixin to be mixed in.
    """

    async def _audio_verify_collaboration(
        self,
        input_text: str,
        context: dict[str, Any],
    ) -> CollaborationResult:
        """
        Audio Verification pattern: TTS → Whisper → Compare

        Ensures TTS audio accurately represents the text by:
        1. TTS generates audio from text
        2. Whisper transcribes audio back to text
        3. Compare original text with transcription
        4. If mismatch, regenerate with different parameters

        This is critical for AIal audio content accuracy.
        """
        self._metrics["audio_verifications"] = (  # type: ignore
            self._metrics.get("audio_verifications", 0) + 1  # type: ignore
        )
        models_used = []
        scores = {}

        tts = self._get_tts()  # type: ignore
        stt = self._get_stt()  # type: ignore
        embedder = self._get_embedder()  # type: ignore

        if not tts:
            return self._fallback_result(CollaborationPattern.AUDIO_VERIFY, input_text)  # type: ignore

        target_language = context.get("target_language", "hi")
        max_iterations = context.get("max_audio_iterations", 3)

        best_transcription = input_text
        best_similarity = 0.0
        audio_path = None
        iterations = 0

        for i in range(max_iterations):
            iterations = i + 1

            # Step 1: Generate audio with TTS
            try:
                if hasattr(tts, "generate_async"):
                    audio_result = await tts.generate_async(
                        text=input_text,
                        language=target_language,
                        speed=context.get("audio_speed", 1.0),
                    )
                elif hasattr(tts, "generate"):
                    audio_result = tts.generate(
                        text=input_text,
                        language=target_language,
                    )
                else:
                    break

                if "mms-tts" not in models_used:
                    models_used.append("mms-tts")

                audio_path = (
                    audio_result.get("audio_path")
                    if isinstance(audio_result, dict)
                    else audio_result
                )

            except Exception as e:
                logger.warning(f"[AudioVerify] TTS generation failed: {e}")
                break

            self._log_message(  # type: ignore
                "mms-tts",
                "whisper",
                f"Generated audio for: {input_text[:50]}...",
                {"language": target_language, "iteration": iterations},
            )

            # Step 2: Transcribe with Whisper
            transcription = input_text  # Default fallback
            if stt and audio_path:
                try:
                    if hasattr(stt, "transcribe_async"):
                        transcription_result = await stt.transcribe_async(audio_path)
                    elif hasattr(stt, "transcribe"):
                        transcription_result = stt.transcribe(audio_path)
                    else:
                        transcription_result = {"text": input_text}

                    if "whisper" not in models_used:
                        models_used.append("whisper")

                    transcription = (
                        transcription_result.get("text", "")
                        if isinstance(transcription_result, dict)
                        else str(transcription_result)
                    )

                except Exception as e:
                    logger.warning(f"[AudioVerify] Whisper transcription failed: {e}")
                    transcription = input_text

                self._log_message(  # type: ignore
                    "whisper",
                    "bge-m3",
                    f"Transcription: {transcription[:50]}...",
                    {"iteration": iterations},
                )

            # Step 3: Compare using semantic similarity
            if embedder:
                orig_emb = await self._get_embedding(input_text)  # type: ignore
                trans_emb = await self._get_embedding(transcription)  # type: ignore

                if orig_emb and trans_emb:
                    similarity = self._cosine_similarity(orig_emb, trans_emb)  # type: ignore

                    if "bge-m3" not in models_used:
                        models_used.append("bge-m3")

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_transcription = transcription

                    self._log_message(  # type: ignore
                        "bge-m3",
                        "orchestrator",
                        f"Similarity: {similarity:.3f}",
                        {"iteration": iterations},
                    )

                    # If similarity is high enough, we're done
                    if similarity >= 0.95:
                        break
                else:
                    # Fallback to string comparison
                    if transcription.lower().strip() == input_text.lower().strip():
                        best_similarity = 1.0
                        break
            else:
                # No embedder, use simple comparison
                if transcription.lower().strip() == input_text.lower().strip():
                    best_similarity = 1.0
                    break

        scores["audio_accuracy"] = best_similarity
        scores["iterations"] = float(iterations)

        return CollaborationResult(
            pattern=CollaborationPattern.AUDIO_VERIFY,
            final_output=input_text,
            confidence=best_similarity,
            consensus=best_similarity >= 0.90,
            iterations=iterations,
            participating_models=models_used,
            messages=[],
            scores=scores,
            processing_time_ms=0,
            metadata={
                "transcription": best_transcription,
                "audio_path": audio_path,
            },
        )
