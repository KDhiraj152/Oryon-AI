"""
Convenience Functions
=====================

High-level convenience functions for common collaboration tasks.
"""

from typing import Any, Dict, Optional, Tuple

from .orchestrator import ModelCollaborator
from .types import CollaborationPattern

# Singleton instance
_collaborator_instance: ModelCollaborator | None = None


def get_model_collaborator() -> ModelCollaborator:
    """Get singleton ModelCollaborator instance."""
    global _collaborator_instance
    if _collaborator_instance is None:
        _collaborator_instance = ModelCollaborator()
    return _collaborator_instance


async def collaborate_and_simplify(
    text: str,
    grade_level: int = 8,
    subject: str = "General",
    verify: bool = True,
) -> tuple[str, float]:
    """
    Simplify text with model collaboration.

    Returns:
        Tuple of (simplified_text, confidence_score)
    """
    collaborator = get_model_collaborator()

    pattern = CollaborationPattern.VERIFY if verify else CollaborationPattern.CHAIN

    result = await collaborator.collaborate(
        task="simplify",
        input_text=text,
        pattern=pattern,
        context={
            "grade_level": grade_level,
            "subject": subject,
        },
    )

    return result.final_output, result.confidence


async def collaborate_and_translate(
    text: str,
    target_language: str = "Hindi",
    verify_quality: bool = True,
) -> tuple[str, float, bool]:
    """
    Translate text with back-translation verification.

    Returns:
        Tuple of (translated_text, confidence_score, quality_verified)
    """
    collaborator = get_model_collaborator()

    pattern = (
        CollaborationPattern.BACK_TRANSLATE
        if verify_quality
        else CollaborationPattern.CHAIN
    )

    result = await collaborator.collaborate(
        task="translate",
        input_text=text,
        pattern=pattern,
        context={
            "target_language": target_language,
            "source_language": "English",
        },
    )

    return result.final_output, result.confidence, result.consensus


async def ensemble_evaluate(
    original_text: str,
    processed_text: str,
    grade_level: int = 8,
    subject: str = "General",
) -> tuple[float, dict[str, float]]:
    """
    Evaluate content quality using ensemble of models.

    Returns:
        Tuple of (overall_confidence, individual_scores)
    """
    collaborator = get_model_collaborator()

    result = await collaborator.collaborate(
        task="evaluate",
        input_text=original_text,
        pattern=CollaborationPattern.ENSEMBLE,
        context={
            "processed_text": processed_text,
            "grade_level": grade_level,
            "subject": subject,
        },
    )

    return result.confidence, result.scores


# =========================================================================
# 8-MODEL CONVENIENCE FUNCTIONS
# =========================================================================


async def verify_audio_output(
    text: str,
    target_language: str = "hi",
    max_iterations: int = 3,
) -> tuple[str, float, str | None]:
    """
    Verify TTS audio accuracy using Whisper transcription.

    Pipeline: MMS-TTS → Whisper → BGE-M3 similarity check

    Args:
        text: Text to convert to audio
        target_language: Target language code
        max_iterations: Max verification attempts

    Returns:
        Tuple of (original_text, audio_accuracy, audio_path)
    """
    collaborator = get_model_collaborator()

    result = await collaborator.collaborate(
        task="audio_verify",
        input_text=text,
        pattern=CollaborationPattern.AUDIO_VERIFY,
        context={
            "target_language": target_language,
            "max_audio_iterations": max_iterations,
        },
    )

    audio_path = result.metadata.get("audio_path")
    return result.final_output, result.confidence, audio_path


async def process_document(
    image_path: str,
    target_language: str | None = None,
    grade_level: int = 8,
    generate_audio: bool = False,
) -> tuple[str, float, dict[str, Any]]:
    """
    Full document processing: OCR → Simplify → Translate → Audio

    Pipeline: GOT-OCR2 → Qwen3-8B → IndicTrans2 → MMS-TTS

    Args:
        image_path: Path to document/image
        target_language: Optional target language for translation
        grade_level: Target grade level for simplification
        generate_audio: Whether to generate audio output

    Returns:
        Tuple of (processed_text, confidence, metadata)
    """
    collaborator = get_model_collaborator()

    result = await collaborator.collaborate(
        task="document_chain",
        input_text="",  # Will be extracted from image
        pattern=CollaborationPattern.DOCUMENT_CHAIN,
        context={
            "image_path": image_path,
            "target_language": target_language,
            "grade_level": grade_level,
            "generate_audio": generate_audio,
        },
    )

    return result.final_output, result.confidence, result.metadata


async def generate_best_output(
    text: str,
    task: str = "simplify",
    num_candidates: int = 3,
    grade_level: int = 8,
    subject: str = "General",
) -> tuple[str, float, dict[str, float]]:
    """
    Generate multiple candidates and select the best using reranking.

    Pipeline: Qwen3-8B (3x) → BGE-Reranker → BGE-M3 verification

    Args:
        text: Input text
        task: Task type (simplify, translate, etc.)
        num_candidates: Number of candidates to generate
        grade_level: Target grade level
        subject: Subject area

    Returns:
        Tuple of (best_output, confidence, all_scores)
    """
    collaborator = get_model_collaborator()

    result = await collaborator.collaborate(
        task=task,
        input_text=text,
        pattern=CollaborationPattern.RERANK,
        context={
            "num_candidates": num_candidates,
            "grade_level": grade_level,
            "subject": subject,
        },
    )

    return result.final_output, result.confidence, result.scores


async def full_educational_pipeline(
    content: str,
    source_language: str = "en",
    target_language: str = "hi",
    grade_level: int = 8,
    subject: str = "General",
    generate_audio: bool = True,
    verify_all_steps: bool = True,
) -> dict[str, Any]:
    """
    Complete educational content processing with all 7 models.

    Full Pipeline:
    1. Qwen3-8B: Simplify content for grade level
    2. BGE-Reranker: Select best simplification
    3. IndicTrans2: Translate to target language
    4. Back-translate verification
    5. BGE-M3: Semantic preservation check
    6. Qwen3-8B: Quality validation
    7. MMS-TTS: Generate audio
    8. Whisper: Verify audio accuracy

    Args:
        content: Original educational content
        source_language: Source language code
        target_language: Target language code
        grade_level: Target grade level
        subject: Subject area
        generate_audio: Generate audio output
        verify_all_steps: Run verification at each step

    Returns:
        Dict with all outputs and quality metrics
    """
    # Initialize collaborator to ensure it's available
    _ = get_model_collaborator()

    results: dict[str, Any] = {
        "original": content,
        "source_language": source_language,
        "target_language": target_language,
        "grade_level": grade_level,
        "subject": subject,
        "steps": {},
        "overall_confidence": 0.0,
        "models_used": [],
        "audio_path": None,
    }

    current_text = content
    all_models: set = set()
    confidences: list = []

    # Step 1: Simplify with reranking
    try:
        simplified, conf, scores = await generate_best_output(
            text=current_text,
            task="simplify",
            num_candidates=3,
            grade_level=grade_level,
            subject=subject,
        )
        current_text = simplified
        results["steps"]["simplification"] = {
            "output": simplified,
            "confidence": conf,
            "scores": scores,
        }
        confidences.append(conf)
        all_models.add("Qwen3-8B")
        all_models.add("bge-reranker")
    except Exception as e:
        results["steps"]["simplification"] = {"error": str(e)}

    # Step 2: Translate with back-translation verification
    if target_language and target_language != source_language:
        try:
            translated, conf, verified = await collaborate_and_translate(
                text=current_text,
                target_language=target_language,
                verify_quality=verify_all_steps,
            )
            current_text = translated
            results["steps"]["translation"] = {
                "output": translated,
                "confidence": conf,
                "verified": verified,
            }
            confidences.append(conf)
            all_models.add("indicTrans2")
            all_models.add("bge-m3")
        except Exception as e:
            results["steps"]["translation"] = {"error": str(e)}

    # Step 3: Quality validation
    try:
        eval_conf, eval_scores = await ensemble_evaluate(
            original_text=content,
            processed_text=current_text,
            grade_level=grade_level,
            subject=subject,
        )
        results["steps"]["validation"] = {
            "confidence": eval_conf,
            "scores": eval_scores,
        }
        confidences.append(eval_conf)
        all_models.add("qwen3-8b")
    except Exception as e:
        results["steps"]["validation"] = {"error": str(e)}

    # Step 4: Generate and verify audio
    if generate_audio:
        try:
            _, audio_conf, audio_path = await verify_audio_output(
                text=current_text,
                target_language=target_language,
                max_iterations=3,
            )
            results["steps"]["audio"] = {
                "confidence": audio_conf,
                "path": audio_path,
            }
            results["audio_path"] = audio_path
            confidences.append(audio_conf)
            all_models.add("mms-tts")
            all_models.add("whisper")
        except Exception as e:
            results["steps"]["audio"] = {"error": str(e)}

    # Final output
    results["final_output"] = current_text
    results["models_used"] = list(all_models)
    results["overall_confidence"] = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )

    return results
