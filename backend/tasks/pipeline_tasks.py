"""Pipeline tasks for async content processing."""

import logging
import time
from typing import Any, Dict, List

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from ..models import ProcessedContent
from ..services.ocr import OCRService
from ..services.simplify.simplifier import TextSimplifier
from ..services.speech.generator import SpeechGenerator
from ..services.translate.engine import TranslationEngine
from ..services.validate.validator import ValidationModule
from .celery_app import celery_app

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task with callbacks for progress tracking."""

    def update_progress(
        self, task_id: str, progress: int, stage: str, message: str = ""
    ):
        """Update task progress."""
        self.update_state(
            task_id=task_id,
            state="PROGRESS",
            meta={
                "progress": progress,
                "stage": stage,
                "message": message,
                "timestamp": time.time(),
            },
        )


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="pipeline.extract_text",
    soft_time_limit=600,
    time_limit=900,
    max_retries=3,
)
def extract_text_task(
    self, file_path: str, languages: list[str] | None = None
) -> dict[str, Any]:
    """
    Extract text from PDF/image file.

    Args:
        file_path: Path to file
        languages: OCR languages to use

    Returns:
        Dict with extracted text and metadata
    """
    try:
        self.update_progress(self.request.id, 10, "extraction", "Starting OCR...")

        # Initialize OCR service
        ocr_service = OCRService(languages=languages or ["English", "Hindi"])

        self.update_progress(
            self.request.id, 30, "extraction", "Processing document..."
        )

        # Extract text
        result = ocr_service.extract_text(file_path)

        # Validate
        if not ocr_service.validate_extraction(result):
            raise ValueError("Extraction validation failed")

        self.update_progress(self.request.id, 100, "extraction", "Extraction complete")

        return {
            "text": result.text,
            "num_pages": result.num_pages,
            "has_formulas": result.has_formulas,
            "formula_blocks": result.formula_blocks,
            "confidence": result.confidence,
            "metadata": result.metadata,
        }

    except SoftTimeLimitExceeded:
        logger.error(f"Text extraction timed out for {file_path}")
        raise
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="pipeline.simplify_text",
    soft_time_limit=300,
    time_limit=600,
    max_retries=3,
)
def simplify_text_task(
    self,
    text: str,
    grade_level: int,
    subject: str,
    formula_blocks: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Simplify text for target grade level.

    Args:
        text: Input text
        grade_level: Target grade (5-12)
        subject: Subject area
        formula_blocks: Preserved formulas

    Returns:
        Dict with simplified text
    """
    try:
        self.update_progress(self.request.id, 10, "simplification", "Loading model...")

        # Initialize simplifier with Qwen model client
        import os

        from ..pipeline.model_clients import QwenSimplificationClient

        api_key = os.getenv("HUGGINGFACE_API_KEY")
        model_client = None
        if api_key:
            try:
                model_client = QwenSimplificationClient(api_key)
                logger.info("Qwen2.5-3B model client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Qwen client: {e}")

        simplifier = TextSimplifier(model_client=model_client)

        self.update_progress(
            self.request.id, 40, "simplification", "Simplifying content..."
        )

        # Simplify
        simplified_result = simplifier.simplify_text(
            content=text, grade_level=grade_level, subject=subject
        )
        simplified = simplified_result.text

        # Restore formulas if present
        if formula_blocks:
            from ..services.ocr import MathFormulaDetector

            simplified = MathFormulaDetector.restore_formulas(
                simplified, formula_blocks
            )

        self.update_progress(
            self.request.id, 100, "simplification", "Simplification complete"
        )

        return {
            "simplified_text": simplified,
            "grade_level": grade_level,
            "subject": subject,
            "complexity_score": simplified_result.complexity_score,
        }

    except SoftTimeLimitExceeded:
        logger.error("Text simplification timed out")
        raise
    except Exception as e:
        logger.error(f"Text simplification failed: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="pipeline.translate_text",
    soft_time_limit=300,
    time_limit=600,
    max_retries=3,
)
def translate_text_task(
    self,
    text: str,
    target_languages: list[str],
    formula_blocks: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Translate text to multiple languages.

    Args:
        text: Input text
        target_languages: List of target languages
        formula_blocks: Preserved formulas

    Returns:
        Dict with translations
    """
    try:
        self.update_progress(
            self.request.id, 10, "translation", "Loading translation model..."
        )

        # Initialize translator with model client
        import os

        from ..pipeline.model_clients import IndicTrans2Client

        api_key = os.getenv("HUGGINGFACE_API_KEY")
        model_client = None
        if api_key:
            try:
                model_client = IndicTrans2Client(api_key)
                logger.info("IndicTrans2 model client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize IndicTrans2 client: {e}")

        translator = TranslationEngine(model_client=model_client)

        translations = {}
        progress_step = 80 / len(target_languages)

        for i, language in enumerate(target_languages):
            self.update_progress(
                self.request.id,
                10 + int((i + 1) * progress_step),
                "translation",
                f"Translating to {language}...",
            )

            # Translate
            translated_result = translator.translate(
                text=text,
                target_language=language,
                subject="General",
                source_language="English",
            )

            # TranslatedText dataclass has .text attribute
            if hasattr(translated_result, "text"):
                translated = translated_result.text
            elif isinstance(translated_result, dict):
                translated = translated_result.get(
                    "text", translated_result.get("translated_text", "")
                )
            else:
                translated = str(translated_result)

            # Restore formulas
            if formula_blocks:
                from ..services.ocr import MathFormulaDetector

                translated = MathFormulaDetector.restore_formulas(
                    translated, formula_blocks
                )

            translations[language] = translated

        self.update_progress(
            self.request.id, 100, "translation", "Translation complete"
        )

        return {"translations": translations, "languages": target_languages}

    except SoftTimeLimitExceeded:
        logger.error("Translation timed out")
        raise
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="pipeline.validate_content",
    soft_time_limit=180,
    time_limit=300,
    max_retries=2,
)
def validate_content_task(
    self, original_text: str, processed_text: str, threshold: float = 0.80
) -> dict[str, Any]:
    """
    Validate semantic similarity between original and processed text.

    Args:
        original_text: Original text
        processed_text: Processed/simplified text
        threshold: Minimum similarity score

    Returns:
        Dict with validation results
    """
    try:
        self.update_progress(self.request.id, 20, "validation", "Loading validator...")

        # Initialize validator
        validator = ValidationModule()

        self.update_progress(
            self.request.id, 60, "validation", "Computing similarity..."
        )

        # Validate
        validation_result = validator.validate_content(
            original_text=original_text,
            translated_text=processed_text,
            grade_level=5,
            subject="General",
            language="English",
        )

        similarity_score = validation_result.get("similarity_score", 0.0)
        is_valid = similarity_score >= threshold

        self.update_progress(self.request.id, 100, "validation", "Validation complete")

        return {
            "similarity_score": similarity_score,
            "threshold": threshold,
            "is_valid": is_valid,
            "requires_review": not is_valid,
            "validation_details": validation_result,
        }

    except Exception as e:
        logger.error(f"Content validation failed: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="pipeline.generate_audio",
    soft_time_limit=600,
    time_limit=900,
    max_retries=3,
)
def generate_audio_task(
    self, text: str, language: str, content_id: str
) -> dict[str, Any]:
    """
    Generate audio from text.

    Args:
        text: Input text
        language: Target language
        content_id: Content ID for file naming

    Returns:
        Dict with audio file path and metadata
    """
    try:
        self.update_progress(
            self.request.id, 20, "audio", f"Generating {language} audio..."
        )

        # Initialize speech generator
        speech_gen = SpeechGenerator()

        self.update_progress(self.request.id, 60, "audio", "Processing audio...")

        # Generate audio
        audio_file = speech_gen.generate_speech(
            text=text, language=language, subject="General"
        )

        # Build result dict - AudioFile has file_path, duration_seconds, accuracy_score
        audio_result = {
            "audio_path": audio_file.file_path,
            "audio_url": f"/api/v2/content/audio/{content_id}",
            "duration": audio_file.duration_seconds,
            "accuracy_score": audio_file.accuracy_score or 0.85,
        }

        self.update_progress(self.request.id, 100, "audio", "Audio generation complete")

        return {
            "audio_path": audio_result["audio_path"],
            "audio_url": audio_result.get("audio_url"),
            "duration": audio_result.get("duration"),
            "language": language,
            "accuracy_score": audio_result.get("accuracy_score", 0.85),
        }

    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="pipeline.full_pipeline",
    soft_time_limit=1500,
    time_limit=1800,
    max_retries=2,
)
def full_pipeline_task(
    self,
    file_path: str,
    grade_level: int,
    subject: str,
    target_languages: list[str],
    output_format: str = "both",
    validation_threshold: float = 0.80,
) -> dict[str, Any]:
    """
    Execute full content processing pipeline.

    Args:
        file_path: Path to input file
        grade_level: Target grade level
        subject: Subject area
        target_languages: List of target languages
        output_format: 'text', 'audio', or 'both'
        validation_threshold: Minimum similarity threshold

    Returns:
        Complete processing results
    """
    try:
        task_id = self.request.id
        logger.info(f"Starting full pipeline task {task_id}")

        # Stage 1: Extract text (0-20%)
        self.update_progress(task_id, 5, "pipeline", "Starting extraction...")
        extraction_result = extract_text_task(file_path)

        original_text = extraction_result["text"]
        formula_blocks = extraction_result.get("formula_blocks", [])

        # Stage 2: Simplify (20-40%)
        self.update_progress(task_id, 25, "pipeline", "Simplifying content...")
        simplification_result = simplify_text_task(
            text=original_text,
            grade_level=grade_level,
            subject=subject,
            formula_blocks=formula_blocks,
        )

        simplified_text = simplification_result["simplified_text"]

        # Stage 3: Validate (40-50%)
        self.update_progress(task_id, 45, "pipeline", "Validating content...")
        validation_result = validate_content_task(
            original_text=original_text,
            processed_text=simplified_text,
            threshold=validation_threshold,
        )

        # Stage 4: Translate (50-70%)
        self.update_progress(task_id, 55, "pipeline", "Translating...")
        translation_result = translate_text_task(
            text=simplified_text,
            target_languages=target_languages,
            formula_blocks=formula_blocks,
        )

        # Stage 5: Generate audio (70-100%)
        audio_results = {}

        if output_format in ["audio", "both"]:
            self.update_progress(task_id, 75, "pipeline", "Generating audio...")

            # Generate audio for each language
            for lang in target_languages:
                audio_result = generate_audio_task(
                    text=translation_result["translations"][lang],
                    language=lang,
                    content_id=task_id,
                )
                audio_results[lang] = audio_result

        # Save to database
        self.update_progress(task_id, 95, "pipeline", "Saving results...")

        from ..database import get_db_session

        with get_db_session() as session:
            try:
                # Create content record for each language
                content_records = []

                for lang in target_languages:
                    content = ProcessedContent(
                        original_text=original_text,
                        simplified_text=simplified_text,
                        translated_text=translation_result["translations"][lang],
                        language=lang,
                        grade_level=grade_level,
                        subject=subject,
                        audio_file_path=audio_results.get(lang, {}).get("audio_path"),
                        ncert_alignment_score=validation_result["similarity_score"],
                        audio_accuracy_score=audio_results.get(lang, {}).get(
                            "accuracy_score"
                        ),
                        content_metadata={
                            "extraction": extraction_result["metadata"],
                            "formulas": len(formula_blocks),
                            "validation": validation_result,
                            "task_id": task_id,
                        },
                    )
                    session.add(content)
                    content_records.append(content)

                session.flush()

                # Get content IDs
                content_ids = {
                    record.language: str(record.id) for record in content_records
                }

            except Exception as e:
                logger.error(f"Failed to save content: {e}")
                raise

        self.update_progress(task_id, 100, "pipeline", "Pipeline complete!")

        # Build final result
        return {
            "status": "completed",
            "task_id": task_id,
            "result": {
                "content_ids": content_ids,
                "original_text": original_text[:500] + "..."
                if len(original_text) > 500
                else original_text,
                "simplified_text": simplified_text[:500] + "..."
                if len(simplified_text) > 500
                else simplified_text,
                "translations": {
                    lang: text[:500] + "..." if len(text) > 500 else text
                    for lang, text in translation_result["translations"].items()
                },
                "audio": {
                    lang: {
                        "url": result.get("audio_url"),
                        "duration": result.get("duration"),
                    }
                    for lang, result in audio_results.items()
                },
                "metadata": {
                    "grade_level": grade_level,
                    "subject": subject,
                    "num_pages": extraction_result["num_pages"],
                    "has_formulas": extraction_result["has_formulas"],
                    "similarity_score": validation_result["similarity_score"],
                    "requires_review": validation_result["requires_review"],
                    "extraction_confidence": extraction_result["confidence"],
                },
            },
        }

    except Exception as e:
        logger.error(f"Full pipeline failed: {e}", exc_info=True)
        raise


@celery_app.task(name="pipeline.cleanup_old_results")
def cleanup_old_results():
    """Periodic task to clean up old task results from Redis."""
    try:
        # Clean up results older than 24 hours
        # This is handled automatically by Celery's result_expires setting
        logger.info("Cleanup task executed")
        return {"status": "cleaned"}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


# Export
__all__ = [
    "cleanup_old_results",
    "extract_text_task",
    "full_pipeline_task",
    "generate_audio_task",
    "simplify_text_task",
    "translate_text_task",
    "validate_content_task",
]
