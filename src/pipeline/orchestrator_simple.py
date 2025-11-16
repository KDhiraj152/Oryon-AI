"""
Simplified Content Pipeline Orchestrator.

Replaces complex retry logic with tenacity library.
Reduction: ~600 lines → ~400 lines (33% reduction)
"""

import time
import logging
from typing import Union, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .model_clients import FlanT5Client, IndicTrans2Client, BERTClient, VITSClient
from ..database import get_db_instance
from ..models import ProcessedContent, PipelineLog

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages."""
    SIMPLIFICATION = "simplification"
    TRANSLATION = "translation"
    VALIDATION = "validation"
    SPEECH = "speech"


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    stage: str
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProcessedContentResult:
    """Result of content processing through the pipeline."""
    id: str
    original_text: str
    simplified_text: str
    translated_text: str
    language: str
    grade_level: int
    subject: str
    audio_file_path: Optional[str]
    ncert_alignment_score: float
    audio_accuracy_score: Optional[float]
    validation_status: str
    created_at: datetime
    metadata: Dict[str, Any]
    metrics: list[StageMetrics]


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class ContentPipelineOrchestrator:
    """
    Orchestrates the four-stage content processing pipeline:
    1. Simplification → 2. Translation → 3. Validation → 4. Speech
    
    Simplified with tenacity for automatic retries.
    """
    
    SUPPORTED_LANGUAGES = ['Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi']
    SUPPORTED_SUBJECTS = ['Mathematics', 'Science', 'Social Studies', 'English', 'History', 'Geography']
    SUPPORTED_FORMATS = ['text', 'audio', 'both']
    MIN_GRADE, MAX_GRADE = 5, 12
    NCERT_ALIGNMENT_THRESHOLD = 0.80
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize pipeline with model clients."""
        self.flant5_client = FlanT5Client(api_key)
        self.indictrans2_client = IndicTrans2Client(api_key)
        self.bert_client = BERTClient(api_key)
        self.vits_client = VITSClient(api_key)
        
        from ..speech import SpeechGenerator
        self.speech_generator = SpeechGenerator()
        
        self.metrics: list[StageMetrics] = []
        logger.info("Pipeline initialized")
    
    def validate_parameters(self, text: str, language: str, grade: int, subject: str, format: str):
        """Validate input parameters."""
        if not text or len(text.strip()) < 10:
            raise PipelineError("Input text too short (min 10 characters)")
        
        if language not in self.SUPPORTED_LANGUAGES:
            raise PipelineError(f"Unsupported language: {language}")
        
        if not (self.MIN_GRADE <= grade <= self.MAX_GRADE):
            raise PipelineError(f"Grade must be between {self.MIN_GRADE}-{self.MAX_GRADE}")
        
        if subject not in self.SUPPORTED_SUBJECTS:
            raise PipelineError(f"Unsupported subject: {subject}")
        
        if format not in self.SUPPORTED_FORMATS:
            raise PipelineError(f"Unsupported format: {format}")
    
    # =========================================================================
    # PIPELINE STAGES WITH AUTOMATIC RETRY
    # =========================================================================
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _simplify_text(self, text: str, grade: int, subject: str) -> str:
        """Simplify text for grade level (with auto-retry)."""
        from ..simplifier import TextSimplifier
        
        simplifier = TextSimplifier(model_client=self.flant5_client)
        result = simplifier.simplify(
            text=text,
            grade_level=grade,
            subject=subject
        )
        
        if not result or len(result.simplified_text) < 10:
            raise PipelineError("Simplification produced invalid output")
        
        return result.simplified_text
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _translate_text(self, text: str, language: str) -> str:
        """Translate text to target language (with auto-retry)."""
        from ..translator import TranslationEngine
        
        engine = TranslationEngine(model_client=self.indictrans2_client)
        result = engine.translate(
            text=text,
            target_language=language,
            source_language='English'
        )
        
        if not result or len(result.translated_text) < 5:
            raise PipelineError("Translation produced invalid output")
        
        return result.translated_text
    
    @retry(
        stop=stop_after_attempt(2),  # Less retries for validation
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _validate_content(self, original: str, translated: str, grade: int, subject: str) -> tuple[float, str]:
        """Validate translated content (with auto-retry)."""
        from ..validator import ValidationModule
        
        validator = ValidationModule(model_client=self.bert_client)
        result = validator.validate_content(
            original_text=original,
            translated_text=translated,
            grade_level=grade,
            subject=subject
        )
        
        status = "passed" if result.ncert_alignment_score >= self.NCERT_ALIGNMENT_THRESHOLD else "failed"
        return result.ncert_alignment_score, status
    
    def _generate_speech(self, text: str, language: str) -> Optional[str]:
        """
        Generate speech (optional stage - no retry, allow failure).
        
        Returns audio path or None if fails.
        """
        try:
            result = self.speech_generator.generate_speech(
                text=text,
                language=language,
                use_fallback=True
            )
            return result.audio_file_path if result else None
        except Exception as e:
            logger.warning(f"Speech generation failed (non-critical): {e}")
            return None
    
    # =========================================================================
    # MAIN PROCESSING METHOD
    # =========================================================================
    
    def process_content(
        self,
        input_data: Union[str, bytes],
        target_language: str,
        grade_level: int,
        subject: str,
        output_format: str = 'both'
    ) -> ProcessedContentResult:
        """
        Process content through complete pipeline.
        
        Stages: Simplify → Translate → Validate → Speech (optional)
        """
        # Extract text if bytes
        original_text = input_data if isinstance(input_data, str) else input_data.decode('utf-8')
        
        # Validate
        self.validate_parameters(original_text, target_language, grade_level, subject, output_format)
        
        self.metrics = []
        start_time = time.time()
        
        logger.info(f"Processing: {target_language}, grade {grade_level}, {subject}")
        
        # Stage 1: Simplification
        simplified_text = self._run_stage(
            PipelineStage.SIMPLIFICATION,
            self._simplify_text,
            original_text, grade_level, subject
        )
        
        # Stage 2: Translation
        translated_text = self._run_stage(
            PipelineStage.TRANSLATION,
            self._translate_text,
            simplified_text, target_language
        )
        
        # Stage 3: Validation
        ncert_score, validation_status = self._run_stage(
            PipelineStage.VALIDATION,
            self._validate_content,
            original_text, translated_text, grade_level, subject
        )
        
        # Stage 4: Speech (optional)
        audio_path = None
        if output_format in ['audio', 'both']:
            audio_path = self._run_stage(
                PipelineStage.SPEECH,
                self._generate_speech,
                translated_text, target_language,
                allow_failure=True
            )
        
        # Store in database
        content = self._store_content(
            original_text=original_text,
            simplified_text=simplified_text,
            translated_text=translated_text,
            language=target_language,
            grade_level=grade_level,
            subject=subject,
            audio_file_path=audio_path,
            ncert_alignment_score=ncert_score
        )
        
        # Build result
        total_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Pipeline completed in {total_time}ms, NCERT score: {ncert_score:.2f}")
        
        return ProcessedContentResult(
            id=str(content.id),
            original_text=original_text,
            simplified_text=simplified_text,
            translated_text=translated_text,
            language=target_language,
            grade_level=grade_level,
            subject=subject,
            audio_file_path=audio_path,
            ncert_alignment_score=ncert_score,
            audio_accuracy_score=None,
            validation_status=validation_status,
            created_at=content.created_at,
            metadata={'total_processing_time_ms': total_time},
            metrics=self.metrics
        )
    
    def _run_stage(self, stage: PipelineStage, func, *args, allow_failure=False):
        """Run a pipeline stage with timing and metrics."""
        stage_start = time.time()
        
        try:
            result = func(*args)
            duration_ms = int((time.time() - stage_start) * 1000)
            
            self.metrics.append(StageMetrics(
                stage=stage.value,
                processing_time_ms=duration_ms,
                success=True
            ))
            
            logger.info(f"✓ {stage.value} completed in {duration_ms}ms")
            return result
            
        except Exception as e:
            duration_ms = int((time.time() - stage_start) * 1000)
            
            self.metrics.append(StageMetrics(
                stage=stage.value,
                processing_time_ms=duration_ms,
                success=False,
                error_message=str(e)
            ))
            
            if allow_failure:
                logger.warning(f"✗ {stage.value} failed (allowed): {e}")
                return None
            else:
                logger.error(f"✗ {stage.value} failed: {e}")
                raise PipelineError(f"{stage.value} failed: {e}")
    
    def _store_content(self, **kwargs) -> ProcessedContent:
        """Store processed content in database."""
        db = get_db_instance()
        session = db.get_session()
        
        try:
            content = ProcessedContent(**kwargs)
            session.add(content)
            session.commit()
            session.refresh(content)
            return content
        finally:
            session.close()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        return {
            'total_stages': len(self.metrics),
            'successful_stages': sum(1 for m in self.metrics if m.success),
            'failed_stages': sum(1 for m in self.metrics if not m.success),
            'total_time_ms': sum(m.processing_time_ms for m in self.metrics),
            'stages': [
                {
                    'stage': m.stage,
                    'time_ms': m.processing_time_ms,
                    'success': m.success,
                    'error': m.error_message
                }
                for m in self.metrics
            ]
        }


__all__ = ['ContentPipelineOrchestrator', 'ProcessedContentResult', 'StageMetrics', 'PipelineError']
