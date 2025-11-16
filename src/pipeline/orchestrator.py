"""Content Pipeline Orchestrator for sequential stage execution."""
import time
import logging
from typing import Union, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .model_clients import FlanT5Client, IndicTrans2Client, BERTClient, VITSClient
from ..database import get_db
from ..models import ProcessedContent, PipelineLog


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages."""
    SIMPLIFICATION = "simplification"
    TRANSLATION = "translation"
    VALIDATION = "validation"
    SPEECH = "speech"


class ProcessingStatus(Enum):
    """Processing status values."""
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


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


class PipelineValidationError(Exception):
    """Raised when pipeline input validation fails."""
    pass


class PipelineStageError(Exception):
    """Raised when a pipeline stage fails after retries."""
    pass


class ContentPipelineOrchestrator:
    """
    Orchestrates the four-stage content processing pipeline:
    1. Simplification (Flan-T5)
    2. Translation (IndicTrans2)
    3. Validation (BERT)
    4. Speech Generation (VITS/Bhashini)
    """
    
    # Supported languages
    SUPPORTED_LANGUAGES = ['Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi']
    
    # Supported subjects
    SUPPORTED_SUBJECTS = ['Mathematics', 'Science', 'Social Studies', 'English', 'History', 'Geography']
    
    # Supported output formats
    SUPPORTED_FORMATS = ['text', 'audio', 'both']
    
    # Grade level range
    MIN_GRADE = 5
    MAX_GRADE = 12
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2
    
    # Quality thresholds
    NCERT_ALIGNMENT_THRESHOLD = 0.80
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the pipeline orchestrator with model clients.
        
        Args:
            api_key: Optional Hugging Face API key
        """
        self.flant5_client = FlanT5Client(api_key)
        self.indictrans2_client = IndicTrans2Client(api_key)
        self.bert_client = BERTClient(api_key)
        self.vits_client = VITSClient(api_key)
        
        # Import SpeechGenerator locally to avoid circular imports
        from ..speech import SpeechGenerator
        self.speech_generator = SpeechGenerator()
        
        self.metrics: list[StageMetrics] = []
        
        logger.info("ContentPipelineOrchestrator initialized")
    
    def process_content(
        self,
        input_data: Union[str, bytes],
        target_language: str,
        grade_level: int,
        subject: str,
        output_format: str = 'both'
    ) -> ProcessedContentResult:
        """
        Process content through the complete pipeline.
        
        Args:
            input_data: Raw text or PDF content
            target_language: Target Indian language for translation
            grade_level: Grade level (5-12) for content adaptation
            subject: Subject area (Mathematics, Science, etc.)
            output_format: Output format ('text', 'audio', 'both')
        
        Returns:
            ProcessedContentResult with all processed content and metrics
        
        Raises:
            PipelineValidationError: If input parameters are invalid
            PipelineStageError: If a stage fails after max retries
        """
        # Validate parameters
        self.validate_parameters(input_data, target_language, grade_level, subject, output_format)
        
        # Reset metrics for this processing run
        self.metrics = []
        
        # Convert bytes to string if needed (PDF processing would go here)
        if isinstance(input_data, bytes):
            input_data = input_data.decode('utf-8')
        
        original_text = input_data
        
        logger.info(f"Starting pipeline processing: language={target_language}, grade={grade_level}, subject={subject}")
        
        try:
            # Stage 1: Simplification
            simplified_text = self._execute_stage_with_retry(
                PipelineStage.SIMPLIFICATION,
                self._simplify_text,
                original_text,
                grade_level,
                subject
            )
            
            # Stage 2: Translation
            translated_text = self._execute_stage_with_retry(
                PipelineStage.TRANSLATION,
                self._translate_text,
                simplified_text,
                target_language
            )
            
            # Stage 3: Validation
            ncert_alignment_score = self._execute_stage_with_retry(
                PipelineStage.VALIDATION,
                self._validate_content,
                original_text,
                translated_text,
                grade_level,
                subject
            )
            
            # Stage 4: Speech Generation (if requested)
            audio_file_path = None
            audio_accuracy_score = None
            
            if output_format in ['audio', 'both']:
                try:
                    audio_file_path, audio_accuracy_score = self._execute_stage_with_retry(
                        PipelineStage.SPEECH,
                        self._generate_speech,
                        translated_text,
                        target_language,
                        subject
                    )
                except PipelineStageError as e:
                    # Speech generation is non-critical - log warning and continue
                    logger.warning(f"Speech generation skipped: {e}")
                    logger.info("Continuing pipeline without audio output")
            
            # Store processed content in database
            content_id = self._store_content(
                original_text=original_text,
                simplified_text=simplified_text,
                translated_text=translated_text,
                language=target_language,
                grade_level=grade_level,
                subject=subject,
                audio_file_path=audio_file_path,
                ncert_alignment_score=ncert_alignment_score,
                audio_accuracy_score=audio_accuracy_score
            )
            
            # Log all metrics to database
            self._log_metrics(content_id)
            
            logger.info(f"Pipeline processing completed successfully: content_id={content_id}")
            
            return ProcessedContentResult(
                id=str(content_id),
                original_text=original_text,
                simplified_text=simplified_text,
                translated_text=translated_text,
                language=target_language,
                grade_level=grade_level,
                subject=subject,
                audio_file_path=audio_file_path,
                ncert_alignment_score=ncert_alignment_score,
                audio_accuracy_score=audio_accuracy_score,
                validation_status="passed" if ncert_alignment_score >= self.NCERT_ALIGNMENT_THRESHOLD else "failed",
                created_at=datetime.now(timezone.utc),
                metadata={
                    'output_format': output_format,
                    'total_processing_time_ms': sum(m.processing_time_ms for m in self.metrics)
                },
                metrics=self.metrics
            )
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            raise
    
    def validate_parameters(
        self,
        input_data: Union[str, bytes],
        target_language: str,
        grade_level: int,
        subject: str,
        output_format: str
    ) -> None:
        """
        Validate input parameters for pipeline processing.
        
        Args:
            input_data: Raw content to process
            target_language: Target language
            grade_level: Grade level
            subject: Subject area
            output_format: Output format
        
        Raises:
            PipelineValidationError: If any parameter is invalid
        """
        errors = []
        
        # Validate input_data
        if not input_data:
            errors.append("input_data cannot be empty")
        elif isinstance(input_data, str) and len(input_data.strip()) == 0:
            errors.append("input_data cannot be empty or whitespace only")
        
        # Validate target_language
        if target_language not in self.SUPPORTED_LANGUAGES:
            errors.append(
                f"target_language must be one of {self.SUPPORTED_LANGUAGES}, got '{target_language}'"
            )
        
        # Validate grade_level
        if not isinstance(grade_level, int):
            errors.append(f"grade_level must be an integer, got {type(grade_level).__name__}")
        elif grade_level < self.MIN_GRADE or grade_level > self.MAX_GRADE:
            errors.append(
                f"grade_level must be between {self.MIN_GRADE} and {self.MAX_GRADE}, got {grade_level}"
            )
        
        # Validate subject
        if subject not in self.SUPPORTED_SUBJECTS:
            errors.append(
                f"subject must be one of {self.SUPPORTED_SUBJECTS}, got '{subject}'"
            )
        
        # Validate output_format
        if output_format not in self.SUPPORTED_FORMATS:
            errors.append(
                f"output_format must be one of {self.SUPPORTED_FORMATS}, got '{output_format}'"
            )
        
        if errors:
            error_message = "Parameter validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            logger.error(error_message)
            raise PipelineValidationError(error_message)
        
        logger.info("Parameter validation passed")
    
    def _execute_stage_with_retry(
        self,
        stage: PipelineStage,
        stage_function,
        *args,
        **kwargs
    ):
        """
        Execute a pipeline stage with retry logic and exponential backoff.
        
        Args:
            stage: Pipeline stage being executed
            stage_function: Function to execute
            *args: Arguments for the stage function
            **kwargs: Keyword arguments for the stage function
        
        Returns:
            Result from the stage function
        
        Raises:
            PipelineStageError: If stage fails after max retries
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= self.MAX_RETRIES:
            start_time = time.time()
            
            try:
                logger.info(f"Executing stage: {stage.value} (attempt {retry_count + 1}/{self.MAX_RETRIES + 1})")
                
                result = stage_function(*args, **kwargs)
                
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Track successful metrics
                metrics = StageMetrics(
                    stage=stage.value,
                    processing_time_ms=processing_time_ms,
                    success=True,
                    retry_count=retry_count
                )
                self.metrics.append(metrics)
                
                logger.info(f"Stage {stage.value} completed successfully in {processing_time_ms}ms")
                
                return result
                
            except Exception as e:
                processing_time_ms = int((time.time() - start_time) * 1000)
                last_error = e
                
                logger.warning(f"Stage {stage.value} failed (attempt {retry_count + 1}): {str(e)}")
                
                if retry_count < self.MAX_RETRIES:
                    # Calculate exponential backoff
                    backoff_time = self.RETRY_BACKOFF_BASE ** retry_count
                    logger.info(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    retry_count += 1
                else:
                    # Max retries reached, log failure and raise
                    metrics = StageMetrics(
                        stage=stage.value,
                        processing_time_ms=processing_time_ms,
                        success=False,
                        error_message=str(e),
                        retry_count=retry_count
                    )
                    self.metrics.append(metrics)
                    
                    error_msg = f"Stage {stage.value} failed after {self.MAX_RETRIES + 1} attempts: {str(e)}"
                    logger.error(error_msg)
                    raise PipelineStageError(error_msg) from last_error
        
        # Should never reach here, but just in case
        raise PipelineStageError(f"Stage {stage.value} failed unexpectedly")
    
    def _simplify_text(self, text: str, grade_level: int, subject: str) -> str:
        """
        Simplify text using Flan-T5 model.
        
        Args:
            text: Original text to simplify
            grade_level: Target grade level
            subject: Subject area
        
        Returns:
            Simplified text
        """
        logger.debug(f"Simplifying text for grade {grade_level}, subject {subject}")
        simplified = self.flant5_client.process(text, grade_level, subject)
        
        if not simplified or len(simplified.strip()) == 0:
            raise ValueError("Simplification produced empty result")
        
        return simplified
    
    def _translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text using IndicTrans2 model.
        
        Args:
            text: Text to translate
            target_language: Target Indian language
        
        Returns:
            Translated text
        """
        logger.debug(f"Translating text to {target_language}")
        translated = self.indictrans2_client.process(text, target_language)
        
        if not translated or len(translated.strip()) == 0:
            raise ValueError("Translation produced empty result")
        
        return translated
    
    def _validate_content(
        self,
        original_text: str,
        translated_text: str,
        grade_level: int,
        subject: str
    ) -> float:
        """
        Validate content using BERT model for semantic accuracy.
        
        Args:
            original_text: Original source text
            translated_text: Translated text to validate
            grade_level: Grade level
            subject: Subject area
        
        Returns:
            NCERT alignment score (0-1)
        
        Raises:
            ValueError: If validation score is below threshold
        """
        logger.debug(f"Validating content for grade {grade_level}, subject {subject}")
        
        # Calculate semantic similarity between original and translated
        similarity_score = self.bert_client.process(original_text, translated_text)
        
        # For now, use similarity as NCERT alignment score
        # In production, this would check against NCERT standards database
        ncert_alignment_score = similarity_score
        
        logger.info(f"NCERT alignment score: {ncert_alignment_score:.2f}")
        
        if ncert_alignment_score < self.NCERT_ALIGNMENT_THRESHOLD:
            raise ValueError(
                f"Content validation failed: NCERT alignment score {ncert_alignment_score:.2f} "
                f"is below threshold {self.NCERT_ALIGNMENT_THRESHOLD}"
            )
        
        return ncert_alignment_score
    
    def _generate_speech(
        self,
        text: str,
        language: str,
        subject: str
    ) -> tuple[str, float]:
        """
        Generate speech audio using SpeechGenerator with optimization and validation.
        
        Args:
            text: Text to convert to speech
            language: Target language
            subject: Subject area (for technical term handling)
        
        Returns:
            Tuple of (audio_file_path, audio_accuracy_score)
        """
        logger.debug(f"Generating speech for {language} in {subject}")
        
        # Use the SpeechGenerator component
        audio_file = self.speech_generator.generate_speech(text, language, subject)
        
        if not audio_file or not audio_file.content:
            raise ValueError("Speech generation produced empty audio")
        
        # Validate audio quality
        if not self.speech_generator.validate_audio_quality(audio_file):
            logger.warning("Generated audio did not meet quality requirements")
        
        logger.info(f"Speech generated successfully: {audio_file.size_mb:.2f}MB, accuracy: {audio_file.accuracy_score:.2%}")
        
        return audio_file.file_path, audio_file.accuracy_score or 0.90
    
    def _store_content(
        self,
        original_text: str,
        simplified_text: str,
        translated_text: str,
        language: str,
        grade_level: int,
        subject: str,
        audio_file_path: Optional[str],
        ncert_alignment_score: float,
        audio_accuracy_score: Optional[float]
    ) -> str:
        """
        Store processed content in the database.
        
        Args:
            original_text: Original input text
            simplified_text: Simplified text
            translated_text: Translated text
            language: Target language
            grade_level: Grade level
            subject: Subject area
            audio_file_path: Path to audio file
            ncert_alignment_score: NCERT alignment score
            audio_accuracy_score: Audio accuracy score
        
        Returns:
            Content ID (UUID)
        """
        session = get_db().get_session()
        
        try:
            content = ProcessedContent(
                original_text=original_text,
                simplified_text=simplified_text,
                translated_text=translated_text,
                language=language,
                grade_level=grade_level,
                subject=subject,
                audio_file_path=audio_file_path,
                ncert_alignment_score=ncert_alignment_score,
                audio_accuracy_score=audio_accuracy_score,
                content_metadata={
                    'pipeline_version': '1.0',
                    'processing_timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            session.add(content)
            session.commit()
            
            content_id = content.id
            logger.info(f"Content stored with ID: {content_id}")
            
            return content_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store content: {str(e)}")
            raise
        finally:
            session.close()
    
    def _log_metrics(self, content_id: str) -> None:
        """
        Log all pipeline metrics to the database.
        
        Args:
            content_id: ID of the processed content
        """
        session = get_db().get_session()
        
        try:
            for metric in self.metrics:
                log_entry = PipelineLog(
                    content_id=content_id,
                    stage=metric.stage,
                    status=ProcessingStatus.SUCCESS.value if metric.success else ProcessingStatus.FAILED.value,
                    processing_time_ms=metric.processing_time_ms,
                    error_message=metric.error_message,
                    timestamp=metric.timestamp
                )
                session.add(log_entry)
            
            session.commit()
            logger.info(f"Logged {len(self.metrics)} metrics for content {content_id}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log metrics: {str(e)}")
            # Don't raise - metrics logging failure shouldn't fail the pipeline
        finally:
            session.close()
    
    def track_metrics(self, stage: str, duration_ms: int, success: bool) -> None:
        """
        Track metrics for a pipeline stage (public method for external use).
        
        Args:
            stage: Stage name
            duration_ms: Processing duration in milliseconds
            success: Whether the stage succeeded
        """
        metrics = StageMetrics(
            stage=stage,
            processing_time_ms=duration_ms,
            success=success
        )
        self.metrics.append(metrics)
        logger.debug(f"Tracked metrics for stage {stage}: {duration_ms}ms, success={success}")
