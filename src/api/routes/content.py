"""Content processing endpoints."""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import aiofiles
import magic
from fastapi import (
    APIRouter, 
    File, 
    UploadFile, 
    HTTPException, 
    Query, 
    Form,
    Depends
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ...tasks.celery_app import celery_app, get_task_info, revoke_task
from ...tasks.pipeline_tasks import (
    extract_text_task,
    simplify_text_task,
    translate_text_task,
    validate_content_task,
    generate_audio_task,
    full_pipeline_task
)
from ...tasks.qa_tasks import process_document_for_qa_task
from ...utils.input_sanitizer import InputSanitizer
from ...utils.auth import get_current_user, TokenData
from ...database import get_db_session
from ...models import ProcessedContent, Feedback
from ...monitoring import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["content"])

# Upload configuration
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
CHUNK_SIZE = 1024 * 1024  # 1MB

# Input sanitizer
sanitizer = InputSanitizer()


# ==================== Custom Error Classes ====================

class AppError(Exception):
    """Application-specific error."""
    def __init__(self, message: str, code: str, status: int = 500):
        self.message = message
        self.code = code
        self.status = status
        super().__init__(message)


# ==================== Request/Response Models ====================

class ChunkedUploadRequest(BaseModel):
    """Chunked upload metadata."""
    filename: str
    chunk_index: int
    total_chunks: int
    upload_id: str
    checksum: Optional[str] = None


class ProcessRequest(BaseModel):
    """Full pipeline processing request."""
    grade_level: int = Field(ge=5, le=12)
    subject: str
    target_languages: List[str]
    output_format: str = Field(default='both', pattern='^(text|audio|both)$')
    validation_threshold: float = Field(default=0.80, ge=0.0, le=1.0)


class SimplifyRequest(BaseModel):
    """Text simplification request."""
    text: str = Field(min_length=10, max_length=50000)
    grade_level: Optional[int] = Field(None, ge=5, le=12)
    target_grade: Optional[int] = Field(None, ge=5, le=12)  # Backward compatibility
    subject: str = Field(default='General')
    
    def get_grade_level(self) -> int:
        """Get grade level from either field."""
        return self.grade_level or self.target_grade or 8


class TranslateRequest(BaseModel):
    """Translation request."""
    text: str = Field(min_length=10, max_length=50000)
    target_languages: Optional[List[str]] = None
    source_language: Optional[str] = None  # Backward compatibility
    target_language: Optional[str] = None  # Backward compatibility
    subject: str = Field(default='General')
    
    def get_target_languages(self) -> List[str]:
        """Get target languages from either format."""
        if self.target_languages:
            return self.target_languages
        elif self.target_language:
            return [self.target_language]
        return ['Hindi']


class ValidateRequest(BaseModel):
    """Content validation request."""
    text: Optional[str] = Field(None, min_length=10)  # Backward compatibility
    original_text: Optional[str] = Field(None, min_length=10)
    processed_text: Optional[str] = Field(None, min_length=10)
    grade_level: int = Field(ge=5, le=12)
    subject: str
    language: Optional[str] = Field(default='English')
    
    def get_texts(self) -> tuple[str, str]:
        """Get original and processed texts."""
        if self.text:
            # Use text for both if only text is provided
            return (self.text, self.text)
        return (self.original_text or '', self.processed_text or '')


class TTSRequest(BaseModel):
    """Text-to-speech request."""
    text: str = Field(min_length=10, max_length=10000)
    language: str
    subject: str = Field(default='General')


class FeedbackRequest(BaseModel):
    """User feedback request."""
    content_id: str
    rating: int = Field(ge=1, le=5)
    feedback_text: Optional[str] = None
    issue_type: Optional[str] = None


class TaskResponse(BaseModel):
    """Task status response."""
    task_id: str
    state: str
    progress: Optional[int] = None
    stage: Optional[str] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==================== Endpoints ====================

@router.post("/upload/chunked")
async def upload_chunk(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
    chunk_index: Optional[int] = Form(None),
    total_chunks: Optional[int] = Form(None),
    checksum: Optional[str] = Form(None)
):
    """
    Upload file chunk for large file support.
    
    Supports chunked uploads for files >10MB.
    """
    try:
        # Parse metadata from JSON blob or discrete form fields for backward compatibility
        parsed_metadata: Dict[str, Any]
        if metadata:
            parsed_metadata = json.loads(metadata)
        elif None not in (upload_id, chunk_index, total_chunks):
            parsed_metadata = {
                "filename": file.filename or "chunk.bin",
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "upload_id": upload_id,
                "checksum": checksum
            }
        else:
            raise HTTPException(status_code=400, detail="Chunk metadata missing")

        if checksum and "checksum" not in parsed_metadata:
            parsed_metadata["checksum"] = checksum

        chunk_request = ChunkedUploadRequest(**parsed_metadata)
        
        # Create upload directory
        upload_path = UPLOAD_DIR / chunk_request.upload_id
        upload_path.mkdir(exist_ok=True)
        
        # Save chunk
        chunk_path = upload_path / f"chunk_{chunk_request.chunk_index}"
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty chunk received")
        async with aiofiles.open(chunk_path, 'wb') as f:
            await f.write(content)
        
        logger.info(f"Saved chunk {chunk_request.chunk_index}/{chunk_request.total_chunks}")
        
        # If last chunk, reassemble file
        if chunk_request.chunk_index == chunk_request.total_chunks - 1:
            final_path = UPLOAD_DIR / chunk_request.filename
            
            async with aiofiles.open(final_path, 'wb') as outfile:
                for i in range(chunk_request.total_chunks):
                    chunk_path = upload_path / f"chunk_{i}"
                    async with aiofiles.open(chunk_path, 'rb') as infile:
                        await outfile.write(await infile.read())

            # Validate the reconstructed file before confirming success
            async with aiofiles.open(final_path, 'rb') as validated:
                final_bytes = await validated.read()
                sanitizer.validate_file_upload(
                    chunk_request.filename,
                    final_bytes,
                    max_size_mb=100
                )
            
            # Cleanup chunks
            import shutil
            shutil.rmtree(upload_path)
            
            logger.info(f"File reassembled: {final_path}")
            
            return {
                "status": "complete",
                "file_path": str(final_path),
                "filename": chunk_request.filename,
                "message": "Upload complete"
            }
        
        return {
            "status": "chunk_received",
            "chunk_index": chunk_request.chunk_index,
            "total_chunks": chunk_request.total_chunks,
            "message": f"Chunk {chunk_request.chunk_index + 1}/{chunk_request.total_chunks} stored"
        }
        
    except Exception as e:
        logger.error(f"Chunk upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    grade_level: Optional[int] = Form(None),
    subject: Optional[str] = Form(None),
    process_for_qa: bool = Form(True, description="Automatically process for Q&A")
):
    """
    Upload file for processing.
    
    Supports PDF, TXT and common image formats. Optionally extracts text.
    Automatically processes document for Q&A if process_for_qa=True.
    """
    try:
        # Read file content
        content = await file.read()
        
        # Validate file type using magic bytes or extension
        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = ['.pdf', '.txt', '.jpeg', '.jpg', '.png']
        
        try:
            mime_type = magic.from_buffer(content, mime=True)
            allowed_types = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg', 'text/plain']
            if mime_type not in allowed_types and file_extension not in allowed_extensions:
                raise AppError(
                    f"Invalid file type: {mime_type}. Allowed: PDF, TXT, JPEG, PNG",
                    "INVALID_FILE_TYPE",
                    400
                )
        except Exception as e:
            logger.warning(f"File type detection issue: {e}, using extension check")
            if file_extension not in allowed_extensions:
                raise AppError(
                    f"Invalid file extension: {file_extension}. Allowed: {', '.join(allowed_extensions)}",
                    "INVALID_FILE_TYPE",
                    400
                )
        
        # Validate
        sanitizer.validate_file_upload(
            file.filename,
            content,
            max_size_mb=100
        )
        
        # Save file
        content_id = str(uuid.uuid4())
        safe_filename = f"{content_id}_{Path(file.filename).name}"
        file_path = UPLOAD_DIR / safe_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Extract text if it's a text file
        extracted_text = ""
        if file_extension == '.txt':
            try:
                extracted_text = content.decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to decode text file: {e}")
                extracted_text = content.decode('utf-8', errors='ignore')
        elif file_extension == '.pdf':
            # Try to extract text from PDF
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=content, filetype="pdf")
                extracted_text = ""
                for page in doc:
                    extracted_text += page.get_text()
                doc.close()
            except Exception as e:
                logger.warning(f"Failed to extract PDF text: {e}")
        
        # Save to database
        with get_db_session() as session:
            processed_content = ProcessedContent(
                id=uuid.UUID(content_id),
                original_text=extracted_text,
                language=grade_level or "en",
                grade_level=grade_level or 5,
                subject=subject or "General",
                metadata={
                    'filename': file.filename,
                    'file_path': str(file_path),
                    'file_size': len(content),
                    'file_type': file_extension
                }
            )
            session.add(processed_content)
            session.commit()
        
        # Start Q&A processing if requested and text is available
        qa_task_id = None
        if process_for_qa and extracted_text:
            try:
                qa_task = process_document_for_qa_task.apply_async(
                    args=[content_id, extracted_text, 512, 50]
                )
                qa_task_id = qa_task.id
                logger.info(f"Q&A processing started: {qa_task_id}")
            except Exception as e:
                logger.warning(f"Failed to start Q&A processing: {e}")
        
        return {
            "status": "uploaded",
            "content_id": content_id,
            "file_id": content_id,  # Alias for backward compatibility
            "file_path": str(file_path),
            "filename": file.filename,
            "size": len(content),
            "extracted_text": extracted_text[:5000] if extracted_text else "",  # Limit to first 5000 chars
            "grade_level": grade_level,
            "subject": subject,
            "qa_processing": {
                "enabled": process_for_qa and bool(extracted_text),
                "task_id": qa_task_id,
                "status_url": f"/api/v1/status/{qa_task_id}" if qa_task_id else None
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AppError:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=TaskResponse)
async def process_content(
    file_path: str,
    request_data: ProcessRequest
):
    """
    Process content through full pipeline asynchronously.
    
    Returns task ID for tracking progress.
    """
    try:
        # Validate file exists
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Validate parameters
        sanitizer.validate_grade_level(request_data.grade_level)
        sanitizer.validate_language(request_data.target_languages[0] if request_data.target_languages else 'Hindi')
        
        # Submit to Celery
        task = full_pipeline_task.delay(
            file_path=file_path,
            grade_level=request_data.grade_level,
            subject=request_data.subject,
            target_languages=request_data.target_languages,
            output_format=request_data.output_format,
            validation_threshold=request_data.validation_threshold
        )
        
        logger.info(f"Pipeline task submitted: {task.id}")
        
        return TaskResponse(
            task_id=task.id,
            state='PENDING',
            message='Processing started'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simplify")
async def simplify_content(
    request: SimplifyRequest,
    wait: bool = Query(default=True, description="Wait for result synchronously")
):
    """Simplify text asynchronously or synchronously."""
    try:
        # Sanitize input
        clean_text = sanitizer.sanitize_text(request.text)
        grade = request.get_grade_level()
        sanitizer.validate_grade_level(grade)
        
        # Submit task
        task = simplify_text_task.delay(
            text=clean_text,
            grade_level=grade,
            subject=request.subject,
            formula_blocks=None
        )
        
        # If wait is True, poll for result using asyncio
        if wait:
            import asyncio
            max_wait = 60  # seconds
            poll_interval = 0.5  # seconds
            elapsed = 0
            
            while elapsed < max_wait:
                # Check task status in a non-blocking way
                task_status = task.state
                
                if task_status in ['SUCCESS', 'FAILURE']:
                    if task_status == 'SUCCESS':
                        result = task.result
                        return {
                            "simplified_text": result.get('simplified_text', ''),
                            "grade_level": result.get('grade_level', grade),
                            "task_id": task.id,
                            "status": "completed"
                        }
                    else:
                        return {
                            "error": str(task.info) if hasattr(task, 'info') else "Task failed",
                            "task_id": task.id,
                            "status": "failed"
                        }
                
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
            
            # Timeout
            return TaskResponse(
                task_id=task.id,
                state='PENDING',
                message='Task is still processing. Use task_id to check status.'
            )
        
        return TaskResponse(
            task_id=task.id,
            state='PENDING',
            message='Simplification started'
        )
        
    except Exception as e:
        logger.error(f"Simplify failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/translate")
async def translate_content(
    request: TranslateRequest,
    wait: bool = Query(default=True, description="Wait for result synchronously")
):
    """Translate text asynchronously or synchronously."""
    try:
        # Sanitize input
        clean_text = sanitizer.sanitize_text(request.text)
        
        # Get target languages
        target_langs = request.get_target_languages()
        
        # Validate languages
        for lang in target_langs:
            sanitizer.validate_language(lang)
        
        # Submit task
        task = translate_text_task.delay(
            text=clean_text,
            target_languages=target_langs,
            formula_blocks=None
        )
        
        # If wait is True, poll for result
        if wait:
            import asyncio
            max_wait = 60
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < max_wait:
                if task.ready():
                    if task.successful():
                        result = task.result
                        # Get the first translation (backward compatibility)
                        translations = result.get('translations', {})
                        first_lang = target_langs[0] if target_langs else 'Hindi'
                        translated_text = translations.get(first_lang, '')
                        return {
                            "translated_text": translated_text,
                            "translations": translations,
                            "task_id": task.id,
                            "status": "completed"
                        }
                    else:
                        return {
                            "error": str(task.info),
                            "task_id": task.id,
                            "status": "failed"
                        }
                await asyncio.sleep(1)
            
            return TaskResponse(
                task_id=task.id,
                state='PENDING',
                message='Task is still processing'
            )
        
        return TaskResponse(
            task_id=task.id,
            state='PENDING',
            message='Translation started'
        )
        
    except Exception as e:
        logger.error(f"Translate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_content(
    request: ValidateRequest,
    wait: bool = Query(default=True, description="Wait for result synchronously")
):
    """Validate content semantically."""
    try:
        # Get texts
        original, processed = request.get_texts()
        
        # Sanitize
        original = sanitizer.sanitize_text(original)
        processed = sanitizer.sanitize_text(processed)
        
        # Submit task
        task = validate_content_task.delay(
            original_text=original,
            processed_text=processed
        )
        
        # If wait is True, poll for result
        if wait:
            import asyncio
            max_wait = 60
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < max_wait:
                if task.ready():
                    if task.successful():
                        result = task.result
                        return {
                            "is_valid": result.get('is_valid', True),
                            "accuracy_score": result.get('accuracy_score', 0.0),
                            "issues": result.get('issues', []),
                            "task_id": task.id,
                            "status": "completed"
                        }
                    else:
                        return {
                            "error": str(task.info),
                            "task_id": task.id,
                            "status": "failed"
                        }
                await asyncio.sleep(1)
            
            return TaskResponse(
                task_id=task.id,
                state='PENDING',
                message='Task is still processing'
            )
        
        return TaskResponse(
            task_id=task.id,
            state='PENDING',
            message='Validation started'
        )
        
    except Exception as e:
        logger.error(f"Validate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    wait: bool = Query(default=True, description="Wait for result synchronously")
):
    """Generate speech audio asynchronously or synchronously."""
    try:
        # Sanitize
        clean_text = sanitizer.sanitize_text(request.text)
        sanitizer.validate_language(request.language)
        
        # Submit task
        task = generate_audio_task.delay(
            text=clean_text,
            language=request.language,
            content_id=f"tts_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        
        # If wait is True, poll for result
        if wait:
            import asyncio
            max_wait = 120  # TTS may take longer
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < max_wait:
                if task.ready():
                    if task.successful():
                        result = task.result
                        return {
                            "audio_url": result.get('audio_url', ''),
                            "duration": result.get('duration', 0),
                            "task_id": task.id,
                            "status": "completed"
                        }
                    else:
                        return {
                            "error": str(task.info),
                            "task_id": task.id,
                            "status": "failed"
                        }
                await asyncio.sleep(1)
            
            return TaskResponse(
                task_id=task.id,
                state='PENDING',
                message='Task is still processing'
            )
        
        return TaskResponse(
            task_id=task.id,
            state='PENDING',
            message='Audio generation started'
        )
        
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """Get task status and result."""
    try:
        task_info = get_task_info(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskResponse(**task_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel running task."""
    try:
        success = revoke_task(task_id, terminate=True)
        
        return {
            "task_id": task_id,
            "cancelled": success
        }
        
    except Exception as e:
        logger.error(f"Task cancel failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback."""
    try:
        # Validate content ID
        sanitizer.validate_uuid(request.content_id)
        
        # Save to database
        with get_db_session() as session:
            feedback = Feedback(
                content_id=request.content_id,
                rating=request.rating,
                feedback_text=request.feedback_text,
                issue_type=request.issue_type
            )
            session.add(feedback)
            session.flush()
            feedback_id = feedback.id
        
        return {
            "status": "submitted",
            "feedback_id": str(feedback_id)
        }
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/content/{content_id}")
async def get_content(content_id: str):
    """Retrieve processed content."""
    try:
        # Validate UUID
        sanitizer.validate_uuid(content_id)
        
        # Query database
        with get_db_session() as session:
            content = session.query(ProcessedContent).filter(
                ProcessedContent.id == content_id
            ).first()
            
            if not content:
                raise HTTPException(status_code=404, detail="Content not found")
            
            return {
                "id": str(content.id),
                "original_text": (content.original_text[:500] + "...") if content.original_text else "",
                "simplified_text": (content.simplified_text[:500] + "...") if content.simplified_text else "",
                "translated_text": (content.translated_text[:500] + "...") if content.translated_text else "",
                "language": content.language,
                "grade_level": content.grade_level,
                "subject": content.subject,
                "audio_url": f"/api/v1/audio/{content.id}" if content.audio_file_path else None,
                "metadata": content.content_metadata or {}
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio/{content_id}")
async def get_audio(content_id: str):
    """Serve audio file."""
    try:
        sanitizer.validate_uuid(content_id)
        
        with get_db_session() as session:
            content = session.query(ProcessedContent).filter(
                ProcessedContent.id == content_id
            ).first()
            
            if not content or not content.audio_file_path:
                raise HTTPException(status_code=404, detail="Audio not found")
            
            audio_path = Path(content.audio_file_path)
            
            if not audio_path.exists():
                raise HTTPException(status_code=404, detail="Audio file missing")
            
            return FileResponse(
                audio_path,
                media_type="audio/mpeg",
                filename=audio_path.name
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _format_library_item(item: ProcessedContent) -> dict:
    """Format a single library item for API response."""
    return {
        "id": str(item.id),
        "original_text": item.original_text[:500] if item.original_text else "",
        "simplified_text": item.simplified_text[:500] if item.simplified_text else "",
        "translations": {item.language: item.translated_text} if item.translated_text else {},
        "validation_score": item.ncert_alignment_score or 0.0,
        "audio_available": bool(item.audio_file_path),
        "grade_level": item.grade_level,
        "subject": item.subject,
        "language": item.language,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": getattr(item, 'updated_at', None).isoformat() if hasattr(item, 'updated_at') and item.updated_at else None
    }


def _apply_library_filters(query, grade: Optional[int], subject: Optional[str], language: Optional[str]):
    """Apply filters to library query."""
    if grade:
        query = query.filter(ProcessedContent.grade_level == grade)
    if subject:
        query = query.filter(ProcessedContent.subject == subject)
    if language:
        query = query.filter(ProcessedContent.language == language)
    return query


@router.get("/library")
async def get_library(
    language: Optional[str] = Query(None),
    grade: Optional[int] = Query(None, ge=5, le=12),
    subject: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get paginated library of processed content."""
    try:
        with get_db_session() as session:
            query = session.query(ProcessedContent)
            query = _apply_library_filters(query, grade, subject, language)
            
            total = query.count()
            items = query.order_by(ProcessedContent.created_at.desc()).offset(offset).limit(limit).all()
            
            formatted_items = [_format_library_item(item) for item in items]
            
            return {
                "items": formatted_items,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
            
    except Exception as e:
        logger.error(f"Library retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/content/search")
async def search_content(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100)
):
    """Search content by text."""
    try:
        with get_db_session() as session:
            # Search in original text, simplified text, and subject
            query = session.query(ProcessedContent).filter(
                (ProcessedContent.original_text.ilike(f"%{q}%")) |
                (ProcessedContent.simplified_text.ilike(f"%{q}%")) |
                (ProcessedContent.subject.ilike(f"%{q}%"))
            ).order_by(ProcessedContent.created_at.desc()).limit(limit)
            
            items = query.all()
            
            # Format results
            results = []
            for item in items:
                results.append({
                    "id": str(item.id),
                    "original_text": item.original_text[:500] if item.original_text else "",
                    "simplified_text": item.simplified_text[:500] if item.simplified_text else "",
                    "translations": {item.language: item.translated_text} if item.translated_text else {},
                    "validation_score": item.ncert_alignment_score or 0.0,
                    "audio_available": bool(item.audio_file_path),
                    "grade_level": item.grade_level,
                    "subject": item.subject,
                    "language": item.language,
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                    "updated_at": getattr(item, 'updated_at', None).isoformat() if hasattr(item, 'updated_at') and item.updated_at else None
                })
            
            return {"results": results}
            
    except Exception as e:
        logger.error(f"Content search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
