"""Q&A (Question & Answer) endpoints."""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Form, Query, Depends
from pydantic import BaseModel

from ...tasks.qa_tasks import (
    process_document_for_qa_task,
    answer_question_task,
    get_chat_history_task
)
from ...utils.input_sanitizer import InputSanitizer
from ...utils.auth import get_current_user
from ...database import get_db_session
from ...models import ProcessedContent, User
from ...monitoring import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])

# Input sanitizer
sanitizer = InputSanitizer()


@router.post("/process")
async def process_document_for_qa(
    content_id: str = Form(..., description="UUID of uploaded content"),
    chunk_size: int = Form(512, description="Characters per chunk"),
    overlap: int = Form(50, description="Characters to overlap"),
    current_user: User = Depends(get_current_user)
):
    """
    Process an uploaded document for Q&A by chunking and generating embeddings.
    
    This must be called after uploading a document before asking questions.
    """
    try:
        # Validate content exists
        with get_db_session() as session:
            content = session.query(ProcessedContent).filter(
                ProcessedContent.id == content_id
            ).first()
            
            if not content:
                raise HTTPException(status_code=404, detail="Content not found")
            
            # Get the text to process
            text = content.original_text or content.simplified_text
            if not text:
                raise HTTPException(status_code=400, detail="No text available in content")
        
        # Start processing task
        task = process_document_for_qa_task.apply_async(
            args=[content_id, text, chunk_size, overlap]
        )
        
        return {
            "message": "Document processing started",
            "task_id": task.id,
            "content_id": content_id,
            "status_url": f"/api/v1/status/{task.id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Q&A processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_question(
    content_id: str = Form(..., description="UUID of processed content"),
    question: str = Form(..., min_length=5, description="Your question"),
    wait: bool = Form(False, description="Wait for answer synchronously"),
    top_k: int = Form(3, description="Number of context chunks to use"),
    current_user: User = Depends(get_current_user)
):
    """
    Ask a question about an uploaded document.
    
    The document must first be processed using /api/v1/qa/process endpoint.
    """
    try:
        # Sanitize question
        question = sanitizer.sanitize_text(question)
        
        # Validate content exists and is processed for Q&A
        with get_db_session() as session:
            content = session.query(ProcessedContent).filter(
                ProcessedContent.id == content_id
            ).first()
            
            if not content:
                raise HTTPException(status_code=404, detail="Content not found")
            
            # Check if document is ready for Q&A
            metadata = content.content_metadata or {}
            if not metadata.get('qa_ready'):
                raise HTTPException(
                    status_code=400,
                    detail="Document not processed for Q&A. Call /api/v1/qa/process first."
                )
        
        # Start Q&A task
        task = answer_question_task.apply_async(
            args=[content_id, question, str(current_user.id), top_k]
        )
        
        if not wait:
            return {
                "message": "Question processing started",
                "task_id": task.id,
                "content_id": content_id,
                "question": question,
                "status_url": f"/api/v1/status/{task.id}"
            }
        
        # Wait for result synchronously
        import asyncio
        max_wait_time = 60  # seconds
        elapsed = 0
        
        while elapsed < max_wait_time:
            if task.ready():
                result = task.get()
                
                if result.get('status') == 'success':
                    return {
                        "answer": result['answer'],
                        "confidence_score": result['confidence_score'],
                        "num_context_chunks": result['num_context_chunks'],
                        "context_scores": result['context_scores'],
                        "chat_id": result['chat_id'],
                        "task_id": task.id
                    }
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=result.get('error', 'Failed to answer question')
                    )
            
            await asyncio.sleep(1)
            elapsed += 1
        
        # Timeout - return task ID for polling
        return {
            "message": "Question processing in progress",
            "task_id": task.id,
            "status_url": f"/api/v1/status/{task.id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Q&A failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{content_id}")
async def get_qa_history(
    content_id: str,
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user)
):
    """
    Get Q&A chat history for a specific document.
    """
    try:
        # Validate content exists
        with get_db_session() as session:
            content = session.query(ProcessedContent).filter(
                ProcessedContent.id == content_id
            ).first()
            
            if not content:
                raise HTTPException(status_code=404, detail="Content not found")
        
        # Get history
        task = get_chat_history_task.apply_async(
            args=[content_id, str(current_user.id), limit]
        )
        result = task.get(timeout=10)
        
        if result.get('status') == 'success':
            return {
                "content_id": content_id,
                "history": result['history'],
                "count": result['count']
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Failed to retrieve history')
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
