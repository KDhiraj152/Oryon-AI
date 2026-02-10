"""
Celery tasks for Q&A (Question Answering) functionality.

These tasks handle:
- Processing uploaded documents for Q&A
- Answering questions using RAG + Qwen2.5
- Storing Q&A history
"""

import logging
from datetime import UTC, datetime, timezone
from typing import Any, Dict
from uuid import UUID

from celery import shared_task

from ..database import SessionLocal
from ..models import ChatHistory, ProcessedContent
from ..services.rag import get_rag_service
from ..services.simplify.simplifier import TextSimplifier

logger = logging.getLogger(__name__)

# Global model instance
_text_simplifier = None


def get_text_simplifier():
    """Get or create global TextSimplifier instance."""
    global _text_simplifier
    if _text_simplifier is None:
        _text_simplifier = TextSimplifier()
    return _text_simplifier


@shared_task(bind=True, name="pipeline.process_document_for_qa")
def process_document_for_qa_task(
    self, content_id: str, text: str, chunk_size: int = 512, overlap: int = 50
) -> dict[str, Any]:
    """
    Process a document for Q&A by chunking and generating embeddings.

    Args:
        content_id: UUID of the processed content
        text: Full document text
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks

    Returns:
        Dict with processing results
    """
    try:
        logger.info(f"Processing document {content_id} for Q&A")

        # Get RAG service
        rag_service = get_rag_service()

        # Store chunks with embeddings
        num_chunks = rag_service.store_document_chunks(
            content_id=UUID(content_id),
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # Update processed content status
        session = SessionLocal()
        try:
            content = (
                session.query(ProcessedContent)
                .filter(ProcessedContent.id == UUID(content_id))
                .first()
            )

            if content:
                # Update metadata to indicate Q&A is ready
                metadata = content.metadata or {}
                metadata["qa_ready"] = True
                metadata["num_chunks"] = num_chunks
                metadata["qa_processed_at"] = datetime.now(UTC).isoformat()
                content.metadata = metadata
                session.commit()
        finally:
            session.close()

        logger.info(f"Document {content_id} processed: {num_chunks} chunks created")

        return {
            "status": "success",
            "content_id": content_id,
            "num_chunks": num_chunks,
            "message": f"Document processed with {num_chunks} chunks",
        }

    except Exception as e:
        logger.error(f"Error processing document for Q&A: {e}", exc_info=True)
        return {"status": "error", "content_id": content_id, "error": str(e)}


@shared_task(bind=True, name="pipeline.answer_question")
def answer_question_task(
    self,
    content_id: str,
    question: str,
    user_id: str,
    top_k: int = 3,
    max_answer_length: int = 256,
) -> dict[str, Any]:
    """
    Answer a question about a document using RAG + Qwen2.5.

    Args:
        content_id: UUID of the processed content
        question: The user question
        user_id: UUID of the user asking the question
        top_k: Number of context chunks to retrieve
        max_answer_length: Maximum length of generated answer

    Returns:
        Dict with answer and metadata
    """
    try:
        logger.info(f"Answering question for content {content_id}: {question[:50]}...")

        # Get RAG service
        rag_service = get_rag_service()

        # Retrieve relevant context
        context_data = rag_service.retrieve_context(
            question=question, content_id=UUID(content_id), top_k=top_k
        )

        if not context_data["has_context"]:
            return {
                "status": "error",
                "content_id": content_id,
                "question": question,
                "error": "No relevant context found in document",
                "answer": "I cannot answer this question based on the document provided.",
            }

        # Build prompt for T5
        context_text = context_data["context_text"]

        # Use a simple extractive approach for now
        # Find the most relevant sentence from context
        sentences = [s.strip() for s in context_text.split(".") if s.strip()]

        # Simple answer generation: take first 2-3 sentences from context
        if sentences:
            answer = ". ".join(sentences[: min(3, len(sentences))]) + "."
        else:
            answer = (
                context_text[:200] + "..." if len(context_text) > 200 else context_text
            )

        # Add disclaimer if confidence is low
        if context_data["avg_score"] < 0.5:
            answer = f"Based on the document: {answer}"

        logger.info(f"Generated answer (length: {len(answer)} chars)")

        # Calculate confidence score based on context similarity
        confidence_score = context_data["avg_score"]

        # Store in chat history
        session = SessionLocal()
        try:
            chat_entry = ChatHistory(
                user_id=UUID(user_id),
                content_id=UUID(content_id),
                question=question,
                answer=answer,
                context_chunks=context_data["chunk_ids"],
                confidence_score=confidence_score,
            )
            session.add(chat_entry)
            session.commit()

            chat_id = str(chat_entry.id)
        finally:
            session.close()

        logger.info(
            f"Question answered successfully (confidence: {confidence_score:.2f})"
        )

        return {
            "status": "success",
            "content_id": content_id,
            "question": question,
            "answer": answer,
            "confidence_score": confidence_score,
            "num_context_chunks": len(context_data["chunk_ids"]),
            "context_scores": context_data["scores"],
            "chat_id": chat_id,
        }

    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        return {
            "status": "error",
            "content_id": content_id,
            "question": question,
            "error": str(e),
            "answer": "An error occurred while processing your question.",
        }


@shared_task(bind=True, name="pipeline.get_chat_history")
def get_chat_history_task(
    self, content_id: str, user_id: str, limit: int = 10
) -> dict[str, Any]:
    """
    Get chat history for a document and user.

    Args:
        content_id: UUID of the processed content
        user_id: UUID of the user
        limit: Maximum number of history entries to return

    Returns:
        Dict with chat history
    """
    try:
        session = SessionLocal()
        try:
            history = (
                session.query(ChatHistory)
                .filter(
                    ChatHistory.content_id == UUID(content_id),
                    ChatHistory.user_id == UUID(user_id),
                )
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
                .all()
            )

            history_list = [
                {
                    "id": str(h.id),
                    "question": h.question,
                    "answer": h.answer,
                    "confidence_score": h.confidence_score,
                    "created_at": h.created_at.isoformat() if h.created_at else None,
                }
                for h in history
            ]

            return {
                "status": "success",
                "content_id": content_id,
                "user_id": user_id,
                "history": history_list,
                "count": len(history_list),
            }
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}", exc_info=True)
        return {
            "status": "error",
            "content_id": content_id,
            "user_id": user_id,
            "error": str(e),
        }
