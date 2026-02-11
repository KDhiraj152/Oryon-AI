"""
V2 API - Chat Routes
====================

Endpoints for AI-powered chat, conversations, and TTS for chat.
Uses database storage for production reliability.
OPTIMIZED: Uses orjson for faster SSE serialization.
"""

import logging
import time
import uuid as uuid_module
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_async_db
from ...models.chat import Conversation, Message, MessageRole
from ...utils.auth import TokenData, get_current_user
from ...utils.memory_guard import require_memory
from ..deps import get_ai_engine as _get_ai_engine, json_dumps as _json_dumps

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# Error constants
ERROR_CONVERSATION_NOT_FOUND = "Conversation not found"
ERROR_ACCESS_DENIED = "Access denied"


# ==================== Models ====================


class ChatMessage(BaseModel):
    """Chat message request model - flexible like ChatGPT/Perplexity."""

    message: str = Field(..., min_length=1, max_length=8000)
    conversation_id: str | None = None
    language: str = Field(default="en")
    subject: str | None = Field(default=None, description="Optional subject context")


class ChatResponse(BaseModel):
    """Chat response model."""

    message_id: str
    response: str
    language: str
    processing_time_ms: float
    conversation_id: str | None = None
    sources: list[str] | None = None
    confidence: float = 1.0


class ConversationCreate(BaseModel):
    """Conversation creation request."""

    title: str = Field(default="New Conversation", max_length=100)
    language: str = Field(default="en")
    subject: str = Field(default="General", max_length=50)


class ConversationResponse(BaseModel):
    """Conversation response model."""

    id: str
    title: str
    language: str = "en"
    subject: str = "General"
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageResponse(BaseModel):
    """Message response model."""

    id: str
    role: str
    content: str
    timestamp: str


class GuestTTSRequest(BaseModel):
    """Request model for guest TTS endpoint."""

    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(default="hi")
    gender: str = Field(default="female", pattern="^(male|female)$")
    voice: str | None = None
    rate: str = Field(default="+0%")
    pitch: str = Field(default="+0Hz")


# ==================== Helper Functions ====================


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Format SSE event using fast JSON serialization."""
    return f"event: {event}\ndata: {_json_dumps(data)}\n\n"


# ==================== Chat Endpoints ====================


@router.post("/chat", response_model=ChatResponse)
@require_memory(action="reject", reject_on=("critical", "emergency"))
async def chat(
    request: ChatMessage,
    current_user: TokenData | None = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Send a chat message and get AI response with RAG and validation."""
    start_time = time.perf_counter()

    try:
        from ...services.ai_core.engine import GenerationConfig

        # OPTIMIZATION: Use cached engine singleton
        engine = _get_ai_engine()

        # Prepare context data - minimal, no constraints
        context_data = {
            "language": request.language,
            "subject": request.subject,
        }

        # Use the full chat method with RAG
        formatted_response = await engine.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            user_id=current_user.user_id if current_user else None,
            config=GenerationConfig(
                temperature=0.7,
                block_harmful=True,
            ),
            context_data=context_data,
        )

        elapsed = (time.perf_counter() - start_time) * 1000
        message_id = str(uuid_module.uuid4())[:8]

        # Store messages in database if conversation exists
        if request.conversation_id and current_user:
            try:
                conv_uuid = UUID(request.conversation_id)
                user_uuid = UUID(current_user.user_id)

                # Verify conversation exists and belongs to user
                conv_stmt = select(Conversation).where(
                    Conversation.id == conv_uuid, Conversation.user_id == user_uuid
                )
                conv_result = await db.execute(conv_stmt)
                conv = conv_result.scalar_one_or_none()

                if conv:
                    # Add user message
                    user_msg = Message(
                        conversation_id=conv_uuid,
                        role=MessageRole.USER.value,
                        content=request.message,
                    )
                    db.add(user_msg)

                    # Add assistant message
                    assistant_msg = Message(
                        conversation_id=conv_uuid,
                        role=MessageRole.ASSISTANT.value,
                        content=formatted_response.content,
                    )
                    db.add(assistant_msg)

                    # Update conversation timestamp
                    conv.updated_at = datetime.now(timezone.utc)

                    await db.commit()
            except Exception as db_error:
                logger.warning(f"Failed to save messages to DB: {db_error}")
                # Don't fail the request if DB save fails

        # Extract source titles and confidence from metadata
        source_titles = None
        confidence = 1.0
        if formatted_response.metadata:
            if formatted_response.metadata.sources:
                source_titles = [s.title for s in formatted_response.metadata.sources]
            confidence = formatted_response.metadata.confidence

        return ChatResponse(
            message_id=message_id,
            response=formatted_response.content,
            language=request.language,
            processing_time_ms=elapsed,
            conversation_id=request.conversation_id,
            sources=source_titles,
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@require_memory(action="reject", reject_on=("critical", "emergency"))
async def chat_stream(request: ChatMessage):
    """Stream chat response with SSE using AI Engine with RAG.

    Optimizations:
    - Buffered token output (reduces network overhead)
    - Early status event for perceived latency
    - Async generator for memory efficiency
    """

    async def generate():
        try:
            from ...services.ai_core.engine import GenerationConfig

            # OPTIMIZATION: Use cached engine singleton
            engine = _get_ai_engine()

            yield sse_event(
                "status", {"stage": "generating", "message": "Generating response..."}
            )

            # Prepare context data - no grade constraints
            context_data = {
                "subject": request.subject,
                "language": request.language,
            }

            config = GenerationConfig(
                stream=True,
                temperature=0.7,
                block_harmful=True,
            )

            # OPTIMIZATION: Buffer tokens for reduced network overhead
            # Send every 3 tokens or after 50ms to balance latency vs throughput
            token_buffer = []
            last_send = 0
            import time

            async for chunk in engine.chat_stream(
                message=request.message,
                conversation_id=request.conversation_id,
                config=config,
                context_data=context_data,
            ):
                token_buffer.append(chunk)
                current_time = time.perf_counter()

                # Flush buffer if 3+ tokens or 50ms elapsed
                if len(token_buffer) >= 3 or (current_time - last_send) > 0.05:
                    yield sse_event("chunk", {"text": "".join(token_buffer)})
                    token_buffer = []
                    last_send = current_time

            # Flush remaining tokens
            if token_buffer:
                yield sse_event("chunk", {"text": "".join(token_buffer)})

            yield sse_event("complete", {"message_id": str(uuid_module.uuid4())[:8]})

        except Exception as e:
            yield sse_event("error", {"error": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post("/chat/guest")
async def chat_guest(request: ChatMessage):
    """
    Guest chat endpoint - Uses FULL OPTIMIZED PIPELINE.

    Pipeline components used:
    - PolicyEngine: Content filtering based on mode
    - RAG Service: Document retrieval (if applicable)
    - Self-Optimizer: Dynamic parameter tuning
    - Resource Scheduler: ANE/GPU-aware scheduling
    - Speculative Decoder: Metal/ANE acceleration
    - Context Allocator: Dynamic context management
    - Safety Pipeline: 3-pass verification

    First request initializes pipeline (~4-5s), subsequent requests ~1-2s.
    """
    import asyncio

    start_time = time.perf_counter()

    try:
        from ...services.ai_core.engine import GenerationConfig

        # OPTIMIZATION: Use cached engine singleton
        engine = _get_ai_engine()

        # Prepare context data
        context_data = {
            "subject": request.subject,
            "language": request.language,
        }

        # Use optimized config for guest chat
        config = GenerationConfig(
            temperature=0.7,
            max_tokens=512,
            block_harmful=True,
            use_rag=False,  # Skip RAG for guest - faster response
            timeout_seconds=30.0,
        )

        # Run in executor to avoid blocking event loop during initialization
        asyncio.get_running_loop()

        # The chat method handles all pipeline components
        formatted_response = await engine.chat(
            message=request.message,
            config=config,
            context_data=context_data,
        )

        elapsed = (time.perf_counter() - start_time) * 1000

        # Extract sources and confidence from response metadata
        sources = []
        confidence = 0.85
        if formatted_response.metadata:
            if (
                hasattr(formatted_response.metadata, "sources")
                and formatted_response.metadata.sources
            ):
                sources = [s.title for s in formatted_response.metadata.sources]
            if hasattr(formatted_response.metadata, "confidence"):
                confidence = formatted_response.metadata.confidence

        return {
            "message_id": str(uuid_module.uuid4())[:8],
            "response": formatted_response.content,
            "language": request.language,
            "processing_time_ms": round(elapsed, 1),
            "sources": sources,
            "confidence": confidence,
        }

    except Exception as e:
        logger.error(f"Guest chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Conversation Endpoints ====================


@router.get("/chat/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """List user's conversations from database."""
    try:
        user_uuid = UUID(current_user.user_id)

        # Query conversations with message count using subquery
        stmt = (
            select(Conversation, func.count(Message.id).label("message_count"))
            .outerjoin(Message, Message.conversation_id == Conversation.id)
            .where(Conversation.user_id == user_uuid)
            .group_by(Conversation.id)
            .order_by(Conversation.updated_at.desc())
        )

        result = await db.execute(stmt)
        rows = result.all()

        return [
            ConversationResponse(
                id=str(conv.id),
                title=conv.title or "Conversation",
                subject=conv.extra_data.get("subject", "General")
                if conv.extra_data
                else "General",
                language=conv.extra_data.get("language", "en")
                if conv.extra_data
                else "en",
                created_at=conv.created_at.isoformat()
                if conv.created_at
                else datetime.now(timezone.utc).isoformat(),
                updated_at=conv.updated_at.isoformat()
                if conv.updated_at
                else datetime.now(timezone.utc).isoformat(),
                message_count=msg_count or 0,
            )
            for conv, msg_count in rows
        ]
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationCreate,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Create a new conversation in database."""
    try:
        user_uuid = UUID(current_user.user_id)
        now = datetime.now(timezone.utc)

        conversation = Conversation(
            user_id=user_uuid,
            title=request.title,
            created_at=now,
            updated_at=now,
            extra_data={"subject": request.subject, "language": request.language},
        )

        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)

        return ConversationResponse(
            id=str(conversation.id),
            title=conversation.title,
            subject=request.subject,
            language=request.language,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            message_count=0,
        )
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/chat/conversations/{conversation_id}", response_model=ConversationResponse
)
async def get_conversation(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get a specific conversation from database."""
    try:
        conv_uuid = UUID(conversation_id)
        user_uuid = UUID(current_user.user_id)

        stmt = (
            select(Conversation, func.count(Message.id).label("message_count"))
            .outerjoin(Message, Message.conversation_id == Conversation.id)
            .where(Conversation.id == conv_uuid)
            .group_by(Conversation.id)
        )

        result = await db.execute(stmt)
        row = result.first()

        if not row:
            raise HTTPException(status_code=404, detail=ERROR_CONVERSATION_NOT_FOUND)

        conv, msg_count = row

        if conv.user_id != user_uuid:
            raise HTTPException(status_code=403, detail=ERROR_ACCESS_DENIED)

        return ConversationResponse(
            id=str(conv.id),
            title=conv.title or "Conversation",
            subject=conv.extra_data.get("subject", "General")
            if conv.extra_data
            else "General",
            language=conv.extra_data.get("language", "en") if conv.extra_data else "en",
            created_at=conv.created_at.isoformat()
            if conv.created_at
            else datetime.now(timezone.utc).isoformat(),
            updated_at=conv.updated_at.isoformat()
            if conv.updated_at
            else datetime.now(timezone.utc).isoformat(),
            message_count=msg_count or 0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/chat/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def get_conversation_messages(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get messages in a conversation from database."""
    try:
        conv_uuid = UUID(conversation_id)
        user_uuid = UUID(current_user.user_id)

        # First verify ownership
        conv_stmt = select(Conversation).where(Conversation.id == conv_uuid)
        conv_result = await db.execute(conv_stmt)
        conv = conv_result.scalar_one_or_none()

        if not conv:
            raise HTTPException(status_code=404, detail=ERROR_CONVERSATION_NOT_FOUND)

        if conv.user_id != user_uuid:
            raise HTTPException(status_code=403, detail=ERROR_ACCESS_DENIED)

        # Get messages
        msg_stmt = (
            select(Message)
            .where(Message.conversation_id == conv_uuid)
            .order_by(Message.created_at.asc())
        )

        result = await db.execute(msg_stmt)
        messages = result.scalars().all()

        return [
            MessageResponse(
                id=str(msg.id),
                role=msg.role,
                content=msg.content,
                timestamp=msg.created_at.isoformat()
                if msg.created_at
                else datetime.now(timezone.utc).isoformat(),
            )
            for msg in messages
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ConversationUpdate(BaseModel):
    """Conversation update request."""

    title: str | None = Field(None, max_length=100)


@router.patch(
    "/chat/conversations/{conversation_id}", response_model=ConversationResponse
)
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdate,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Update a conversation (title)."""
    try:
        conv_uuid = UUID(conversation_id)
        user_uuid = UUID(current_user.user_id)

        # Verify ownership
        conv_stmt = select(Conversation).where(Conversation.id == conv_uuid)
        conv_result = await db.execute(conv_stmt)
        conv = conv_result.scalar_one_or_none()

        if not conv:
            raise HTTPException(status_code=404, detail=ERROR_CONVERSATION_NOT_FOUND)

        if conv.user_id != user_uuid:
            raise HTTPException(status_code=403, detail=ERROR_ACCESS_DENIED)

        # Update fields
        if request.title is not None:
            conv.title = request.title
        conv.updated_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(conv)

        # Get message count
        msg_stmt = select(func.count(Message.id)).where(
            Message.conversation_id == conv_uuid
        )
        msg_result = await db.execute(msg_stmt)
        msg_count = msg_result.scalar() or 0

        return ConversationResponse(
            id=str(conv.id),
            title=conv.title or "Conversation",
            subject=conv.extra_data.get("subject", "General")
            if conv.extra_data
            else "General",
            language=conv.extra_data.get("language", "en") if conv.extra_data else "en",
            created_at=conv.created_at.isoformat()
            if conv.created_at
            else datetime.now(timezone.utc).isoformat(),
            updated_at=conv.updated_at.isoformat()
            if conv.updated_at
            else datetime.now(timezone.utc).isoformat(),
            message_count=msg_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Delete a conversation from database."""
    try:
        conv_uuid = UUID(conversation_id)
        user_uuid = UUID(current_user.user_id)

        # Verify ownership
        conv_stmt = select(Conversation).where(Conversation.id == conv_uuid)
        conv_result = await db.execute(conv_stmt)
        conv = conv_result.scalar_one_or_none()

        if not conv:
            raise HTTPException(status_code=404, detail=ERROR_CONVERSATION_NOT_FOUND)

        if conv.user_id != user_uuid:
            raise HTTPException(status_code=403, detail=ERROR_ACCESS_DENIED)

        # Delete (cascade will handle messages)
        await db.delete(conv)
        await db.commit()

        return {"message": "Conversation deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Chat TTS Endpoints ====================


@router.post("/chat/tts")
async def guest_text_to_speech(request: GuestTTSRequest):
    """
    Guest TTS endpoint (no auth required) - returns base64 audio.

    Uses Edge TTS (high quality, 400+ voices) with MMS-TTS fallback.
    Supports all Indian languages: hi, te, ta, bn, mr, gu, kn, ml, pa, or
    """
    import base64

    try:
        # Try Edge TTS first (high quality, requires internet)
        try:
            from ...services.tts.edge_tts_service import get_edge_tts_service

            edge_tts = get_edge_tts_service()

            audio_bytes = await edge_tts.synthesize(
                text=request.text,
                language=request.language,
                gender=request.gender,
                voice_name=request.voice,
                rate=request.rate,
                pitch=request.pitch,
            )

            # Return base64 encoded audio
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "success": True,
                "audio_data": audio_base64,
                "audio_format": "audio/mpeg",
                "use_browser_tts": False,
            }

        except Exception as edge_error:
            logger.warning(f"Edge TTS failed, trying MMS-TTS: {edge_error}")

            # Fallback to MMS-TTS (offline, but lower quality)
            try:
                from ...services.tts.mms_tts_service import get_mms_tts_service

                mms_tts = get_mms_tts_service()

                audio_bytes = mms_tts.synthesize(
                    text=request.text, language=request.language
                )

                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                return {
                    "success": True,
                    "audio_data": audio_base64,
                    "audio_format": "audio/wav",
                    "use_browser_tts": False,
                }

            except Exception as mms_error:
                logger.warning(f"MMS-TTS also failed: {mms_error}")
                # Signal frontend to use browser TTS
                return {
                    "success": False,
                    "use_browser_tts": True,
                    "error": "TTS services unavailable, use browser speech synthesis",
                }

    except Exception as e:
        logger.error(f"Guest TTS error: {e}")
        return {"success": False, "use_browser_tts": True, "error": str(e)}


@router.get("/chat/tts/voices")
async def get_chat_tts_voices():
    """Get available TTS voices for chat (guest endpoint)."""
    try:
        from ...services.tts.edge_tts_service import get_edge_tts_service

        edge_tts = get_edge_tts_service()
        voices = edge_tts.get_available_voices()

        # Format for frontend
        voice_list = []
        for lang_code, lang_voices in voices.items():
            for gender in ["male", "female"]:
                if lang_voices.get(gender):
                    for voice_name in lang_voices[gender]:
                        voice_list.append(
                            {
                                "name": voice_name,
                                "language": lang_code,
                                "gender": gender,
                                "locale": lang_code,
                            }
                        )

        return {"voices": voice_list}

    except Exception as e:
        logger.warning(f"Failed to get voices: {e}")
        return {
            "voices": [
                {
                    "name": "hi-IN-SwaraNeural",
                    "language": "hi",
                    "gender": "female",
                    "locale": "hi-IN",
                },
                {
                    "name": "hi-IN-MadhurNeural",
                    "language": "hi",
                    "gender": "male",
                    "locale": "hi-IN",
                },
                {
                    "name": "en-IN-NeerjaNeural",
                    "language": "en",
                    "gender": "female",
                    "locale": "en-IN",
                },
            ]
        }
