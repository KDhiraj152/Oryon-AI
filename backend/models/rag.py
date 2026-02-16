"""
RAG (Retrieval-Augmented Generation) models.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    TIMESTAMP,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from backend.db.database import Base


def utcnow():
    """Get current UTC time as naive datetime (for TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.utcnow()

class DocumentChunk(Base):
    """Stores text chunks from uploaded documents for RAG."""

    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processed_content.id"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_size = Column(Integer)
    chunk_metadata = Column(
        JSONB
    )  # page number, section, etc. (renamed from 'metadata' to avoid SQLAlchemy reserved word)
    created_at = Column(TIMESTAMP, default=utcnow)

class Embedding(Base):
    """Stores vector embeddings for semantic search using pgvector."""

    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id"), nullable=False, index=True
    )
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processed_content.id"),
        nullable=False,
        index=True,
    )
    # Note: vector column will be added via Alembic migration with pgvector extension
    # embedding = Column(Vector(1024))  # multilingual-e5-large dimension (upgraded from 384)
    embedding_model = Column(String(100), default="intfloat/multilingual-e5-large")
    embedding_version = Column(
        Integer, default=2
    )  # Version 2 = E5-large (1024-dim), Version 1 = MiniLM (384-dim)
    created_at = Column(TIMESTAMP, default=utcnow)

class ChatHistory(Base):
    """Stores Q&A chat history for context-aware conversations."""

    __tablename__ = "chat_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processed_content.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    context_chunks: Any = Column(ARRAY(UUID(as_uuid=True)))  # IDs of chunks used
    confidence_score = Column(Float)
    created_at = Column(TIMESTAMP, default=utcnow, index=True)
