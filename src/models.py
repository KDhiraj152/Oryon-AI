"""
Simplified data models - merged content and auth models.

Replaces: repository/models.py, repository/auth_models.py
All models in one file for easier management.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, TIMESTAMP, ForeignKey, ARRAY, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from .database import Base


# =============================================================================
# CONTENT MODELS
# =============================================================================

class ProcessedContent(Base):
    """Stores processed educational content with translations and audio."""
    __tablename__ = 'processed_content'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_text = Column(Text, nullable=False)
    simplified_text = Column(Text)
    translated_text = Column(Text)
    language = Column(String(50), nullable=False, index=True)
    grade_level = Column(Integer, nullable=False, index=True)
    subject = Column(String(100), nullable=False, index=True)
    audio_file_path = Column(Text)
    ncert_alignment_score = Column(Float)
    audio_accuracy_score = Column(Float)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    content_metadata = Column('metadata', JSONB)
    user_id = Column(UUID(as_uuid=True), index=True)  # Track owner
    
    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_user_content', 'user_id', 'created_at'),
        Index('idx_grade_subject', 'grade_level', 'subject'),
        Index('idx_language_grade', 'language', 'grade_level'),
        Index('idx_subject_created', 'subject', 'created_at'),
    )


class NCERTStandard(Base):
    """Reference database for NCERT curriculum standards."""
    __tablename__ = 'ncert_standards'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    grade_level = Column(Integer, nullable=False, index=True)
    subject = Column(String(100), nullable=False, index=True)
    topic = Column(String(200), nullable=False)
    learning_objectives = Column(ARRAY(Text))
    keywords = Column(ARRAY(Text))


class Feedback(Base):
    """User feedback for content quality."""
    __tablename__ = 'feedback'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(UUID(as_uuid=True), ForeignKey('processed_content.id'), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), index=True)
    rating = Column(Integer, nullable=False)  # 1-5 stars
    feedback_text = Column(Text)
    issue_type = Column(String(100))  # e.g., 'translation', 'audio', 'simplification'
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)


class PipelineLog(Base):
    """Logs for pipeline processing stages and performance metrics."""
    __tablename__ = 'pipeline_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(UUID(as_uuid=True), ForeignKey('processed_content.id'), index=True)
    stage = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False)
    processing_time_ms = Column(Integer)
    error_message = Column(Text)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow, index=True)


# =============================================================================
# Q&A / RAG MODELS
# =============================================================================

class DocumentChunk(Base):
    """Stores text chunks from uploaded documents for RAG."""
    __tablename__ = 'document_chunks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(UUID(as_uuid=True), ForeignKey('processed_content.id'), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_size = Column(Integer)
    chunk_metadata = Column(JSONB)  # page number, section, etc. (renamed from 'metadata' to avoid SQLAlchemy reserved word)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class Embedding(Base):
    """Stores vector embeddings for semantic search using pgvector."""
    __tablename__ = 'embeddings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey('document_chunks.id'), nullable=False, index=True)
    content_id = Column(UUID(as_uuid=True), ForeignKey('processed_content.id'), nullable=False, index=True)
    # Note: vector column will be added via Alembic migration with pgvector extension
    # embedding = Column(Vector(384))  # sentence-transformers default dimension
    embedding_model = Column(String(100), default='sentence-transformers/all-MiniLM-L6-v2')
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class ChatHistory(Base):
    """Stores Q&A chat history for context-aware conversations."""
    __tablename__ = 'chat_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    content_id = Column(UUID(as_uuid=True), ForeignKey('processed_content.id'), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    context_chunks = Column(ARRAY(UUID(as_uuid=True)))  # IDs of chunks used
    confidence_score = Column(Float)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)


# =============================================================================
# USER & AUTH MODELS
# =============================================================================

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    organization = Column(String(255))
    role = Column(String(50), default='user', nullable=False)  # 'user', 'admin', 'educator'
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(TIMESTAMP)


class APIKey(Base):
    """API key model for programmatic access."""
    __tablename__ = 'api_keys'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    expires_at = Column(TIMESTAMP)
    last_used = Column(TIMESTAMP)


__all__ = [
    'ProcessedContent',
    'NCERTStandard',
    'Feedback',
    'PipelineLog',
    'DocumentChunk',
    'Embedding',
    'ChatHistory',
    'User',
    'APIKey'
]
