"""
Content models.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    TIMESTAMP,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from backend.db.database import Base


def utcnow():
    """Get current UTC time as naive datetime (for TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.utcnow()

class ProcessedContent(Base):
    """Stores processed content with translations and audio."""

    __tablename__ = "processed_content"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_text = Column(Text, nullable=False)
    simplified_text = Column(Text)
    translated_text = Column(Text)
    language = Column(String(50), nullable=False, index=True)
    complexity_level = Column(Integer, nullable=False, index=True)
    subject = Column(String(100), nullable=False, index=True)
    audio_file_path = Column(Text)
    content_quality_score = Column(Float)
    audio_accuracy_score = Column(Float)
    created_at = Column(TIMESTAMP, default=utcnow, index=True)
    content_metadata = Column("metadata", JSONB)
    user_id = Column(UUID(as_uuid=True), index=True)  # Track owner

    # Composite indexes for common query patterns
    __table_args__ = (
        Index("idx_user_content", "user_id", "created_at"),
        Index("idx_grade_subject", "complexity_level", "subject"),
        Index("idx_language_grade", "language", "complexity_level"),
        Index("idx_subject_created", "subject", "created_at"),
    )

class ContentStandard(Base):
    """Reference database for content domain standards."""

    __tablename__ = "content_standards"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    complexity_level = Column(Integer, nullable=False, index=True)
    subject = Column(String(100), nullable=False, index=True)
    topic = Column(String(200), nullable=False)
    description = Column(Text)  # Added for test compatibility
    learning_objectives: Any = Column(ARRAY(Text))
    keywords: Any = Column(ARRAY(Text))

class ContentTranslation(Base):
    """Normalized table for content translations (extracted from ProcessedContent metadata)."""

    __tablename__ = "content_translations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processed_content.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    language = Column(String(50), nullable=False, index=True)
    translated_text = Column(Text, nullable=False)
    translation_model = Column(String(100))
    translation_quality_score = Column(Float)
    created_at = Column(TIMESTAMP, default=utcnow)

class ContentAudio(Base):
    """Normalized table for audio files (extracted from ProcessedContent)."""

    __tablename__ = "content_audio"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processed_content.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    language = Column(String(50), nullable=False)
    audio_file_path = Column(Text, nullable=False)
    audio_format = Column(String(20))
    duration_seconds = Column(Float)
    tts_model = Column(String(100))
    accuracy_score = Column(Float)
    created_at = Column(TIMESTAMP, default=utcnow)

class ContentValidation(Base):
    """Normalized table for validation results (content_domain, script, factual)."""

    __tablename__ = "content_validation"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processed_content.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    validation_type = Column(
        String(50), nullable=False, index=True
    )  # 'content_domain', 'script', 'factual'
    alignment_score = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False)
    issues_found = Column(JSONB)  # Structured validation issues
    validated_at = Column(TIMESTAMP, default=utcnow)

class Feedback(Base):
    """User feedback for content quality."""

    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processed_content.id"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    rating = Column(Integer, nullable=False)  # 1-5 stars
    feedback_text = Column(Text)
    issue_type = Column(String(100))  # e.g., 'translation', 'audio', 'simplification'
    created_at = Column(TIMESTAMP, default=utcnow, index=True)

class PipelineLog(Base):
    """Logs for pipeline processing stages and performance metrics."""

    __tablename__ = "pipeline_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(
        UUID(as_uuid=True), ForeignKey("processed_content.id"), index=True
    )
    stage = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False)
    processing_time_ms = Column(Integer)
    error_message = Column(Text)
    timestamp = Column(TIMESTAMP, default=utcnow, index=True)
