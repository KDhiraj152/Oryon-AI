"""
Chat conversation models for the frontend chat interface.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Column, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from ..database import Base


def utcnow():
    """Get current UTC time as naive datetime (for TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.utcnow()


class MessageRole(str, enum.Enum):
    """Message role enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base):
    """Stores chat conversations for users."""

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = Column(String(255), default="New Chat")
    created_at = Column(TIMESTAMP(timezone=True), default=utcnow)
    updated_at = Column(TIMESTAMP(timezone=True), default=utcnow, onupdate=utcnow)
    extra_data = Column(
        "metadata", JSONB, default=dict
    )  # For additional conversation settings

    # Relationships
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "id": str(self.id),
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.extra_data or {},
        }


class Message(Base):
    """Stores individual messages within conversations."""

    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), default=MessageRole.USER.value)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=utcnow)
    extra_data = Column(
        "metadata", JSONB, default=dict
    )  # For attachments, sources, etc.

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.extra_data or {},
        }


class FileUpload(Base):
    """Stores file upload metadata."""

    __tablename__ = "file_uploads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    content_type = Column(String(100))
    size = Column(String(50))  # Human-readable size
    file_size = Column(String(50))  # Bytes
    path = Column(String(1000))  # Storage path
    url = Column(String(1000))  # Access URL
    created_at = Column(TIMESTAMP(timezone=True), default=utcnow)
    extra_data = Column("metadata", JSONB, default=dict)

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "id": str(self.id),
            "filename": self.filename,
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "size": self.size,
            "url": self.url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
