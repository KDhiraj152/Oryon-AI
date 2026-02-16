"""
Authentication models.
"""

import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Boolean, Column, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID

from backend.db.database import Base


def utcnow():
    """Get current UTC time as naive datetime (for TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.utcnow()

class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    organization = Column(String(255))
    role = Column(
        String(50), default="user", nullable=False
    )  # 'user', 'admin', 'educator'
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    created_at = Column(TIMESTAMP, default=utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=utcnow, onupdate=utcnow)
    last_login = Column(TIMESTAMP)

class APIKey(Base):
    """API key model for programmatic access."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(TIMESTAMP, default=utcnow, nullable=False)
    expires_at = Column(TIMESTAMP)
    last_used = Column(TIMESTAMP)

class TokenBlacklist(Base):
    """Blacklisted tokens (logout, rotation, compromise)."""

    __tablename__ = "token_blacklist"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token_jti = Column(String(255), unique=True, nullable=False, index=True)  # JWT ID
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    reason = Column(
        String(100), nullable=False
    )  # 'logout', 'rotation', 'compromise', 'expired'
    blacklisted_at = Column(TIMESTAMP, default=utcnow, nullable=False, index=True)
    expires_at = Column(
        TIMESTAMP, nullable=False
    )  # When token would have expired naturally

    def __repr__(self):
        return f"<TokenBlacklist(jti={self.token_jti}, reason={self.reason})>"

class RefreshToken(Base):
    """Store active refresh tokens for rotation tracking."""

    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token_jti = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    device_fingerprint = Column(String(255), index=True)  # Browser/device identifier
    ip_address = Column(String(50))
    user_agent = Column(Text)

    # Rotation tracking
    parent_jti = Column(String(255), index=True)  # JTI of token that created this one
    rotation_count = Column(Integer, default=0)  # Number of times rotated

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    last_used_at = Column(TIMESTAMP, default=utcnow)
    created_at = Column(TIMESTAMP, default=utcnow, nullable=False)
    expires_at = Column(TIMESTAMP, nullable=False)

    def __repr__(self):
        return f"<RefreshToken(user={self.user_id}, active={self.is_active})>"
