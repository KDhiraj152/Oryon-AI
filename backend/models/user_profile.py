"""
User profile models.
Maps to existing user_profiles table for personalization.
"""

from datetime import datetime
from enum import Enum, StrEnum
from typing import Any

from sqlalchemy import TIMESTAMP, Column, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from backend.db.database import Base


def utcnow():
    """Get current UTC time as naive datetime (for TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.utcnow()

class UserPreference(StrEnum):
    """User learning style preferences."""

    VISUAL = "visual"  # Prefers diagrams, charts, images
    AUDITORY = "auditory"  # Prefers audio explanations
    READING = "reading"  # Prefers text-based content
    KINESTHETIC = "kinesthetic"  # Prefers hands-on examples

class UserProfile(Base):
    """
    User profile for personalized AI responses.

    Maps to existing user_profiles table created in migrations.
    Used to personalize:
    - Response complexity based on complexity_level
    - Language preferences
    - Subject focus areas
    - Learning style adaptations
    """

    __tablename__ = "user_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=True,
    )

    # Core preferences (from original schema)
    language_preference = Column(String(50), nullable=False, default="en")
    complexity_level = Column(Integer, nullable=False, default=8)
    subjects_of_interest: Any = Column(ARRAY(Text), default=[])

    # Extended personalization (stored in JSONB for flexibility)
    offline_content_cache = Column(JSONB, default={})

    # Timestamps
    created_at = Column(TIMESTAMP, default=utcnow)
    updated_at = Column(TIMESTAMP, default=utcnow, onupdate=utcnow)

    def _get_cache(self) -> dict[str, Any]:
        """Get offline_content_cache as a typed dict."""
        cache = self.offline_content_cache
        if isinstance(cache, dict):
            return cache
        return {}

    def __repr__(self):
        return f"<UserProfile(id={self.id}, grade={self.complexity_level}, lang={self.language_preference})>"

    @property
    def user_preference(self) -> UserPreference:
        """Get learning style from cache or default to reading."""
        style = self._get_cache().get("user_preference", "reading")
        try:
            return UserPreference(style)
        except ValueError:
            return UserPreference.READING

    @user_preference.setter
    def user_preference(self, value: UserPreference):
        """Set learning style in cache."""
        if self.offline_content_cache is None:
            self.offline_content_cache = {}
        self.offline_content_cache["user_preference"] = value.value

    @property
    def difficulty_preference(self) -> str:
        """Get difficulty preference: 'easy', 'medium', 'challenging'."""
        return str(self._get_cache().get("difficulty", "medium"))

    @difficulty_preference.setter
    def difficulty_preference(self, value: str):
        """Set difficulty preference."""
        if self.offline_content_cache is None:
            self.offline_content_cache = {}
        self.offline_content_cache["difficulty"] = value

    @property
    def interaction_count(self) -> int:
        """Get total AI interaction count."""
        return int(self._get_cache().get("interaction_count", 0))

    def increment_interactions(self) -> int:
        """Increment and return interaction count."""
        if self.offline_content_cache is None:
            self.offline_content_cache = {}
        count = int(self._get_cache().get("interaction_count", 0)) + 1
        self.offline_content_cache["interaction_count"] = count
        return count

    def to_context_dict(self) -> dict[str, Any]:
        """Convert to dictionary for AI context injection."""
        return {
            "complexity_level": self.complexity_level,
            "language": self.language_preference,
            "subjects": self.subjects_of_interest or [],
            "user_preference": self.user_preference.value,
            "difficulty": self.difficulty_preference,
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def create_default(cls, user_id: str | None = None) -> "UserProfile":
        """Create a default profile for new users."""
        import uuid

        return cls(
            id=uuid.uuid4(),
            user_id=user_id,
            language_preference="en",
            complexity_level=8,
            subjects_of_interest=[],
            offline_content_cache={
                "user_preference": "reading",
                "difficulty": "medium",
                "interaction_count": 0,
            },
        )
