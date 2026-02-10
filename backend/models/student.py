"""
Student profile models.
Maps to existing student_profiles table for personalization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from sqlalchemy import TIMESTAMP, Column, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from ..database import Base


def utcnow():
    """Get current UTC time as naive datetime (for TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.utcnow()


class LearningStyle(str, Enum):
    """Student learning style preferences."""

    VISUAL = "visual"  # Prefers diagrams, charts, images
    AUDITORY = "auditory"  # Prefers audio explanations
    READING = "reading"  # Prefers text-based content
    KINESTHETIC = "kinesthetic"  # Prefers hands-on examples


class StudentProfile(Base):
    """
    Student profile for personalized AI responses.

    Maps to existing student_profiles table created in migrations.
    Used to personalize:
    - Response complexity based on grade_level
    - Language preferences
    - Subject focus areas
    - Learning style adaptations
    """

    __tablename__ = "student_profiles"

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
    grade_level = Column(Integer, nullable=False, default=8)
    subjects_of_interest = Column(ARRAY(Text), default=[])

    # Extended personalization (stored in JSONB for flexibility)
    offline_content_cache = Column(JSONB, default={})

    # Timestamps
    created_at = Column(TIMESTAMP, default=utcnow)
    updated_at = Column(TIMESTAMP, default=utcnow, onupdate=utcnow)

    def __repr__(self):
        return f"<StudentProfile(id={self.id}, grade={self.grade_level}, lang={self.language_preference})>"

    @property
    def learning_style(self) -> LearningStyle:
        """Get learning style from cache or default to reading."""
        style = (self.offline_content_cache or {}).get("learning_style", "reading")
        try:
            return LearningStyle(style)
        except ValueError:
            return LearningStyle.READING

    @learning_style.setter
    def learning_style(self, value: LearningStyle):
        """Set learning style in cache."""
        if self.offline_content_cache is None:
            self.offline_content_cache = {}
        self.offline_content_cache["learning_style"] = value.value

    @property
    def difficulty_preference(self) -> str:
        """Get difficulty preference: 'easy', 'medium', 'challenging'."""
        return (self.offline_content_cache or {}).get("difficulty", "medium")

    @difficulty_preference.setter
    def difficulty_preference(self, value: str):
        """Set difficulty preference."""
        if self.offline_content_cache is None:
            self.offline_content_cache = {}
        self.offline_content_cache["difficulty"] = value

    @property
    def interaction_count(self) -> int:
        """Get total AI interaction count."""
        return (self.offline_content_cache or {}).get("interaction_count", 0)

    def increment_interactions(self) -> int:
        """Increment and return interaction count."""
        if self.offline_content_cache is None:
            self.offline_content_cache = {}
        count = self.offline_content_cache.get("interaction_count", 0) + 1
        self.offline_content_cache["interaction_count"] = count
        return count

    def to_context_dict(self) -> dict[str, Any]:
        """Convert to dictionary for AI context injection."""
        return {
            "grade_level": self.grade_level,
            "language": self.language_preference,
            "subjects": self.subjects_of_interest or [],
            "learning_style": self.learning_style.value,
            "difficulty": self.difficulty_preference,
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def create_default(cls, user_id: str | None = None) -> "StudentProfile":
        """Create a default profile for new users."""
        import uuid

        return cls(
            id=uuid.uuid4(),
            user_id=user_id,
            language_preference="en",
            grade_level=8,
            subjects_of_interest=[],
            offline_content_cache={
                "learning_style": "reading",
                "difficulty": "medium",
                "interaction_count": 0,
            },
        )
