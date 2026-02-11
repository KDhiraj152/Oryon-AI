"""
Progress tracking models.
"""

from datetime import datetime

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from ..database import Base


def utcnow():
    """Get current UTC time as naive datetime (for TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.utcnow()


class UserProgress(Base):
    """Track student progress on content items."""

    __tablename__ = "user_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processed_content.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # Progress data
    progress_percent = Column(Float, default=0.0)  # 0-100
    completed = Column(Boolean, default=False)
    started_at = Column(TIMESTAMP, default=utcnow)
    last_accessed = Column(TIMESTAMP, default=utcnow, onupdate=utcnow)
    completed_at = Column(TIMESTAMP, nullable=True)

    # Time tracking
    time_spent_seconds = Column(Integer, default=0)
    session_count = Column(Integer, default=0)

    # Additional data
    progress_data = Column(JSONB, default={})  # Custom progress data
    notes = Column(Text, nullable=True)

    def __repr__(self):
        return f"<UserProgress(user={self.user_id}, content={self.content_id}, progress={self.progress_percent}%)>"


class QuizScore(Base):
    """Track quiz scores and attempts."""

    __tablename__ = "quiz_scores"

    id = Column(Integer, primary_key=True, index=True)
    progress_id = Column(
        Integer, ForeignKey("user_progress.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processed_content.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    quiz_id = Column(String(255), nullable=False)

    # Score data
    score = Column(Float, nullable=False)  # 0-100
    max_score = Column(Float, default=100.0)
    passed = Column(Boolean, default=False)

    # Attempt tracking
    attempt_number = Column(Integer, default=1)
    time_taken_seconds = Column(Integer, nullable=True)
    submitted_at = Column(TIMESTAMP, default=utcnow)

    # Detailed data
    answers = Column(JSONB, default={})  # User answers
    correct_answers = Column(JSONB, default={})  # Correct answers
    question_scores = Column(JSONB, default={})  # Per-question scores

    def __repr__(self):
        return (
            f"<QuizScore(user={self.user_id}, quiz={self.quiz_id}, score={self.score})>"
        )


class InteractionSession(Base):
    """Track individual learning sessions."""

    __tablename__ = "interaction_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processed_content.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )  # Nullable

    # Session data
    started_at = Column(TIMESTAMP, default=utcnow)
    ended_at = Column(TIMESTAMP, nullable=True)
    duration_seconds = Column(Integer, default=0)

    # Activity data
    interactions = Column(Integer, default=0)
    pages_viewed = Column(Integer, default=0)
    videos_watched = Column(Integer, default=0)
    exercises_completed = Column(Integer, default=0)

    # Session metadata
    device_type = Column(String(50), nullable=True)
    network_status = Column(String(50), nullable=True)
    session_data = Column(JSONB, default={})

    def __repr__(self):
        return (
            f"<InteractionSession(user={self.user_id}, duration={self.duration_seconds}s)>"
        )


class UsageReport(Base):
    """Store generated parent reports."""

    __tablename__ = "usage_reports"

    id = Column(Integer, primary_key=True, index=True)
    user_ref_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    admin_ref_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # Report metadata
    report_type = Column(String(50), nullable=False)  # weekly, monthly, quarterly
    period_start = Column(TIMESTAMP, nullable=False)
    period_end = Column(TIMESTAMP, nullable=False)
    generated_at = Column(TIMESTAMP, default=utcnow)

    # Report data
    summary = Column(JSONB, default={})
    statistics = Column(JSONB, default={})
    achievements = Column(JSONB, default={})
    recommendations = Column(JSONB, default={})

    # Report file
    report_html = Column(Text, nullable=True)
    report_pdf_path = Column(String(500), nullable=True)

    def __repr__(self):
        return f"<UsageReport(student={self.user_ref_id}, type={self.report_type})>"


class Achievement(Base):
    """Track student achievements and milestones."""

    __tablename__ = "achievements"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # Achievement data
    achievement_type = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    icon = Column(String(255), nullable=True)

    # Achievement criteria
    criteria_met = Column(JSONB, default={})
    earned_at = Column(TIMESTAMP, default=utcnow)

    # Rewards
    points = Column(Integer, default=0)
    badge = Column(String(255), nullable=True)

    def __repr__(self):
        return f"<Achievement(user={self.user_id}, type={self.achievement_type})>"
