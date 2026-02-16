"""
User Profile Service
=======================

Manages user profiles - simplified version without personalization constraints.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class UserProfileService:
    """
    Service for loading and managing user profiles.

    Simplified version - no personalization constraints.
    Just stores user preferences like language and subjects.
    """

    def __init__(self, db: Session | None = None):
        # Only store externally provided session (for DI/testing)
        self._external_db = db
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: dict[str, float] = {}

    @contextmanager
    def _get_db(self) -> Generator[Session, None, None]:
        """
        Get database session as context manager.

        If external session was provided, uses that.
        Otherwise creates a new session and ensures proper cleanup.
        """
        if self._external_db is not None:
            # Use externally provided session (caller manages lifecycle)
            yield self._external_db
        else:
            # Create new session with proper cleanup
            from backend.db.database import SessionLocal

            session = SessionLocal()
            try:
                yield session
            finally:
                session.close()

    def get_profile(self, user_id: str) -> dict[str, Any]:
        """
        Get user profile for user, with caching.

        Returns context dict with user preferences:
        {
            "language": "en",
            "subjects": ["math", "science"],
        }
        """
        # Check cache first
        cached = self._get_cached(user_id)
        if cached:
            return cached

        # Load from database
        profile_dict = self._load_from_db(user_id)

        # Cache the result
        self._cache[user_id] = profile_dict
        self._cache_timestamps[user_id] = datetime.now(UTC).timestamp()

        return profile_dict

    def _get_cached(self, user_id: str) -> dict[str, Any] | None:
        """Get cached profile if still valid."""
        if user_id not in self._cache:
            return None

        # Check TTL
        cached_at = self._cache_timestamps.get(user_id, 0)
        age = datetime.now(UTC).timestamp() - cached_at

        if age > self._cache_ttl:
            del self._cache[user_id]
            del self._cache_timestamps[user_id]
            return None

        return self._cache[user_id]

    def _load_from_db(self, user_id: str) -> dict[str, Any]:
        """Load profile from database or return defaults."""
        try:
            from ..models.student import UserProfile

            with self._get_db() as db:
                profile = (
                    db.query(UserProfile)
                    .filter(UserProfile.user_id == UUID(user_id))
                    .first()
                )

                if profile:
                    logger.debug("Loaded profile for user %s", user_id)
                    return {
                        "language": profile.language_preference,
                        "subjects": profile.subjects_of_interest or [],
                    }

            # No profile found - return defaults
            logger.debug("No profile for user %s, using defaults", user_id)
            return self._default_profile()

        except (OSError, RuntimeError) as e:
            logger.warning("Failed to load profile for %s: %s", user_id, e)
            return self._default_profile()

    def _default_profile(self) -> dict[str, Any]:
        """Return default profile for unknown users."""
        return {
            "language": "en",
            "subjects": [],
        }

    def _create_new_profile(
        self,
        db,
        user_id: str,
        language: str | None,
        subjects: list | None,
    ):
        """Create a new user profile."""
        import uuid

        from ..models.student import UserProfile

        profile = UserProfile(
            id=uuid.uuid4(),
            user_id=UUID(user_id),
            language_preference=language or "en",
            complexity_level=8,  # Keep default for DB compatibility
            subjects_of_interest=subjects or [],
            offline_content_cache={},
        )
        db.add(profile)
        return profile

    def _update_existing_profile(
        self,
        profile,
        language: str | None,
        subjects: list | None,
    ) -> None:
        """Update fields on an existing profile."""
        if language is not None:
            profile.language_preference = language
        if subjects is not None:
            profile.subjects_of_interest = subjects

    def update_profile(
        self,
        user_id: str,
        language: str | None = None,
        subjects: list | None = None,
    ) -> dict[str, Any]:
        """
        Update user's profile and return updated dict.
        Creates profile if doesn't exist.
        """
        with self._get_db() as db:
            try:
                from ..models.student import UserProfile

                profile = (
                    db.query(UserProfile)
                    .filter(UserProfile.user_id == UUID(user_id))
                    .first()
                )

                if not profile:
                    profile = self._create_new_profile(db, user_id, language, subjects)
                else:
                    self._update_existing_profile(profile, language, subjects)

                db.commit()
                db.refresh(profile)

                # Invalidate cache
                self._cache.pop(user_id, None)

                logger.info("Updated profile for user %s", user_id)
                return {
                    "language": profile.language_preference,
                    "subjects": profile.subjects_of_interest or [],
                }

            except (OSError, RuntimeError) as e:
                logger.error("Failed to update profile: %s", e)
                db.rollback()
                raise

    def clear_cache(self):
        """Clear all cached profiles."""
        self._cache.clear()
        self._cache_timestamps.clear()

# Singleton instance (thread-safe)
_profile_service: UserProfileService | None = None

def get_profile_service(db: Session | None = None) -> UserProfileService:
    """Get or create the profile service singleton."""
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _profile_service
    if _profile_service is not None:
        return _profile_service
    _profile_service = UserProfileService(db)
    return _profile_service
