"""
Response Review Queue
=====================

Flags low-confidence AI responses for teacher review.
Provides quality control without blocking the user experience.

Features:
- Confidence threshold-based flagging
- Redis-backed review queue
- Teacher review workflow support
- Analytics on flagged responses
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    """Status of a flagged response."""

    PENDING = "pending"  # Awaiting teacher review
    APPROVED = "approved"  # Teacher approved
    REJECTED = "rejected"  # Teacher marked as incorrect
    IMPROVED = "improved"  # Teacher provided correction


class FlagReason(str, Enum):
    """Reason for flagging a response."""

    LOW_CONFIDENCE = "low_confidence"
    NO_SOURCES = "no_sources"
    SAFETY_CONCERN = "safety_concern"
    USER_REPORTED = "user_reported"
    CURRICULUM_MISMATCH = "curriculum_mismatch"


@dataclass
class FlaggedResponse:
    """A response flagged for review."""

    id: str
    conversation_id: str
    user_id: str | None

    # Original interaction
    user_query: str
    ai_response: str

    # Why it was flagged
    reason: FlagReason
    confidence: float
    sources_count: int

    # Metadata
    grade_level: int
    subject: str | None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Review data
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_id: str | None = None
    reviewed_at: datetime | None = None
    reviewer_notes: str | None = None
    corrected_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "user_query": self.user_query,
            "ai_response": self.ai_response,
            "reason": self.reason.value,
            "confidence": self.confidence,
            "sources_count": self.sources_count,
            "grade_level": self.grade_level,
            "subject": self.subject,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "reviewer_id": self.reviewer_id,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer_notes": self.reviewer_notes,
            "corrected_response": self.corrected_response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlaggedResponse":
        return cls(
            id=data["id"],
            conversation_id=data["conversation_id"],
            user_id=data.get("user_id"),
            user_query=data["user_query"],
            ai_response=data["ai_response"],
            reason=FlagReason(data["reason"]),
            confidence=data["confidence"],
            sources_count=data.get("sources_count", 0),
            grade_level=data.get("grade_level", 8),
            subject=data.get("subject"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(UTC),
            status=ReviewStatus(data.get("status", "pending")),
            reviewer_id=data.get("reviewer_id"),
            reviewed_at=datetime.fromisoformat(data["reviewed_at"])
            if data.get("reviewed_at")
            else None,
            reviewer_notes=data.get("reviewer_notes"),
            corrected_response=data.get("corrected_response"),
        )


class ResponseReviewQueue:
    """
    Queue for flagging and reviewing low-confidence responses.

    Thresholds:
    - confidence < 0.6: Flag for review
    - no sources on RAG query: Flag for review
    - safety concerns: Always flag

    Storage:
    - Redis list for pending queue
    - Redis hash for individual responses
    - Fallback to in-memory if Redis unavailable
    """

    # Thresholds
    CONFIDENCE_THRESHOLD = 0.6
    MIN_SOURCES_FOR_RAG = 1

    # Redis keys
    QUEUE_KEY = "review:queue:pending"
    RESPONSE_KEY_PREFIX = "review:response:"
    STATS_KEY = "review:stats"

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._fallback_queue: list[FlaggedResponse] = []
        self._fallback_responses: dict[str, FlaggedResponse] = {}

    def should_flag(
        self,
        confidence: float,
        sources_count: int = 0,
        used_rag: bool = False,
        safety_concerns: bool = False,
    ) -> tuple[bool, FlagReason | None]:
        """
        Determine if a response should be flagged.

        Returns (should_flag, reason) tuple.
        """
        if safety_concerns:
            return True, FlagReason.SAFETY_CONCERN

        if confidence < self.CONFIDENCE_THRESHOLD:
            return True, FlagReason.LOW_CONFIDENCE

        if used_rag and sources_count < self.MIN_SOURCES_FOR_RAG:
            return True, FlagReason.NO_SOURCES

        return False, None

    def flag_response(
        self,
        conversation_id: str,
        user_query: str,
        ai_response: str,
        confidence: float,
        reason: FlagReason,
        user_id: str | None = None,
        sources_count: int = 0,
        grade_level: int = 8,
        subject: str | None = None,
    ) -> FlaggedResponse:
        """
        Flag a response for teacher review.

        Returns the created FlaggedResponse.
        """
        flagged = FlaggedResponse(
            id=str(uuid4()),
            conversation_id=conversation_id,
            user_id=user_id,
            user_query=user_query[:1000],  # Truncate for storage
            ai_response=ai_response[:5000],  # Truncate for storage
            reason=reason,
            confidence=confidence,
            sources_count=sources_count,
            grade_level=grade_level,
            subject=subject,
        )

        self._store_flagged(flagged)
        logger.info(
            f"Flagged response {flagged.id} for review: "
            f"reason={reason.value}, confidence={confidence:.2f}"
        )

        return flagged

    def _store_flagged(self, flagged: FlaggedResponse):
        """Store flagged response in Redis or fallback."""
        if self._redis:
            try:
                # Store the response
                key = f"{self.RESPONSE_KEY_PREFIX}{flagged.id}"
                self._redis.setex(
                    key, 86400 * 7, json.dumps(flagged.to_dict())
                )  # 7 days TTL

                # Add to pending queue
                self._redis.lpush(self.QUEUE_KEY, flagged.id)

                # Update stats
                self._redis.hincrby(self.STATS_KEY, "total_flagged", 1)
                self._redis.hincrby(self.STATS_KEY, f"reason:{flagged.reason.value}", 1)

                return
            except Exception as e:
                logger.warning(f"Redis store failed, using fallback: {e}")

        # Fallback to in-memory
        self._fallback_responses[flagged.id] = flagged
        self._fallback_queue.append(flagged)

    def get_pending(self, limit: int = 20, offset: int = 0) -> list[FlaggedResponse]:
        """Get pending responses for review."""
        if self._redis:
            try:
                # Get IDs from queue
                ids = self._redis.lrange(self.QUEUE_KEY, offset, offset + limit - 1)

                responses = []
                for response_id in ids:
                    if isinstance(response_id, bytes):
                        response_id = response_id.decode()

                    key = f"{self.RESPONSE_KEY_PREFIX}{response_id}"
                    data = self._redis.get(key)
                    if data:
                        if isinstance(data, bytes):
                            data = data.decode()
                        responses.append(FlaggedResponse.from_dict(json.loads(data)))

                return responses
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        # Fallback
        pending = [r for r in self._fallback_queue if r.status == ReviewStatus.PENDING]
        return pending[offset : offset + limit]

    def get_by_id(self, response_id: str) -> FlaggedResponse | None:
        """Get a specific flagged response by ID."""
        if self._redis:
            try:
                key = f"{self.RESPONSE_KEY_PREFIX}{response_id}"
                data = self._redis.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode()
                    return FlaggedResponse.from_dict(json.loads(data))
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        return self._fallback_responses.get(response_id)

    def review(
        self,
        response_id: str,
        status: ReviewStatus,
        reviewer_id: str,
        notes: str | None = None,
        corrected_response: str | None = None,
    ) -> FlaggedResponse | None:
        """
        Submit a review for a flagged response.

        Args:
            response_id: ID of the flagged response
            status: New status (approved, rejected, improved)
            reviewer_id: ID of the reviewing teacher
            notes: Optional review notes
            corrected_response: Corrected response if status is IMPROVED

        Returns updated FlaggedResponse or None if not found.
        """
        flagged = self.get_by_id(response_id)
        if not flagged:
            logger.warning(f"Response {response_id} not found for review")
            return None

        # Update the response
        flagged.status = status
        flagged.reviewer_id = reviewer_id
        flagged.reviewed_at = datetime.now(UTC)
        flagged.reviewer_notes = notes
        if corrected_response:
            flagged.corrected_response = corrected_response

        # Store updated response
        if self._redis:
            try:
                key = f"{self.RESPONSE_KEY_PREFIX}{response_id}"
                self._redis.setex(
                    key, 86400 * 30, json.dumps(flagged.to_dict())
                )  # 30 days

                # Remove from pending queue
                self._redis.lrem(self.QUEUE_KEY, 1, response_id)

                # Update stats
                self._redis.hincrby(self.STATS_KEY, f"status:{status.value}", 1)
            except Exception as e:
                logger.warning(f"Redis update failed: {e}")
        else:
            self._fallback_responses[response_id] = flagged

        logger.info(f"Response {response_id} reviewed: status={status.value}")
        return flagged

    def get_stats(self) -> dict[str, Any]:
        """Get review queue statistics."""
        if self._redis:
            try:
                stats = self._redis.hgetall(self.STATS_KEY)
                pending_count = self._redis.llen(self.QUEUE_KEY)

                # Decode bytes to strings
                decoded = {}
                for k, v in stats.items():
                    key = k.decode() if isinstance(k, bytes) else k
                    val = int(v.decode() if isinstance(v, bytes) else v)
                    decoded[key] = val

                decoded["pending_count"] = pending_count
                return decoded
            except Exception as e:
                logger.warning(f"Redis stats failed: {e}")

        # Fallback stats
        pending = len(
            [r for r in self._fallback_queue if r.status == ReviewStatus.PENDING]
        )
        return {
            "total_flagged": len(self._fallback_responses),
            "pending_count": pending,
        }

    def get_pending_count(self) -> int:
        """Get count of pending reviews."""
        if self._redis:
            try:
                return self._redis.llen(self.QUEUE_KEY)
            except Exception:
                pass

        return len(
            [r for r in self._fallback_queue if r.status == ReviewStatus.PENDING]
        )


# Singleton instance (thread-safe)
_review_queue: ResponseReviewQueue | None = None
_review_queue_lock = None  # Lazy init to avoid import-time issues


def get_review_queue(redis_client=None) -> ResponseReviewQueue:
    """Get or create the review queue singleton (thread-safe with double-checked locking)."""
    global _review_queue, _review_queue_lock
    import threading

    if _review_queue_lock is None:
        _review_queue_lock = threading.Lock()
    if _review_queue is None:
        with _review_queue_lock:
            if _review_queue is None:
                _review_queue = ResponseReviewQueue(redis_client)
    return _review_queue
