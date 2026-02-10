"""
Explainability Module - Transparency and Auditability
======================================================

Provides explainability features:
- Retrieval hit analysis
- Reasoning summary (internal, redacted for users)
- Token accounting
- Confidence breakdowns
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReasoningStage(str, Enum):
    """Stages in the reasoning process."""

    INPUT_PROCESSING = "input_processing"
    INTENT_DETECTION = "intent_detection"
    RETRIEVAL = "retrieval"
    CONTEXT_BUILDING = "context_building"
    GENERATION = "generation"
    VALIDATION = "validation"
    FORMATTING = "formatting"


@dataclass
class RetrievalHit:
    """A single retrieval hit from the vector database."""

    chunk_id: str
    text_preview: str
    similarity_score: float
    document_id: str | None = None
    document_title: str | None = None
    page_or_section: str | None = None
    was_used: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text_preview": self.text_preview[:200] + "..."
            if len(self.text_preview) > 200
            else self.text_preview,
            "similarity_score": round(self.similarity_score, 4),
            "document_id": self.document_id,
            "document_title": self.document_title,
            "page_or_section": self.page_or_section,
            "was_used": self.was_used,
        }


@dataclass
class StageResult:
    """Result from a reasoning stage."""

    stage: ReasoningStage
    duration_ms: float
    success: bool
    output_summary: str
    tokens_used: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "output_summary": self.output_summary,
            "tokens_used": self.tokens_used,
        }


@dataclass
class TokenAccounting:
    """Detailed token usage breakdown."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_tokens: int = 0
    system_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = (
                self.prompt_tokens
                + self.completion_tokens
                + self.context_tokens
                + self.system_tokens
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "context_tokens": self.context_tokens,
            "system_tokens": self.system_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


@dataclass
class ConfidenceBreakdown:
    """Breakdown of confidence factors."""

    retrieval_confidence: float = 1.0
    model_confidence: float = 1.0
    validation_score: float = 1.0
    factual_grounding: float = 1.0
    overall_confidence: float = 1.0
    factors: dict[str, float] = field(default_factory=dict)

    def compute_overall(self) -> float:
        """Compute overall confidence from factors."""
        scores = [
            self.retrieval_confidence,
            self.model_confidence,
            self.validation_score,
            self.factual_grounding,
        ]
        # Weighted geometric mean
        weights = [0.3, 0.3, 0.2, 0.2]

        weighted_sum = sum(s * w for s, w in zip(scores, weights, strict=False))
        self.overall_confidence = weighted_sum
        return self.overall_confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_confidence": round(self.retrieval_confidence, 3),
            "model_confidence": round(self.model_confidence, 3),
            "validation_score": round(self.validation_score, 3),
            "factual_grounding": round(self.factual_grounding, 3),
            "overall_confidence": round(self.overall_confidence, 3),
            "factors": {k: round(v, 3) for k, v in self.factors.items()},
        }


@dataclass
class ExplainabilityReport:
    """
    Complete explainability report for a response.

    This provides transparency into how a response was generated,
    what sources were used, and confidence levels.
    """

    request_id: str
    query: str
    response_summary: str

    # Processing details
    stages: list[StageResult] = field(default_factory=list)
    retrieval_hits: list[RetrievalHit] = field(default_factory=list)

    # Token accounting
    tokens: TokenAccounting = field(default_factory=TokenAccounting)

    # Confidence
    confidence: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)

    # Timing
    total_latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Model info
    model_id: str = ""
    model_tier: str = ""

    # Internal reasoning (redacted in user-facing version)
    internal_reasoning: str = ""  # Not included in to_dict

    def to_dict(self, include_internal: bool = False) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "request_id": self.request_id,
            "query": self.query[:200] + "..." if len(self.query) > 200 else self.query,
            "response_summary": self.response_summary,
            "stages": [s.to_dict() for s in self.stages],
            "retrieval_hits": [h.to_dict() for h in self.retrieval_hits],
            "tokens": self.tokens.to_dict(),
            "confidence": self.confidence.to_dict(),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "model_tier": self.model_tier,
        }

        if include_internal and self.internal_reasoning:
            result["internal_reasoning_summary"] = self._redact_reasoning(
                self.internal_reasoning
            )

        return result

    def _redact_reasoning(self, reasoning: str) -> str:
        """Redact sensitive details from internal reasoning."""
        # Remove any specific user data
        import re

        redacted = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", reasoning
        )
        redacted = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", redacted)

        # Truncate if too long
        if len(redacted) > 500:
            redacted = redacted[:500] + "... [truncated]"

        return redacted

    def add_stage(
        self,
        stage: ReasoningStage,
        duration_ms: float,
        success: bool,
        summary: str,
        tokens: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add a reasoning stage result."""
        self.stages.append(
            StageResult(
                stage=stage,
                duration_ms=duration_ms,
                success=success,
                output_summary=summary,
                tokens_used=tokens,
                details=details or {},
            )
        )

    def add_retrieval_hit(
        self,
        chunk_id: str,
        text: str,
        score: float,
        document_id: str | None = None,
        title: str | None = None,
        location: str | None = None,
        was_used: bool = True,
    ) -> None:
        """Add a retrieval hit."""
        self.retrieval_hits.append(
            RetrievalHit(
                chunk_id=chunk_id,
                text_preview=text[:300] if len(text) > 300 else text,
                similarity_score=score,
                document_id=document_id,
                document_title=title,
                page_or_section=location,
                was_used=was_used,
            )
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of the explainability report."""
        lines = [
            "## Response Explainability Report",
            "",
            f"**Request ID:** {self.request_id}",
            f"**Model:** {self.model_id} ({self.model_tier})",
            f"**Latency:** {self.total_latency_ms:.0f}ms",
            f"**Confidence:** {self.confidence.overall_confidence:.0%}",
            "",
            "### Token Usage",
            f"- Prompt: {self.tokens.prompt_tokens}",
            f"- Completion: {self.tokens.completion_tokens}",
            f"- Context: {self.tokens.context_tokens}",
            f"- Total: {self.tokens.total_tokens}",
            "",
            "### Processing Stages",
        ]

        for stage in self.stages:
            status = "✓" if stage.success else "✗"
            lines.append(f"- {status} {stage.stage.value}: {stage.duration_ms:.0f}ms")

        if self.retrieval_hits:
            lines.extend(
                [
                    "",
                    f"### Retrieved Sources ({len(self.retrieval_hits)})",
                ]
            )
            for i, hit in enumerate(self.retrieval_hits[:3], 1):
                used = "used" if hit.was_used else "discarded"
                lines.append(
                    f"{i}. [{hit.similarity_score:.0%}] {hit.document_title or hit.chunk_id} ({used})"
                )

        return "\n".join(lines)


class Explainer:
    """
    Creates and manages explainability reports.

    Features:
    - Track all reasoning stages
    - Record retrieval hits
    - Compute confidence scores
    - Generate human-readable summaries
    """

    def __init__(self):
        self._reports: dict[str, ExplainabilityReport] = {}
        self._max_cached_reports = 100

    def create_report(
        self,
        request_id: str,
        query: str,
        model_id: str = "",
        model_tier: str = "",
    ) -> ExplainabilityReport:
        """Create a new explainability report."""
        report = ExplainabilityReport(
            request_id=request_id,
            query=query,
            response_summary="",
            model_id=model_id,
            model_tier=model_tier,
        )

        # Cache report
        self._reports[request_id] = report

        # Cleanup old reports if needed
        if len(self._reports) > self._max_cached_reports:
            oldest_keys = list(self._reports.keys())[: -self._max_cached_reports]
            for key in oldest_keys:
                del self._reports[key]

        return report

    def get_report(self, request_id: str) -> ExplainabilityReport | None:
        """Get an existing report by request ID."""
        return self._reports.get(request_id)

    def finalize_report(
        self,
        report: ExplainabilityReport,
        response: str,
        total_latency_ms: float,
    ) -> ExplainabilityReport:
        """Finalize a report with response details."""
        report.response_summary = (
            response[:200] + "..." if len(response) > 200 else response
        )
        report.total_latency_ms = total_latency_ms

        # Compute overall confidence
        report.confidence.compute_overall()

        # Compute total tokens from stages
        report.tokens.total_tokens = sum(s.tokens_used for s in report.stages)

        return report

    def compute_retrieval_confidence(self, hits: list[RetrievalHit]) -> float:
        """Compute confidence score from retrieval hits."""
        if not hits:
            return 0.5  # No retrieval = moderate confidence (pure generation)

        # Average similarity of used hits
        used_hits = [h for h in hits if h.was_used]
        if not used_hits:
            return 0.5

        avg_similarity = sum(h.similarity_score for h in used_hits) / len(used_hits)

        # Boost if multiple good hits
        hit_count_bonus = min(0.1 * len(used_hits), 0.2)

        return min(avg_similarity + hit_count_bonus, 1.0)

    def compute_factual_grounding(
        self,
        response: str,
        retrieved_texts: list[str],
    ) -> float:
        """Compute how well the response is grounded in retrieved sources."""
        if not retrieved_texts:
            return 0.5  # No sources = moderate grounding

        # Simple word overlap check
        response_words = set(response.lower().split())
        source_words = set()
        for text in retrieved_texts:
            source_words.update(text.lower().split())

        # Remove common words
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "also",
        }

        response_words -= stopwords
        source_words -= stopwords

        if not response_words:
            return 0.5

        overlap = len(response_words & source_words)
        grounding = overlap / len(response_words)

        return min(grounding * 1.2, 1.0)  # Slight boost for good grounding
