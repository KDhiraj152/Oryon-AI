"""
Content Quality Validator

Validates content against content domain standards.
"""

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from backend.models import ContentStandard
from .content_standards import ContentStandardsLoader

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of content validation."""

    alignment_score: float
    errors: list[str]
    warnings: list[str]
    suggestions: list[str]
    matched_topics: list[str]
    missing_topics: list[str]
    terminology_issues: list[str]

class ContentValidator:
    """Validates content against content domain standards."""

    def __init__(self, embedding_client: Any | None = None):
        """
        Initialize content_domain validator.

        Args:
            embedding_client: Optional embedding client for semantic matching
        """
        self.embedding_client = embedding_client
        self.standards_loader = ContentStandardsLoader(self.embedding_client)
        self.alignment_threshold = 0.70

    def validate_content(
        self,
        text: str,
        complexity_level: int | None,
        subject: str,
    ) -> dict[str, Any]:
        """
        Validate content against content domain standards.

        Args:
            text: Content text to validate
            complexity_level: Optional complexity level (1-12), None for unconstrained
            subject: Subject area
            standards: List of content standards to validate against

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating content for Grade %s, %s", complexity_level or 'any', subject)

        errors: list[str] = []
        warnings: list[str] = []
        suggestions: list[str] = []
        matched_topics: list[str] = []
        missing_topics: list[str] = []
        terminology_issues: list[str] = []

        # Find matching standards using embeddings
        matching_standards = self.standards_loader.find_matching_standards(
            content=text, complexity_level=complexity_level or 0, subject=subject, top_k=5
        )

        if not matching_standards:
            grade_msg = f"Grade {complexity_level}, " if complexity_level else ""
            errors.append(
                f"No matching content_domain standards found for {grade_msg}{subject}"
            )
            return {
                "alignment_score": 0.0,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
                "matched_topics": matched_topics,
                "missing_topics": missing_topics,
                "terminology_issues": terminology_issues,
            }

        # Calculate overall alignment score
        alignment_scores = []
        for standard_data, similarity in matching_standards:
            alignment_scores.append(similarity)

            if similarity >= self.alignment_threshold:
                matched_topics.append(standard_data.topic)
            else:
                missing_topics.append(standard_data.topic)

        overall_alignment = np.mean(alignment_scores) if alignment_scores else 0.0

        # Check keyword overlap
        for standard_data, _ in matching_standards:
            keyword_overlap = self.standards_loader.check_keyword_overlap(
                text, standard_data
            )
            if keyword_overlap < 0.3:
                warnings.append(
                    f"Low keyword overlap ({keyword_overlap:.1%}) for topic: {standard_data.topic}"
                )
                suggestions.append(
                    f"Consider adding keywords: {', '.join(standard_data.keywords[:3])}"
                )

        # Check learning objectives match
        for standard_data, _ in matching_standards[:3]:  # Check top 3 matches
            objectives_match = self.standards_loader.get_learning_objectives_match(
                text, standard_data
            )
            if objectives_match < 0.5:
                warnings.append(
                    f"Content may not fully address learning objectives for: {standard_data.topic}"
                )

        # Provide suggestions based on alignment score
        if overall_alignment < 0.5:
            grade_msg = f"Grade {complexity_level} " if complexity_level else ""
            errors.append(
                f"Content alignment too low ({overall_alignment:.1%}). "
                f"Content may not be appropriate for {grade_msg}{subject}."
            )
            suggestions.append(
                "Review content domain guidelines and adjust content to better match "
                "expected learning outcomes."
            )
        elif overall_alignment < self.alignment_threshold:
            warnings.append(
                f"Content alignment below threshold ({overall_alignment:.1%}). "
                "Consider improvements."
            )

        return {
            "alignment_score": float(overall_alignment),
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "matched_topics": matched_topics,
            "missing_topics": missing_topics,
            "terminology_issues": terminology_issues,
        }

    def check_factual_accuracy(self, text: str, subject: str) -> dict[str, Any]:
        """
        Check factual accuracy of content.

        Args:
            text: Content text
            subject: Subject area

        Returns:
            Dictionary with accuracy results
        """
        logger.info("Checking factual accuracy for %s", subject)

        # This is a simplified implementation
        # In production, you would use:
        # - Fact-checking APIs
        # - Knowledge graph validation
        # - Subject-specific fact databases

        factual_errors = []
        accuracy_score = 0.85  # Placeholder score

        # Basic checks (expand based on requirements)
        if not text or len(text.strip()) < 10:
            factual_errors.append("Content too short for accuracy validation")
            accuracy_score = 0.0

        return {
            "accuracy_score": accuracy_score,
            "factual_errors": factual_errors,
            "confidence": 0.8,
        }

    # Terminology rules: (subject, min_grade_for_complex, complex_terms)
    TERMINOLOGY_RULES: ClassVar[dict[str, tuple[int, list[str]]]] = {
        "mathematics": (11, ["calculus", "derivative", "integral"]),
        "science": (9, ["quantum", "thermodynamics", "electromagnetism"]),
    }

    def _check_complex_terms(
        self,
        text: str,
        terms: list[str],
        complexity_level: int,
        min_grade: int,
    ) -> list[str]:
        """Check if complex terms are used below appropriate complexity level."""
        if complexity_level >= min_grade:
            return []

        text_lower = text.lower()
        return [
            f"Advanced term '{term}' may be inappropriate for Grade {complexity_level}"
            for term in terms
            if term.lower() in text_lower
        ]

    def validate_terminology(
        self, text: str, subject: str, complexity_level: int | None = None
    ) -> list[str]:
        """
        Validate terminology appropriateness for complexity level.

        Args:
            text: Content text
            subject: Subject area
            complexity_level: Optional complexity level (None skips grade-based checks)

        Returns:
            List of terminology issues
        """
        if complexity_level is None:
            return []  # Skip grade-based terminology checks for unconstrained

        subject_lower = subject.lower()
        rule = self.TERMINOLOGY_RULES.get(subject_lower)

        if not rule:
            return []

        min_grade, complex_terms = rule
        return self._check_complex_terms(text, complex_terms, complexity_level, min_grade)
