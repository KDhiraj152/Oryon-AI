"""
Curriculum Validator Integration Service

Issue: CODE-REVIEW-GPT #11 (HIGH)
Problem: Curriculum validator exists but not integrated into content pipeline

Solution: Integrate NCERT curriculum validation into content processing flow
"""

import logging
from typing import Any, Dict, List, Tuple

from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models import ContentValidation, NCERTStandard, ProcessedContent
from .validate.ncert import NCERTValidator

logger = logging.getLogger(__name__)


# Use lazy settings getter instead of creating new instance
def _get_settings():
    return get_settings()


class CurriculumValidationService:
    """Service for validating content against curriculum standards."""

    def __init__(self, db: Session):
        self.db = db
        self.validator = NCERTValidator()
        self.alignment_threshold = 0.70  # 70% minimum alignment required

    async def validate_content_against_curriculum(
        self,
        content_id: str,
        grade_level: int,
        subject: str,
        text: str,
        language: str = "en",
    ) -> ContentValidation:
        """
        Validate content against NCERT curriculum standards.

        Args:
            content_id: Content ID to validate
            grade_level: Grade level (1-12)
            subject: Subject area
            text: Content text to validate
            language: Content language

        Returns:
            ContentValidation object with results
        """
        logger.info(
            f"Validating content {content_id} against curriculum (Grade {grade_level}, {subject})"
        )

        # Get relevant curriculum standards
        standards = (
            self.db.query(NCERTStandard)
            .filter(
                NCERTStandard.grade_level == grade_level,
                NCERTStandard.subject == subject,
            )
            .all()
        )

        if not standards:
            logger.warning(
                f"No curriculum standards found for Grade {grade_level}, {subject}"
            )
            # Create validation record with warning
            validation = ContentValidation(
                content_id=content_id,
                validation_type="ncert",
                alignment_score=0.0,
                passed=False,
                issues_found={
                    "errors": [
                        f"No curriculum standards available for Grade {grade_level}, {subject}"
                    ],
                    "warnings": [],
                    "suggestions": [
                        "Add curriculum standards for this grade/subject combination"
                    ],
                },
            )
            self.db.add(validation)
            self.db.commit()
            return validation

        # Perform validation
        validation_result = await self.validator.validate_content(
            text=text, grade_level=grade_level, subject=subject, standards=standards
        )

        # Analyze results
        alignment_score = validation_result.get("alignment_score", 0.0)
        passed = alignment_score >= self.alignment_threshold

        # Extract issues
        issues = {
            "errors": validation_result.get("errors", []),
            "warnings": validation_result.get("warnings", []),
            "suggestions": validation_result.get("suggestions", []),
            "matched_topics": validation_result.get("matched_topics", []),
            "missing_topics": validation_result.get("missing_topics", []),
            "terminology_issues": validation_result.get("terminology_issues", []),
        }

        # Create validation record
        validation = ContentValidation(
            content_id=content_id,
            validation_type="ncert",
            alignment_score=alignment_score,
            passed=passed,
            issues_found=issues,
        )

        self.db.add(validation)
        self.db.commit()
        self.db.refresh(validation)

        logger.info(
            f"Validation complete: {alignment_score:.2%} alignment, "
            f"{'PASSED' if passed else 'FAILED'}, "
            f"{len(issues['errors'])} errors, {len(issues['warnings'])} warnings"
        )

        return validation

    async def validate_factual_accuracy(
        self, content_id: str, text: str, subject: str
    ) -> ContentValidation:
        """
        Validate factual accuracy using knowledge base.

        Args:
            content_id: Content ID
            text: Content text
            subject: Subject area

        Returns:
            ContentValidation object
        """
        logger.info(f"Validating factual accuracy for content {content_id}")

        # Use factual validator
        result = await self.validator.check_factual_accuracy(text=text, subject=subject)

        alignment_score = result.get("accuracy_score", 0.0)
        passed = alignment_score >= 0.80  # 80% accuracy required

        issues = {
            "errors": result.get("factual_errors", []),
            "warnings": result.get("warnings", []),
            "suggestions": result.get("corrections", []),
        }

        validation = ContentValidation(
            content_id=content_id,
            validation_type="factual",
            alignment_score=alignment_score,
            passed=passed,
            issues_found=issues,
        )

        self.db.add(validation)
        self.db.commit()
        self.db.refresh(validation)

        return validation

    async def validate_language_appropriateness(
        self, content_id: str, text: str, grade_level: int, language: str
    ) -> ContentValidation:
        """
        Validate language complexity for grade level.

        Args:
            content_id: Content ID
            text: Content text
            grade_level: Target grade level
            language: Content language

        Returns:
            ContentValidation object
        """
        logger.info(f"Validating language appropriateness for Grade {grade_level}")

        result = await self.validator.check_language_complexity(
            text=text, grade_level=grade_level, language=language
        )

        alignment_score = result.get("appropriateness_score", 0.0)
        passed = alignment_score >= 0.75

        issues = {
            "errors": result.get("errors", []),
            "warnings": result.get("warnings", []),
            "complex_words": result.get("complex_words", []),
            "readability_metrics": result.get("readability", {}),
            "suggestions": result.get("simplification_suggestions", []),
        }

        validation = ContentValidation(
            content_id=content_id,
            validation_type="language",
            alignment_score=alignment_score,
            passed=passed,
            issues_found=issues,
        )

        self.db.add(validation)
        self.db.commit()
        self.db.refresh(validation)

        return validation

    async def comprehensive_validation(
        self,
        content_id: str,
        text: str,
        grade_level: int,
        subject: str,
        language: str = "en",
    ) -> dict[str, ContentValidation]:
        """
        Run all validation checks comprehensively.

        Returns:
            Dictionary with validation results for each type
        """
        logger.info(f"Running comprehensive validation for content {content_id}")

        validations = {}

        # Run all validations in parallel
        try:
            # NCERT curriculum validation
            validations["ncert"] = await self.validate_content_against_curriculum(
                content_id, grade_level, subject, text, language
            )

            # Factual accuracy validation
            validations["factual"] = await self.validate_factual_accuracy(
                content_id, text, subject
            )

            # Language appropriateness validation
            validations["language"] = await self.validate_language_appropriateness(
                content_id, text, grade_level, language
            )

        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            raise

        # Calculate overall validation status
        all_passed = all(v.passed for v in validations.values())
        avg_score = sum(v.alignment_score for v in validations.values()) / len(
            validations
        )

        logger.info(
            f"Comprehensive validation complete: "
            f"Overall {'PASSED' if all_passed else 'FAILED'}, "
            f"Average score: {avg_score:.2%}"
        )

        return validations

    def get_validation_summary(self, content_id: str) -> dict[str, Any]:
        """Get summary of all validations for content."""
        validations = (
            self.db.query(ContentValidation)
            .filter(ContentValidation.content_id == content_id)
            .all()
        )

        if not validations:
            return {"validated": False, "message": "No validation records found"}

        summary = {
            "validated": True,
            "total_checks": len(validations),
            "passed_checks": sum(1 for v in validations if v.passed),
            "failed_checks": sum(1 for v in validations if not v.passed),
            "validations": {},
        }

        for validation in validations:
            summary["validations"][validation.validation_type] = {
                "score": validation.alignment_score,
                "passed": validation.passed,
                "errors_count": len(validation.issues_found.get("errors", [])),
                "warnings_count": len(validation.issues_found.get("warnings", [])),
                "validated_at": validation.validated_at.isoformat(),
            }

        # Overall status
        summary["overall_passed"] = summary["passed_checks"] == summary["total_checks"]
        summary["overall_score"] = sum(v.alignment_score for v in validations) / len(
            validations
        )

        return summary

    def get_improvement_suggestions(self, content_id: str) -> list[str]:
        """Get consolidated improvement suggestions from all validations."""
        validations = (
            self.db.query(ContentValidation)
            .filter(
                ContentValidation.content_id == content_id,
                ContentValidation.passed.is_(False),
            )
            .all()
        )

        suggestions = []
        for validation in validations:
            issues = validation.issues_found or {}
            suggestions.extend(issues.get("suggestions", []))

            # Add specific suggestions based on validation type
            if validation.validation_type == "ncert":
                if issues.get("missing_topics"):
                    suggestions.append(
                        f"Add coverage of: {', '.join(issues['missing_topics'][:3])}"
                    )
            elif validation.validation_type == "factual":
                if issues.get("errors"):
                    suggestions.append("Review factual accuracy with subject experts")
            elif validation.validation_type == "language":
                if validation.alignment_score < 0.6:
                    suggestions.append("Simplify language for target grade level")

        return list(set(suggestions))  # Remove duplicates


# Integration with content pipeline
async def validate_in_pipeline(
    db: Session, content: ProcessedContent, text: str
) -> tuple[bool, dict[str, Any]]:
    """
    Integrate validation into content processing pipeline.

    Returns:
        Tuple of (all_passed, validation_summary)
    """
    service = CurriculumValidationService(db)

    try:
        # Run comprehensive validation
        await service.comprehensive_validation(
            content_id=str(content.id),
            text=text,
            grade_level=content.grade_level,
            subject=content.subject,
            language=content.language,
        )

        # Get summary
        summary = service.get_validation_summary(str(content.id))

        # Update content with validation scores
        if summary["overall_passed"]:
            content.ncert_alignment_score = summary["overall_score"] * 100

        db.commit()

        return summary["overall_passed"], summary

    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}", exc_info=True)
        return False, {"error": str(e)}
