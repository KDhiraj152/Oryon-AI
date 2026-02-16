"""
Unit Tests for Curriculum Validation Service

Tests Issue #11 implementation
"""

from datetime import UTC, datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.models import ContentValidation, ContentStandard, ProcessedContent
from backend.services.content.content_validation import (
    ContentValidationService,
    validate_in_pipeline,
)


@pytest.fixture
def mock_db():
    """Mock database session."""
    db = Mock()
    db.add = Mock()
    db.commit = Mock()
    db.refresh = Mock()
    db.query = Mock()
    return db


@pytest.fixture
def mock_validator():
    """Mock content_domain validator."""
    validator = Mock()
    validator.validate_content = AsyncMock()
    validator.check_factual_accuracy = AsyncMock()
    validator.check_language_complexity = AsyncMock()
    return validator


@pytest.fixture
def sample_standards():
    """Sample content standards."""
    return [
        ContentStandard(
            id=1,
            subject="Mathematics",
            topic="Algebra",
            description="Basic algebraic expressions",
        ),
        ContentStandard(
            id=2,
            subject="Mathematics",
            topic="Geometry",
            description="Properties of triangles",
        ),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_content_against_content_domain_success(
    mock_db, mock_validator, sample_standards
):
    """Test successful content_domain validation."""
    # Setup
    mock_db.query.return_value.filter.return_value.all.return_value = sample_standards

    mock_validator.validate_content.return_value = {
        "alignment_score": 0.85,
        "errors": [],
        "warnings": [],
        "suggestions": ["Consider adding more examples"],
        "matched_topics": ["Algebra", "Geometry"],
        "missing_topics": [],
        "terminology_issues": [],
    }

    service = ContentValidationService(mock_db)
    service.validator = mock_validator

    # Execute
    validation = await service.validate_content_against_content_domain(
        content_id="test-content-123",
        complexity_level=10,
        subject="Mathematics",
        text="This lesson covers algebraic expressions and geometric properties.",
        language="en",
    )

    # Assert
    assert validation.content_id == "test-content-123"
    assert validation.validation_type == "content_domain"
    assert validation.alignment_score == 0.85
    assert validation.passed is True
    assert len(validation.issues_found["errors"]) == 0
    assert "Algebra" in validation.issues_found["matched_topics"]

    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_content_below_threshold(
    mock_db, mock_validator, sample_standards
):
    """Test validation failure when score below threshold."""
    # Setup
    mock_db.query.return_value.filter.return_value.all.return_value = sample_standards

    mock_validator.validate_content.return_value = {
        "alignment_score": 0.55,
        "errors": ["Topic coverage incomplete"],
        "warnings": ["Some terminology unclear"],
        "suggestions": ["Add coverage of quadratic equations"],
        "matched_topics": ["Algebra"],
        "missing_topics": ["Quadratic Equations", "Functions"],
        "terminology_issues": ["Unclear definition of 'variable'"],
    }

    service = ContentValidationService(mock_db)
    service.validator = mock_validator

    # Execute
    validation = await service.validate_content_against_content_domain(
        content_id="test-content-456",
        complexity_level=10,
        subject="Mathematics",
        text="Basic algebra content.",
        language="en",
    )

    # Assert
    assert validation.passed is False
    assert validation.alignment_score == 0.55
    assert len(validation.issues_found["errors"]) > 0
    assert "Quadratic Equations" in validation.issues_found["missing_topics"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_no_standards_available(mock_db, mock_validator):
    """Test handling when no content_domain standards exist."""
    # Setup - no standards found
    mock_db.query.return_value.filter.return_value.all.return_value = []

    service = ContentValidationService(mock_db)
    service.validator = mock_validator

    # Execute
    validation = await service.validate_content_against_content_domain(
        content_id="test-content-789",
        complexity_level=11,
        subject="Physics",
        text="Physics content.",
        language="en",
    )

    # Assert
    assert validation.passed is False
    assert validation.alignment_score == 0.0
    assert "No content_domain standards available" in validation.issues_found["errors"][0]

    # Validator should not be called
    mock_validator.validate_content.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_factual_accuracy(mock_db, mock_validator):
    """Test factual accuracy validation."""
    # Setup
    mock_validator.check_factual_accuracy.return_value = {
        "accuracy_score": 0.92,
        "factual_errors": [],
        "warnings": ["Verify latest scientific consensus on topic X"],
        "corrections": [],
    }

    service = ContentValidationService(mock_db)
    service.validator = mock_validator

    # Execute
    validation = await service.validate_factual_accuracy(
        content_id="test-content-fact",
        text="Water boils at 100°C at sea level.",
        subject="Science",
    )

    # Assert
    assert validation.validation_type == "factual"
    assert validation.passed is True
    assert validation.alignment_score == 0.92
    assert len(validation.issues_found["errors"]) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_factual_accuracy_failure(mock_db, mock_validator):
    """Test factual accuracy validation failure."""
    # Setup
    mock_validator.check_factual_accuracy.return_value = {
        "accuracy_score": 0.45,
        "factual_errors": [
            "Incorrect statement: Earth is flat",
            "Outdated information about speed of light",
        ],
        "warnings": [],
        "corrections": [
            "Earth is approximately spherical",
            "Update speed of light to 299,792,458 m/s",
        ],
    }

    service = ContentValidationService(mock_db)
    service.validator = mock_validator

    # Execute
    validation = await service.validate_factual_accuracy(
        content_id="test-content-bad-facts",
        text="The Earth is flat and light travels at 300,000 km/s exactly.",
        subject="Science",
    )

    # Assert
    assert validation.passed is False
    assert validation.alignment_score == 0.45
    assert len(validation.issues_found["errors"]) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_language_appropriateness(mock_db, mock_validator):
    """Test language complexity validation."""
    # Setup
    mock_validator.check_language_complexity.return_value = {
        "appropriateness_score": 0.88,
        "errors": [],
        "warnings": ["Some sentences are long"],
        "complex_words": ["photosynthesis", "chlorophyll"],
        "readability": {"flesch_reading_ease": 65.0, "flesch_kincaid_grade": 6.5},
        "simplification_suggestions": ["Break down complex sentences"],
    }

    service = ContentValidationService(mock_db)
    service.validator = mock_validator

    # Execute
    validation = await service.validate_language_appropriateness(
        content_id="test-content-lang",
        text="Photosynthesis is the process by which plants make food.",
        language="en",
        complexity_level=10,
    )

    # Assert
    assert validation.validation_type == "language"
    assert validation.passed is True
    assert "readability_metrics" in validation.issues_found
    assert validation.issues_found["readability_metrics"]["flesch_reading_ease"] == 65.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_comprehensive_validation(mock_db, mock_validator, sample_standards):
    """Test comprehensive validation runs all checks."""
    # Setup
    mock_db.query.return_value.filter.return_value.all.return_value = sample_standards

    mock_validator.validate_content.return_value = {
        "alignment_score": 0.80,
        "errors": [],
        "warnings": [],
        "suggestions": [],
        "matched_topics": ["Algebra"],
        "missing_topics": [],
        "terminology_issues": [],
    }

    mock_validator.check_factual_accuracy.return_value = {
        "accuracy_score": 0.85,
        "factual_errors": [],
        "warnings": [],
        "corrections": [],
    }

    mock_validator.check_language_complexity.return_value = {
        "appropriateness_score": 0.78,
        "errors": [],
        "warnings": [],
        "complex_words": [],
        "readability": {},
        "simplification_suggestions": [],
    }

    service = ContentValidationService(mock_db)
    service.validator = mock_validator

    # Execute
    validations = await service.comprehensive_validation(
        content_id="test-content-comp",
        text="Sample content.",
        subject="Mathematics",
        language="en",
        complexity_level=10,
    )

    # Assert
    assert len(validations) == 3
    assert "content_domain" in validations
    assert "factual" in validations
    assert "language" in validations
    assert all(v.passed for v in validations.values())


@pytest.mark.unit
def test_get_validation_summary(mock_db):
    """Test validation summary generation."""
    # Setup
    validations = [
        ContentValidation(
            content_id="test-123",
            validation_type="content_domain",
            alignment_score=0.85,
            passed=True,
            issues_found={"errors": [], "warnings": ["Minor issue"]},
            validated_at=datetime.now(UTC),
        ),
        ContentValidation(
            content_id="test-123",
            validation_type="factual",
            alignment_score=0.90,
            passed=True,
            issues_found={"errors": [], "warnings": []},
            validated_at=datetime.now(UTC),
        ),
        ContentValidation(
            content_id="test-123",
            validation_type="language",
            alignment_score=0.70,
            passed=False,
            issues_found={"errors": ["Too complex"], "warnings": []},
            validated_at=datetime.now(UTC),
        ),
    ]

    mock_db.query.return_value.filter.return_value.all.return_value = validations

    service = ContentValidationService(mock_db)

    # Execute
    summary = service.get_validation_summary("test-123")

    # Assert
    assert summary["validated"] is True
    assert summary["total_checks"] == 3
    assert summary["passed_checks"] == 2
    assert summary["failed_checks"] == 1
    assert summary["overall_passed"] is False
    assert 0.80 < summary["overall_score"] < 0.85


@pytest.mark.unit
def test_get_improvement_suggestions(mock_db):
    """Test improvement suggestions extraction."""
    # Setup
    validations = [
        ContentValidation(
            content_id="test-456",
            validation_type="content_domain",
            alignment_score=0.65,
            passed=False,
            issues_found={
                "errors": [],
                "warnings": [],
                "suggestions": ["Add more examples", "Cover missing topics"],
                "missing_topics": ["Quadratic Equations", "Functions"],
            },
        ),
        ContentValidation(
            content_id="test-456",
            validation_type="factual",
            alignment_score=0.70,
            passed=False,
            issues_found={
                "errors": ["Incorrect fact"],
                "suggestions": ["Verify sources"],
            },
        ),
    ]

    mock_db.query.return_value.filter.return_value.all.return_value = validations

    service = ContentValidationService(mock_db)

    # Execute
    suggestions = service.get_improvement_suggestions("test-456")

    # Assert
    assert len(suggestions) > 0
    assert "Add more examples" in suggestions
    assert "Review factual accuracy with subject experts" in suggestions
    assert any("Quadratic Equations" in s for s in suggestions)
