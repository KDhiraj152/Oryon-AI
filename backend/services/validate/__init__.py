"""Validation module for educational content quality assurance."""

from .standards import (
    NCERTStandardData,
    NCERTStandardsLoader,
    initialize_ncert_standards,
)
from .validator import QualityReport, ValidationModule, ValidationResult

__all__ = [
    "NCERTStandardData",
    "NCERTStandardsLoader",
    "QualityReport",
    "ValidationModule",
    "ValidationResult",
    "initialize_ncert_standards",
]
