"""Validation module for educational content quality assurance."""

from .validation_module import ValidationModule, ValidationResult, QualityReport
from .ncert_standards import NCERTStandardsLoader, NCERTStandardData, initialize_ncert_standards

__all__ = [
    'ValidationModule',
    'ValidationResult', 
    'QualityReport',
    'NCERTStandardsLoader',
    'NCERTStandardData',
    'initialize_ncert_standards'
]
