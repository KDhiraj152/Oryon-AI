"""
Grade-Level Adaptation Service

Issue: CODE-REVIEW-GPT #13 (HIGH)
Problem: No grade-level content adaptation

Solution: Adaptive complexity, vocabulary filtering, readability analysis
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..models import ProcessedContent

logger = logging.getLogger(__name__)


class ReadabilityLevel(str, Enum):
    """Readability levels for content."""

    VERY_EASY = "very_easy"  # Grade 1-3
    EASY = "easy"  # Grade 4-5
    MODERATE = "moderate"  # Grade 6-8
    DIFFICULT = "difficult"  # Grade 9-10
    VERY_DIFFICULT = "very_difficult"  # Grade 11-12


@dataclass
class GradeLevelConfig:
    """Configuration for each grade level."""

    grade_range: tuple[int, int]
    max_sentence_length: int
    max_word_length: int
    max_syllables: int
    min_flesch_score: float
    max_flesch_kincaid_grade: float
    vocabulary_complexity: str
    recommended_examples: list[str]


class GradeLevelAdaptationService:
    """Service for adapting content to appropriate grade levels."""

    # Grade level configurations
    GRADE_CONFIGS = {
        (1, 3): GradeLevelConfig(
            grade_range=(1, 3),
            max_sentence_length=10,
            max_word_length=8,
            max_syllables=3,
            min_flesch_score=90.0,
            max_flesch_kincaid_grade=3.0,
            vocabulary_complexity="very_simple",
            recommended_examples=["counting", "colors", "shapes", "animals"],
        ),
        (4, 5): GradeLevelConfig(
            grade_range=(4, 5),
            max_sentence_length=15,
            max_word_length=10,
            max_syllables=4,
            min_flesch_score=80.0,
            max_flesch_kincaid_grade=5.0,
            vocabulary_complexity="simple",
            recommended_examples=["basic math", "plants", "seasons", "family"],
        ),
        (6, 8): GradeLevelConfig(
            grade_range=(6, 8),
            max_sentence_length=20,
            max_word_length=12,
            max_syllables=5,
            min_flesch_score=70.0,
            max_flesch_kincaid_grade=8.0,
            vocabulary_complexity="moderate",
            recommended_examples=[
                "fractions",
                "geography",
                "history",
                "science experiments",
            ],
        ),
        (9, 10): GradeLevelConfig(
            grade_range=(9, 10),
            max_sentence_length=25,
            max_word_length=15,
            max_syllables=6,
            min_flesch_score=60.0,
            max_flesch_kincaid_grade=10.0,
            vocabulary_complexity="advanced",
            recommended_examples=["algebra", "chemistry", "literature", "civics"],
        ),
        (11, 12): GradeLevelConfig(
            grade_range=(11, 12),
            max_sentence_length=30,
            max_word_length=18,
            max_syllables=7,
            min_flesch_score=50.0,
            max_flesch_kincaid_grade=12.0,
            vocabulary_complexity="very_advanced",
            recommended_examples=["calculus", "physics", "advanced literature"],
        ),
    }

    # Simple vocabulary by grade level (examples)
    VOCABULARY_LISTS = {
        "very_simple": {
            "math": ["add", "take", "count", "more", "less"],
            "science": ["see", "hear", "touch", "smell", "taste"],
            "general": ["big", "small", "good", "bad", "happy", "sad"],
        },
        "simple": {
            "math": ["plus", "minus", "multiply", "divide", "equal"],
            "science": ["plant", "animal", "water", "air", "soil"],
            "general": ["beautiful", "interesting", "important", "different"],
        },
        "moderate": {
            "math": ["fraction", "decimal", "percentage", "equation"],
            "science": ["photosynthesis", "respiration", "evaporation"],
            "general": ["comprehend", "analyze", "evaluate", "construct"],
        },
        "advanced": {
            "math": ["quadratic", "polynomial", "trigonometry", "logarithm"],
            "science": ["mitochondria", "chromosome", "ecosystem", "catalyst"],
            "general": ["hypothesize", "synthesize", "critique", "formulate"],
        },
    }

    def __init__(self, db: Session):
        self.db = db

    def _get_grade_config(self, grade_level: int) -> GradeLevelConfig:
        """Get configuration for grade level."""
        for grade_range, config in self.GRADE_CONFIGS.items():
            if grade_range[0] <= grade_level <= grade_range[1]:
                return config
        # Default to moderate if grade not found
        return self.GRADE_CONFIGS[(6, 8)]

    def analyze_readability(self, text: str, grade_level: int) -> dict[str, Any]:
        """
        Analyze text readability for grade level.

        Args:
            text: Content text
            grade_level: Target grade level (1-12)

        Returns:
            Readability analysis with scores and issues
        """
        logger.info(f"Analyzing readability for grade {grade_level}")

        config = self._get_grade_config(grade_level)

        # Calculate basic metrics
        sentences = self._split_into_sentences(text)
        words = text.split()

        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(self._count_syllables(word) for word in words)

        # Avoid division by zero
        if total_sentences == 0 or total_words == 0:
            return {"error": "Invalid text for analysis"}

        # Calculate readability scores
        avg_sentence_length = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words

        # Flesch Reading Ease (higher is easier)
        flesch_score = (
            206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        )
        flesch_score = max(0, min(100, flesch_score))

        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        fk_grade = max(0, fk_grade)

        # Identify issues
        issues = []

        if avg_sentence_length > config.max_sentence_length:
            issues.append(
                {
                    "type": "long_sentences",
                    "severity": "high",
                    "details": f"Average sentence length ({avg_sentence_length:.1f}) exceeds "
                    f"recommended ({config.max_sentence_length})",
                    "recommendation": "Break long sentences into shorter ones",
                }
            )

        if flesch_score < config.min_flesch_score:
            issues.append(
                {
                    "type": "low_readability",
                    "severity": "high",
                    "details": f"Flesch score ({flesch_score:.1f}) below recommended "
                    f"({config.min_flesch_score})",
                    "recommendation": "Simplify vocabulary and sentence structure",
                }
            )

        if fk_grade > config.max_flesch_kincaid_grade:
            issues.append(
                {
                    "type": "high_grade_level",
                    "severity": "medium",
                    "details": f"Flesch-Kincaid grade ({fk_grade:.1f}) exceeds target "
                    f"({config.max_flesch_kincaid_grade})",
                    "recommendation": "Use simpler language appropriate for grade level",
                }
            )

        # Check for complex words
        complex_words = self._find_complex_words(words, config)
        if complex_words:
            issues.append(
                {
                    "type": "complex_vocabulary",
                    "severity": "medium",
                    "details": f"Found {len(complex_words)} complex words",
                    "examples": complex_words[:10],
                    "recommendation": "Replace complex words with simpler alternatives",
                }
            )

        # Determine readability level
        if flesch_score >= 90:
            readability_level = ReadabilityLevel.VERY_EASY
        elif flesch_score >= 80:
            readability_level = ReadabilityLevel.EASY
        elif flesch_score >= 70:
            readability_level = ReadabilityLevel.MODERATE
        elif flesch_score >= 60:
            readability_level = ReadabilityLevel.DIFFICULT
        else:
            readability_level = ReadabilityLevel.VERY_DIFFICULT

        return {
            "grade_level": grade_level,
            "metrics": {
                "total_sentences": total_sentences,
                "total_words": total_words,
                "total_syllables": total_syllables,
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_syllables_per_word": round(avg_syllables_per_word, 2),
                "flesch_reading_ease": round(flesch_score, 2),
                "flesch_kincaid_grade": round(fk_grade, 2),
            },
            "readability_level": readability_level,
            "is_appropriate": flesch_score >= config.min_flesch_score
            and fk_grade <= config.max_flesch_kincaid_grade,
            "issues": issues,
            "recommendations": self._generate_simplification_recommendations(
                text, config
            ),
        }

    def adapt_content_complexity(
        self, text: str, current_grade: int, target_grade: int
    ) -> dict[str, Any]:
        """
        Adapt content from one grade level to another.

        Args:
            text: Original content
            current_grade: Current grade level
            target_grade: Target grade level

        Returns:
            Adaptation recommendations
        """
        logger.info(f"Adapting content from grade {current_grade} to {target_grade}")

        current_config = self._get_grade_config(current_grade)
        target_config = self._get_grade_config(target_grade)

        adaptations = []

        if target_grade < current_grade:
            # Simplification needed
            adaptations.append(
                {
                    "type": "vocabulary_simplification",
                    "action": "Replace complex words with simpler alternatives",
                    "priority": "high",
                }
            )
            adaptations.append(
                {
                    "type": "sentence_shortening",
                    "action": f"Reduce average sentence length from {current_config.max_sentence_length} "
                    f"to {target_config.max_sentence_length} words",
                    "priority": "high",
                }
            )
            adaptations.append(
                {
                    "type": "concept_breakdown",
                    "action": "Break complex concepts into smaller, sequential steps",
                    "priority": "medium",
                }
            )
            adaptations.append(
                {
                    "type": "concrete_examples",
                    "action": "Add more concrete, relatable examples",
                    "examples": target_config.recommended_examples,
                    "priority": "medium",
                }
            )
        else:
            # Enhancement needed
            adaptations.append(
                {
                    "type": "vocabulary_enrichment",
                    "action": "Introduce more advanced terminology with explanations",
                    "priority": "medium",
                }
            )
            adaptations.append(
                {
                    "type": "concept_depth",
                    "action": "Add deeper explanations and connections",
                    "priority": "high",
                }
            )
            adaptations.append(
                {
                    "type": "critical_thinking",
                    "action": "Include analysis and evaluation questions",
                    "priority": "medium",
                }
            )

        return {
            "current_grade": current_grade,
            "target_grade": target_grade,
            "direction": "simplify" if target_grade < current_grade else "enhance",
            "adaptations": adaptations,
            "estimated_effort": "high"
            if abs(target_grade - current_grade) > 3
            else "moderate",
        }

    def suggest_vocabulary_replacements(
        self, text: str, grade_level: int
    ) -> list[dict[str, str]]:
        """
        Suggest simpler vocabulary replacements.

        Args:
            text: Content text
            grade_level: Target grade level

        Returns:
            List of word replacement suggestions
        """
        config = self._get_grade_config(grade_level)
        words = text.split()

        # Find complex words
        complex_words = self._find_complex_words(words, config)

        # Generate replacements
        replacements = []
        for word in complex_words[:20]:  # Limit to top 20
            simpler = self._get_simpler_alternative(word, config)
            if simpler and simpler != word:
                replacements.append(
                    {
                        "original": word,
                        "replacement": simpler,
                        "reason": "Simpler alternative for grade level",
                    }
                )

        return replacements

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be enhanced with NLTK)
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (simple heuristic).

        This is a simplified version. For production, use pyphen or similar.
        """
        word = word.lower().strip(".,!?;:")
        if not word:
            return 0

        # Count vowel groups
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e"):
            syllable_count -= 1

        # Ensure at least 1 syllable
        return max(1, syllable_count)

    def _find_complex_words(
        self, words: list[str], config: GradeLevelConfig
    ) -> list[str]:
        """Find words that are too complex for grade level."""
        complex_words = []

        for word in words:
            clean_word = re.sub(r"[^\w]", "", word.lower())
            if not clean_word:
                continue

            # Check word length
            if len(clean_word) > config.max_word_length:
                complex_words.append(word)
                continue

            # Check syllables
            syllables = self._count_syllables(clean_word)
            if syllables > config.max_syllables:
                complex_words.append(word)

        return list(set(complex_words))

    def _get_simpler_alternative(
        self, word: str, config: GradeLevelConfig
    ) -> str | None:
        """Get simpler alternative for a word."""
        # Simple word replacement map (can be expanded)
        simplification_map = {
            "utilize": "use",
            "purchase": "buy",
            "obtain": "get",
            "demonstrate": "show",
            "commence": "start",
            "terminate": "end",
            "facilitate": "help",
            "approximately": "about",
            "numerous": "many",
            "sufficient": "enough",
            "acquire": "get",
            "construct": "build",
            "examine": "look at",
            "comprehend": "understand",
            "illustrate": "show",
            "magnificent": "great",
            "difficult": "hard",
            "continue": "go on",
            "however": "but",
            "therefore": "so",
        }

        word_lower = word.lower()
        return simplification_map.get(word_lower, word)

    def _generate_simplification_recommendations(
        self, text: str, config: GradeLevelConfig
    ) -> list[str]:
        """Generate specific simplification recommendations."""
        recommendations = []

        sentences = self._split_into_sentences(text)

        # Check for long sentences
        long_sentences = [
            s for s in sentences if len(s.split()) > config.max_sentence_length
        ]
        if long_sentences:
            recommendations.append(
                f"Break {len(long_sentences)} long sentences into shorter ones"
            )

        # Check for passive voice (simple detection)
        passive_indicators = ["is", "are", "was", "were", "been", "being"]
        if any(indicator in text.lower() for indicator in passive_indicators):
            recommendations.append(
                "Use active voice instead of passive voice where possible"
            )

        # Check for complex connectors
        complex_connectors = [
            "nevertheless",
            "furthermore",
            "consequently",
            "meanwhile",
        ]
        if any(connector in text.lower() for connector in complex_connectors):
            recommendations.append(
                "Replace complex connecting words with simpler alternatives"
            )

        # Suggest examples
        if len(sentences) > 5 and "example" not in text.lower():
            recommendations.append(
                f"Add concrete examples: {', '.join(config.recommended_examples[:3])}"
            )

        return recommendations

    def validate_grade_appropriateness(
        self, text: str, grade_level: int, subject: str
    ) -> dict[str, Any]:
        """
        Comprehensive validation of grade appropriateness.

        Args:
            text: Content text
            grade_level: Target grade (1-12)
            subject: Subject area

        Returns:
            Comprehensive validation results
        """
        logger.info(f"Validating grade appropriateness: Grade {grade_level}, {subject}")

        # Readability analysis
        readability = self.analyze_readability(text, grade_level)

        # Vocabulary check
        vocabulary_replacements = self.suggest_vocabulary_replacements(
            text, grade_level
        )

        # Calculate appropriateness score
        appropriateness_score = 1.0

        # Deduct for readability issues
        if not readability.get("is_appropriate", False):
            appropriateness_score -= 0.3

        # Deduct for complex vocabulary
        if len(vocabulary_replacements) > 10:
            appropriateness_score -= 0.2
        elif len(vocabulary_replacements) > 5:
            appropriateness_score -= 0.1

        appropriateness_score = max(0.0, appropriateness_score)

        return {
            "grade_level": grade_level,
            "subject": subject,
            "appropriateness_score": round(appropriateness_score, 2),
            "passed": appropriateness_score >= 0.7,
            "readability": readability,
            "vocabulary_suggestions": vocabulary_replacements[:10],
            "overall_recommendations": self._get_priority_recommendations(
                readability, vocabulary_replacements
            ),
        }

    def _get_priority_recommendations(
        self, readability: dict[str, Any], vocabulary_replacements: list[dict[str, str]]
    ) -> list[str]:
        """Get prioritized recommendations."""
        recommendations = []

        # High priority: Readability issues
        high_severity_issues = [
            issue
            for issue in readability.get("issues", [])
            if issue.get("severity") == "high"
        ]
        for issue in high_severity_issues[:3]:
            recommendations.append(f"HIGH: {issue.get('recommendation', '')}")

        # Medium priority: Vocabulary
        if len(vocabulary_replacements) > 5:
            recommendations.append(
                f"MEDIUM: Replace {len(vocabulary_replacements)} complex words"
            )

        # Add general recommendations
        for rec in readability.get("recommendations", [])[:2]:
            if rec not in recommendations:
                recommendations.append(f"GENERAL: {rec}")

        return recommendations


# Pipeline integration
async def adapt_grade_level(
    db: Session, content: ProcessedContent, text: str
) -> dict[str, Any]:
    """
    Apply grade-level adaptation in content pipeline.

    Args:
        db: Database session
        content: ProcessedContent object
        text: Content text

    Returns:
        Grade-level validation and adaptation results
    """
    service = GradeLevelAdaptationService(db)

    # Validate appropriateness
    validation = service.validate_grade_appropriateness(
        text=text, grade_level=content.grade_level, subject=content.subject
    )

    return validation
