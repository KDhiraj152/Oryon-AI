"""
Unified Cultural Context Service.

Consolidates cultural adaptation functionality from:
- backend/services/cultural_context.py
- backend/services/cultural_context_service.py

Provides:
- Region-specific content adaptation
- Cultural sensitivity checking
- Festival and local example injection
- Culturally appropriate terminology
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ==================== Enums ====================


class Region(str, Enum):
    """Indian geographical regions."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    CENTRAL = "central"
    GENERAL = "general"


class Subject(str, Enum):
    """Educational subjects."""

    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    SOCIAL_STUDIES = "social_studies"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    ENGLISH = "english"
    HINDI = "hindi"
    GENERAL = "general"


class SensitivityLevel(str, Enum):
    """Content sensitivity levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ==================== Data Classes ====================


@dataclass
class CulturalExample:
    """A culturally relevant example for content."""

    topic: str
    example: str
    region: Region
    subject: Subject = Subject.GENERAL
    grade_range: tuple[int, int] = (1, 12)
    keywords: list[str] = field(default_factory=list)


@dataclass
class RegionalContext:
    """Regional context data."""

    languages: list[str]
    festivals: list[str]
    food_examples: list[str]
    cultural_references: list[str]
    notable_figures: list[str]
    traditional_items: list[str]


@dataclass
class AdaptedContent:
    """Content with cultural adaptations applied."""

    original_text: str
    adapted_text: str
    region: Region
    examples_added: list[CulturalExample] = field(default_factory=list)
    replacements_made: dict[str, str] = field(default_factory=dict)
    sensitivity_flags: list[str] = field(default_factory=list)


# ==================== Regional Data ====================

REGIONAL_CONTEXTS: dict[Region, RegionalContext] = {
    Region.NORTH: RegionalContext(
        languages=["hi", "pa", "ur", "raj"],
        festivals=["diwali", "holi", "lohri", "baisakhi", "teej", "karwa chauth"],
        food_examples=["roti", "paratha", "dal makhani", "paneer", "lassi", "kulfi"],
        cultural_references=["punjab", "delhi", "rajasthan", "haryana", "himachal"],
        notable_figures=["bhagat singh", "rani laxmibai", "prithviraj chauhan"],
        traditional_items=["phulkari", "juttis", "pagri", "dhol"],
    ),
    Region.SOUTH: RegionalContext(
        languages=["ta", "te", "kn", "ml"],
        festivals=["pongal", "onam", "ugadi", "vishu", "sankranti"],
        food_examples=["idli", "dosa", "sambar", "rasam", "filter coffee", "payasam"],
        cultural_references=["tamil nadu", "kerala", "karnataka", "andhra pradesh"],
        notable_figures=["tipu sultan", "raja raja chola", "cv raman"],
        traditional_items=["silk sarees", "kathakali", "bharatanatyam", "veena"],
    ),
    Region.EAST: RegionalContext(
        languages=["bn", "or", "as"],
        festivals=["durga puja", "rath yatra", "bihu", "chhath puja", "saraswati puja"],
        food_examples=[
            "rosogolla",
            "mishti doi",
            "machher jhol",
            "pitha",
            "litti chokha",
        ],
        cultural_references=["kolkata", "odisha", "bihar", "jharkhand", "assam"],
        notable_figures=[
            "rabindranath tagore",
            "netaji subhas chandra bose",
            "swami vivekananda",
        ],
        traditional_items=["dhokra art", "pattachitra", "tant saree", "gamcha"],
    ),
    Region.WEST: RegionalContext(
        languages=["gu", "mr", "kok"],
        festivals=["ganesh chaturthi", "navratri", "gudi padwa", "janmashtami"],
        food_examples=[
            "dhokla",
            "thepla",
            "vada pav",
            "puran poli",
            "shrikhand",
            "modak",
        ],
        cultural_references=["mumbai", "pune", "ahmedabad", "goa", "gujarat"],
        notable_figures=["shivaji maharaj", "sardar patel", "mahatma gandhi"],
        traditional_items=["bandhani", "warli art", "kolhapuri chappals", "paithani"],
    ),
    Region.NORTHEAST: RegionalContext(
        languages=["as", "mni", "nep"],
        festivals=["bihu", "hornbill", "wangala", "me-dam-me-phi"],
        food_examples=["bamboo shoot", "pitha", "jadoh", "thukpa", "momos"],
        cultural_references=["guwahati", "shillong", "imphal", "kohima", "arunachal"],
        notable_figures=["bhupen hazarika", "mary kom", "lovlina borgohain"],
        traditional_items=[
            "muga silk",
            "naga shawls",
            "bamboo crafts",
            "cane furniture",
        ],
    ),
    Region.CENTRAL: RegionalContext(
        languages=["hi", "cg", "bundeli"],
        festivals=["diwali", "holi", "teej", "gangaur", "makar sankranti"],
        food_examples=["bhutte ka kees", "dal bafla", "poha", "jalebi", "malpua"],
        cultural_references=["bhopal", "indore", "raipur", "jabalpur", "gwalior"],
        notable_figures=["rani durgavati", "tansen", "chandrashekhar azad"],
        traditional_items=["chanderi silk", "gond art", "dhurrie", "bell metal craft"],
    ),
    Region.GENERAL: RegionalContext(
        languages=["hi", "en"],
        festivals=["independence day", "republic day", "gandhi jayanti"],
        food_examples=["rice", "roti", "dal", "sabzi", "chai"],
        cultural_references=["india", "bharat"],
        notable_figures=["mahatma gandhi", "jawaharlal nehru", "apj abdul kalam"],
        traditional_items=["khadi", "rangoli", "diyas"],
    ),
}


# ==================== Sensitive Terms ====================

# Terms that should be handled with care
SENSITIVE_TERMS: set[str] = {
    "caste",
    "untouchable",
    "backward",
    "reservation",
    "religious",
    "communal",
    "partition",
    "riot",
    "terrorism",
    "extremism",
    "separatist",
}

# Terms to replace with more neutral alternatives
TERM_REPLACEMENTS: dict[str, str] = {
    "backward class": "scheduled categories",
    "lower caste": "scheduled community",
    "primitive tribe": "particularly vulnerable tribal group",
    "handicapped": "differently-abled",
    "mentally retarded": "intellectually challenged",
}


# ==================== Subject Examples ====================

SUBJECT_EXAMPLES: dict[Subject, dict[Region, list[CulturalExample]]] = {
    Subject.MATHEMATICS: {
        Region.NORTH: [
            CulturalExample(
                topic="fractions",
                example="If a lassi shop sells 3/4 of its lassi in the morning...",
                region=Region.NORTH,
                subject=Subject.MATHEMATICS,
                keywords=["fraction", "division", "ratio"],
            ),
            CulturalExample(
                topic="geometry",
                example="The Jantar Mantar in Jaipur uses geometric shapes...",
                region=Region.NORTH,
                subject=Subject.MATHEMATICS,
                keywords=["geometry", "angles", "measurement"],
            ),
        ],
        Region.SOUTH: [
            CulturalExample(
                topic="fractions",
                example="A dosa batter recipe requires 3/4 rice and 1/4 urad dal...",
                region=Region.SOUTH,
                subject=Subject.MATHEMATICS,
                keywords=["fraction", "ratio", "proportion"],
            ),
            CulturalExample(
                topic="patterns",
                example="Kolam designs use mathematical patterns and symmetry...",
                region=Region.SOUTH,
                subject=Subject.MATHEMATICS,
                keywords=["patterns", "symmetry", "geometry"],
            ),
        ],
    },
    Subject.SCIENCE: {
        Region.GENERAL: [
            CulturalExample(
                topic="agriculture",
                example="Indian farmers use crop rotation, similar to how nutrients cycle...",
                region=Region.GENERAL,
                subject=Subject.SCIENCE,
                keywords=["agriculture", "farming", "soil"],
            ),
        ],
    },
}


# ==================== Service Class ====================


class UnifiedCulturalContextService:
    """
    Unified service for cultural context adaptation.

    Thread-safe, cacheable, and optimized for performance.
    """

    def __init__(self):
        self._contexts = REGIONAL_CONTEXTS
        self._examples = SUBJECT_EXAMPLES
        self._sensitive_terms = SENSITIVE_TERMS
        self._replacements = TERM_REPLACEMENTS
        # Instance cache instead of lru_cache to avoid potential memory leak
        self._region_cache: dict[Region, RegionalContext] = {}

    def get_regional_context(self, region: Region) -> RegionalContext:
        """Get context for a specific region (cached in instance)."""
        if region in self._region_cache:
            return self._region_cache[region]
        result = self._contexts.get(region, self._contexts[Region.GENERAL])
        self._region_cache[region] = result
        return result

    def detect_region_from_language(self, language_code: str) -> Region:
        """Detect likely region from language code."""
        lang_to_region = {
            "hi": Region.NORTH,
            "pa": Region.NORTH,
            "ta": Region.SOUTH,
            "te": Region.SOUTH,
            "kn": Region.SOUTH,
            "ml": Region.SOUTH,
            "bn": Region.EAST,
            "or": Region.EAST,
            "gu": Region.WEST,
            "mr": Region.WEST,
            "as": Region.NORTHEAST,
        }
        return lang_to_region.get(language_code[:2].lower(), Region.GENERAL)

    def adapt_content(
        self,
        text: str,
        region: Region = Region.GENERAL,
        subject: Subject = Subject.GENERAL,
        complexity_level: int | None = None,
        check_sensitivity: bool = True,
    ) -> AdaptedContent:
        """
        Adapt content with cultural context.

        Args:
            text: Original text to adapt
            region: Target region
            subject: Subject area
            complexity_level: Optional target complexity level (None for unconstrained)
            check_sensitivity: Whether to flag sensitive content

        Returns:
            AdaptedContent with adaptations applied
        """
        adapted_text = text
        replacements = {}
        sensitivity_flags = []

        # Apply term replacements
        for old_term, new_term in self._replacements.items():
            if old_term.lower() in adapted_text.lower():
                pattern = re.compile(re.escape(old_term), re.IGNORECASE)
                adapted_text = pattern.sub(new_term, adapted_text)
                replacements[old_term] = new_term

        # Check for sensitive terms
        if check_sensitivity:
            for term in self._sensitive_terms:
                if term.lower() in adapted_text.lower():
                    sensitivity_flags.append(f"Contains sensitive term: {term}")

        # Add relevant examples if space permits
        relevant_examples = self._get_relevant_examples(subject, region, complexity_level)

        return AdaptedContent(
            original_text=text,
            adapted_text=adapted_text,
            region=region,
            examples_added=relevant_examples,
            replacements_made=replacements,
            sensitivity_flags=sensitivity_flags,
        )

    def _get_relevant_examples(
        self, subject: Subject, region: Region, complexity_level: int | None = None
    ) -> list[CulturalExample]:
        """Get relevant cultural examples for subject and region."""
        examples = []

        # Check subject-specific examples
        if subject in self._examples:
            subject_examples = self._examples[subject]

            # Try region-specific first
            if region in subject_examples:
                examples.extend(subject_examples[region])

            # Fall back to general
            if Region.GENERAL in subject_examples:
                examples.extend(subject_examples[Region.GENERAL])

        # Only filter by complexity level if specified
        if complexity_level is not None:
            return [
                ex
                for ex in examples
                if ex.grade_range[0] <= complexity_level <= ex.grade_range[1]
            ]

        # Return all examples when complexity_level is unconstrained
        return examples

    def inject_local_references(
        self, text: str, region: Region, max_injections: int = 2
    ) -> str:
        """
        Inject local references to make content more relatable.

        Args:
            text: Original text
            region: Target region
            max_injections: Maximum number of references to inject

        Returns:
            Text with local references
        """
        context = self.get_regional_context(region)

        # Simple substitutions for common terms
        substitutions = {
            "a popular festival": f"festivals like {context.festivals[0]}",
            "local food": f"dishes like {context.food_examples[0]}",
            "this region": f"regions like {context.cultural_references[0]}",
        }

        result = text
        count = 0

        for old, new in substitutions.items():
            if count >= max_injections:
                break
            if old in result.lower():
                result = re.sub(
                    re.escape(old), new, result, count=1, flags=re.IGNORECASE
                )
                count += 1

        return result

    def get_festival_greeting(self, region: Region) -> str | None:
        """Get an appropriate festival greeting for the region."""
        context = self.get_regional_context(region)
        if context.festivals:
            festival = context.festivals[0]
            return f"Wishing you a happy {festival}!"
        return None

    def check_cultural_appropriateness(
        self, text: str, region: Region
    ) -> dict[str, Any]:
        """
        Check if content is culturally appropriate.

        Returns:
            Dict with 'is_appropriate', 'issues', 'suggestions'
        """
        issues = []
        suggestions = []

        # Check for sensitive terms
        for term in self._sensitive_terms:
            if term.lower() in text.lower():
                issues.append(f"Contains potentially sensitive term: '{term}'")
                suggestions.append(f"Consider providing context for '{term}'")

        # Check for terms needing replacement
        for old_term in self._replacements:
            if old_term.lower() in text.lower():
                issues.append(f"Contains outdated term: '{old_term}'")
                suggestions.append(
                    f"Replace '{old_term}' with '{self._replacements[old_term]}'"
                )

        return {
            "is_appropriate": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "region": region.value,
        }


# ==================== Singleton Instance ====================

_service_instance: UnifiedCulturalContextService | None = None


def get_cultural_context_service() -> UnifiedCulturalContextService:
    """Get singleton instance of cultural context service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = UnifiedCulturalContextService()
    return _service_instance


# ==================== Exports ====================

__all__ = [
    "REGIONAL_CONTEXTS",
    "AdaptedContent",
    "CulturalExample",
    "Region",
    "RegionalContext",
    "SensitivityLevel",
    "Subject",
    "UnifiedCulturalContextService",
    "get_cultural_context_service",
]
