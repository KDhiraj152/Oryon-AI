"""
Prompt Registry - Versioned Prompt Management
==============================================

Implements versioned prompt management with:
- Version tracking for all prompts
- Test requirements per prompt
- Changelog entries
- A/B testing support
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptCategory(str, Enum):
    """Categories of prompts."""

    SYSTEM = "system"
    CHAT = "chat"
    REASONING = "reasoning"
    CODE = "code"
    TRANSLATION = "translation"
    SIMPLIFICATION = "simplification"
    VALIDATION = "validation"
    RAG = "rag"
    SAFETY = "safety"


@dataclass
class PromptTest:
    """Test case for a prompt."""

    name: str
    input_text: str
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    max_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_text": self.input_text,
            "expected_contains": self.expected_contains,
            "expected_not_contains": self.expected_not_contains,
            "max_tokens": self.max_tokens,
        }


@dataclass
class ChangelogEntry:
    """Entry in the prompt changelog."""

    version: str
    date: datetime
    author: str
    changes: list[str]
    breaking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "date": self.date.isoformat(),
            "author": self.author,
            "changes": self.changes,
            "breaking": self.breaking,
        }


@dataclass
class PromptVersion:
    """
    A versioned prompt template.

    Attributes:
        name: Unique identifier for the prompt
        version: Semantic version string (e.g., "1.2.0")
        category: Prompt category
        template: The prompt template with {placeholders}
        description: Human-readable description
        variables: List of required template variables
        tests: Test cases for this prompt
        changelog: Version history
        active: Whether this version is currently active
        metadata: Additional metadata
    """

    name: str
    version: str
    category: PromptCategory
    template: str
    description: str = ""
    variables: list[str] = field(default_factory=list)
    tests: list[PromptTest] = field(default_factory=list)
    changelog: list[ChangelogEntry] = field(default_factory=list)
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        # Auto-detect variables from template
        if not self.variables:
            import re

            self.variables = re.findall(r"\{(\w+)\}", self.template)

    @property
    def hash(self) -> str:
        """Get hash of template content for change detection."""
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            logger.warning(f"Missing variables for prompt {self.name}: {missing}")

        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Template render error: {e}")
            raise ValueError(f"Missing required variable: {e}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "category": self.category.value,
            "template": self.template,
            "description": self.description,
            "variables": self.variables,
            "tests": [t.to_dict() for t in self.tests],
            "changelog": [c.to_dict() for c in self.changelog],
            "active": self.active,
            "hash": self.hash,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptVersion":
        return cls(
            name=data["name"],
            version=data["version"],
            category=PromptCategory(data["category"]),
            template=data["template"],
            description=data.get("description", ""),
            variables=data.get("variables", []),
            tests=[PromptTest(**t) for t in data.get("tests", [])],
            changelog=[
                ChangelogEntry(
                    version=c["version"],
                    date=datetime.fromisoformat(c["date"]),
                    author=c["author"],
                    changes=c["changes"],
                    breaking=c.get("breaking", False),
                )
                for c in data.get("changelog", [])
            ],
            active=data.get("active", True),
            metadata=data.get("metadata", {}),
        )


class PromptRegistry:
    """
    Registry for versioned prompts.

    Features:
    - Version management
    - A/B testing support
    - Prompt validation
    - Export/import
    """

    # Built-in system prompts
    DEFAULT_PROMPTS = {
        "system.base": PromptVersion(
            name="system.base",
            version="1.1.0",
            category=PromptCategory.SYSTEM,
            template=(
                "You are ShikshaSetu, an intelligent AI assistant. "
                "Your goal is to provide helpful, accurate responses.\n\n"
                "Guidelines:\n"
                "1. Be clear, accurate, and helpful\n"
                "2. Use appropriate language for the student's complexity level\n"
                "3. Provide examples when explaining concepts\n"
                "4. Encourage curiosity and critical thinking\n"
                "5. Admit when you don't know something\n\n"
                "FORMATTING:\n"
                "- For math equations use LaTeX: inline with $...$ and block with $$...$$\n"
                "- Examples: $x^2 + y^2 = z^2$, $$\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$\n"
                "- Use markdown for code blocks, lists, and emphasis\n"
                "- Give complete, detailed explanations\n"
            ),
            description="Base system prompt for all interactions",
            tests=[
                PromptTest(
                    name="basic_rendering",
                    input_text="Hello",
                    expected_contains=["ShikshaSetu", "educational"],
                ),
            ],
            changelog=[
                ChangelogEntry(
                    version="1.0.0",
                    date=datetime(2025, 12, 1, tzinfo=UTC),
                    author="system",
                    changes=["Initial version"],
                    breaking=False,
                ),
                ChangelogEntry(
                    version="1.1.0",
                    date=datetime(2025, 12, 2, tzinfo=UTC),
                    author="system",
                    changes=["Added LaTeX math formatting instructions"],
                    breaking=False,
                ),
            ],
        ),
        "system.code": PromptVersion(
            name="system.code",
            version="1.0.0",
            category=PromptCategory.CODE,
            template=(
                "You are a coding assistant. Help users write clean, efficient code.\n\n"
                "When providing code:\n"
                "1. Include complete, runnable examples\n"
                "2. Add comments explaining the logic\n"
                "3. Follow best practices for the language\n"
                "4. Suggest tests when appropriate\n"
                "5. Handle edge cases\n\n"
                "Language: {language}\n"
                "Task: {task}\n"
            ),
            description="System prompt for code generation tasks",
            variables=["language", "task"],
        ),
        "system.explanation": PromptVersion(
            name="system.explanation",
            version="1.0.0",
            category=PromptCategory.REASONING,
            template=(
                "You are explaining a concept clearly.\n\n"
                "Guidelines:\n"
                "1. Use clear, accessible language\n"
                "2. Break down complex ideas into smaller parts\n"
                "3. Use analogies and real-world examples\n"
                "4. Check for understanding with questions\n"
                "5. Build on what the user already knows\n\n"
                "Topic: {topic}\n"
            ),
            description="System prompt for explanations",
            variables=["topic"],
        ),
        "rag.context": PromptVersion(
            name="rag.context",
            version="1.0.0",
            category=PromptCategory.RAG,
            template=(
                "Use the following context to answer the question. "
                "If the answer is not in the context, say so.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            description="RAG context injection template",
            variables=["context", "question"],
        ),
        "rag.citation": PromptVersion(
            name="rag.citation",
            version="1.0.0",
            category=PromptCategory.RAG,
            template=(
                "Answer the question based on the provided sources. "
                "Cite sources using [1], [2], etc.\n\n"
                "Sources:\n{sources}\n\n"
                "Question: {question}\n\n"
                "Provide a well-sourced answer:"
            ),
            description="RAG template with citation requirements",
            variables=["sources", "question"],
        ),
        "validation.factual": PromptVersion(
            name="validation.factual",
            version="1.0.0",
            category=PromptCategory.VALIDATION,
            template=(
                "Evaluate the following response for factual accuracy.\n\n"
                "Response: {response}\n\n"
                "Context (ground truth): {context}\n\n"
                "Rate the factual accuracy from 0-10 and explain any issues:\n"
                "SCORE: "
            ),
            description="Factual accuracy validation prompt",
            variables=["response", "context"],
        ),
    }

    def __init__(self):
        self._prompts: dict[
            str, dict[str, PromptVersion]
        ] = {}  # name -> version -> prompt
        self._active_versions: dict[str, str] = {}  # name -> active version
        self._ab_tests: dict[str, dict[str, float]] = {}  # name -> version -> weight

        # Load default prompts
        for _name, prompt in self.DEFAULT_PROMPTS.items():
            self.register(prompt)

    def register(self, prompt: PromptVersion) -> None:
        """Register a new prompt version."""
        if prompt.name not in self._prompts:
            self._prompts[prompt.name] = {}

        self._prompts[prompt.name][prompt.version] = prompt

        if prompt.active:
            self._active_versions[prompt.name] = prompt.version

        logger.info(f"Registered prompt: {prompt.name} v{prompt.version}")

    def get(
        self,
        name: str,
        version: str | None = None,
    ) -> PromptVersion | None:
        """Get a prompt by name and optionally version."""
        if name not in self._prompts:
            return None

        if version:
            return self._prompts[name].get(version)

        # Get active version
        active_version = self._active_versions.get(name)
        if active_version:
            return self._prompts[name].get(active_version)

        # Return latest version
        versions = sorted(self._prompts[name].keys())
        if versions:
            return self._prompts[name][versions[-1]]

        return None

    def render(
        self,
        name: str,
        version: str | None = None,
        **kwargs,
    ) -> str:
        """Render a prompt template with variables."""
        prompt = self.get(name, version)
        if not prompt:
            raise ValueError(f"Prompt not found: {name}")

        return prompt.render(**kwargs)

    def set_active_version(self, name: str, version: str) -> bool:
        """Set the active version for a prompt."""
        if name not in self._prompts:
            return False

        if version not in self._prompts[name]:
            return False

        # Deactivate current version
        current = self._active_versions.get(name)
        if current and current in self._prompts[name]:
            self._prompts[name][current].active = False

        # Activate new version
        self._prompts[name][version].active = True
        self._active_versions[name] = version

        logger.info(f"Set active version: {name} → v{version}")
        return True

    def setup_ab_test(
        self,
        name: str,
        version_weights: dict[str, float],
    ) -> bool:
        """
        Setup A/B testing for a prompt.

        Args:
            name: Prompt name
            version_weights: Dict of version -> weight (should sum to 1.0)

        Returns:
            True if setup successful
        """
        if name not in self._prompts:
            return False

        # Validate versions exist
        for version in version_weights:
            if version not in self._prompts[name]:
                return False

        self._ab_tests[name] = version_weights
        logger.info(f"Setup A/B test for {name}: {version_weights}")
        return True

    def get_ab_version(self, name: str) -> PromptVersion | None:
        """Get a version based on A/B test weights."""
        if name not in self._ab_tests:
            return self.get(name)

        import random

        weights = self._ab_tests[name]
        versions = list(weights.keys())
        probabilities = list(weights.values())

        chosen = random.choices(versions, weights=probabilities, k=1)[0]
        return self.get(name, chosen)

    def list_prompts(
        self,
        category: PromptCategory | None = None,
    ) -> list[dict[str, Any]]:
        """List all registered prompts."""
        result = []

        for name, versions in self._prompts.items():
            for version, prompt in versions.items():
                if category and prompt.category != category:
                    continue

                result.append(
                    {
                        "name": name,
                        "version": version,
                        "category": prompt.category.value,
                        "active": prompt.active,
                        "hash": prompt.hash,
                    }
                )

        return result

    def export_all(self) -> dict[str, Any]:
        """Export all prompts to JSON-serializable dict."""
        return {
            "prompts": {
                name: {
                    version: prompt.to_dict() for version, prompt in versions.items()
                }
                for name, versions in self._prompts.items()
            },
            "active_versions": self._active_versions.copy(),
            "ab_tests": self._ab_tests.copy(),
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def import_prompts(self, data: dict[str, Any]) -> int:
        """Import prompts from exported data. Returns count imported."""
        count = 0

        for name, versions in data.get("prompts", {}).items():
            for version, prompt_data in versions.items():
                try:
                    prompt = PromptVersion.from_dict(prompt_data)
                    self.register(prompt)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to import {name} v{version}: {e}")

        # Restore active versions
        for name, version in data.get("active_versions", {}).items():
            self.set_active_version(name, version)

        # Restore A/B tests
        for name, weights in data.get("ab_tests", {}).items():
            self.setup_ab_test(name, weights)

        return count

    def run_tests(self, name: str, version: str | None = None) -> dict[str, Any]:
        """Run tests for a prompt."""
        prompt = self.get(name, version)
        if not prompt:
            return {"error": f"Prompt not found: {name}"}

        if not prompt.tests:
            return {"warning": "No tests defined", "passed": True}

        results = []
        all_passed = True

        for test in prompt.tests:
            passed = True
            issues = []

            try:
                # Render with test variables (use input_text as placeholder values)
                variables = {v: f"[{v}]" for v in prompt.variables}
                rendered = prompt.render(**variables)

                # Check expected contains
                for expected in test.expected_contains:
                    if expected not in rendered:
                        passed = False
                        issues.append(f"Missing expected: {expected}")

                # Check expected not contains
                for unexpected in test.expected_not_contains:
                    if unexpected in rendered:
                        passed = False
                        issues.append(f"Contains unexpected: {unexpected}")

            except Exception as e:
                passed = False
                issues.append(f"Render error: {e}")

            results.append(
                {
                    "name": test.name,
                    "passed": passed,
                    "issues": issues,
                }
            )

            if not passed:
                all_passed = False

        return {
            "prompt": name,
            "version": prompt.version,
            "passed": all_passed,
            "tests": results,
        }
