"""
Response Formatter - Consistent, Deterministic Output
=====================================================

Implements ChatGPT/Gemini-quality formatting with:
- Headings, numbered steps, code fences
- Concise summaries with explicit metadata
- Intent detection and source attribution
- Token usage tracking
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum, StrEnum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

class OutputFormat(StrEnum):
    """Supported output formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    PLAIN_TEXT = "plain"
    HTML = "html"

class Intent(StrEnum):
    """Detected user intents."""

    QUESTION = "question"
    CODE_REQUEST = "code_request"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    TRANSLATION = "translation"
    SIMPLIFICATION = "simplification"
    TASK = "task"
    SMALL_TALK = "small_talk"
    UNKNOWN = "unknown"

@dataclass
class SourceReference:
    """Reference to a source document or code."""

    source_id: str
    source_type: str  # "document", "code", "web", "database"
    title: str | None = None
    location: str | None = None  # file:line or doc:page
    confidence: float = 1.0
    quote: str | None = None
    is_inferred: bool = False  # True if inferred, False if quoted

@dataclass
class ResponseMetadata:
    """Metadata for formatted response."""

    intent: Intent
    tokens_used: int
    tokens_prompt: int
    tokens_completion: int
    latency_ms: float
    model_id: str
    sources: list[SourceReference] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "tokens": {
                "total": self.tokens_used,
                "prompt": self.tokens_prompt,
                "completion": self.tokens_completion,
            },
            "latency_ms": self.latency_ms,
            "model": self.model_id,
            "sources": [
                {
                    "id": s.source_id,
                    "type": s.source_type,
                    "title": s.title,
                    "location": s.location,
                    "confidence": s.confidence,
                    "inferred": s.is_inferred,
                }
                for s in self.sources
            ],
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }

@dataclass
class FormattedResponse:
    """A fully formatted AI response."""

    content: str
    summary: str
    metadata: ResponseMetadata
    format: OutputFormat = OutputFormat.MARKDOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "summary": self.summary,
            "metadata": self.metadata.to_dict(),
            "format": self.format.value,
        }

    def to_markdown(self) -> str:
        """Convert to markdown with metadata footer."""
        parts = [self.content]

        if self.metadata.sources:
            parts.append("\n---\n**Sources:**")
            for i, src in enumerate(self.metadata.sources, 1):
                marker = "📝" if src.is_inferred else "📖"
                parts.append(
                    f"{i}. {marker} {src.title or src.source_id} ({src.location or 'N/A'})"
                )

        return "\n".join(parts)

    def to_json(self) -> str:
        """Convert to JSON with full provenance."""
        return json.dumps(self.to_dict(), indent=2)

    def to_plain(self) -> str:
        """Convert to plain text."""
        # Strip markdown formatting
        text = self.content
        # Remove code fences
        text = re.sub(r"```\w*\n", "", text)
        text = text.replace("```", "")
        # Remove headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        return text

class ResponseFormatter:
    """
    Formats AI responses to ChatGPT/Gemini quality standards.

    Features:
    - Automatic intent detection
    - Consistent structure (summary, details, actions, sources)
    - Code fence wrapping for code snippets
    - Source citation formatting
    """

    # Intent detection patterns
    INTENT_PATTERNS: ClassVar[dict] = {
        Intent.CODE_REQUEST: [
            r"\b(write|create|implement|code|function|class|script)\b",
            r"\b(fix|debug|refactor)\s+(the|this|my)?\s*(code|bug|error)\b",
        ],
        Intent.EXPLANATION: [
            r"\b(explain|what\s+is|what\s+are|how\s+does|why\s+does)\b",
            r"\b(tell\s+me\s+about|describe)\b",
        ],
        Intent.COMPARISON: [
            r"\b(compare|vs|versus|difference|between)\b",
            r"\b(better|worse|faster|slower)\s+(than|vs)\b",
        ],
        Intent.TRANSLATION: [
            r"\b(translate|in\s+hindi|in\s+tamil|in\s+telugu)\b",
        ],
        Intent.SIMPLIFICATION: [
            r"\b(simplify|simpler|easier|dumb\s+down)\b",
            r"\b(for\s+(kids|children|beginners|users))\b",
        ],
        Intent.TASK: [
            r"\b(do|make|build|generate|run|execute)\b",
        ],
        Intent.SMALL_TALK: [
            r"^(hi|hello|hey|thanks|thank you|bye|goodbye)\b",
            r"\b(how\s+are\s+you|what\'s\s+up)\b",
        ],
        Intent.QUESTION: [
            r"\?$",
            r"^(who|what|when|where|why|how|which|is|are|can|could|would|should)\b",
        ],
    }

    # Pattern for simple arithmetic expressions
    ARITHMETIC_PATTERN = re.compile(r"^[\d\s\+\-\*\/\^\(\)\.]+[\?]?$")

    def __init__(self, default_format: OutputFormat = OutputFormat.MARKDOWN):
        self.default_format = default_format
        self._compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }

    def detect_intent(self, query: str) -> Intent:
        """Detect the user's intent from their query."""
        query = query.strip()

        # Fast path: Check for pure arithmetic expressions first
        # e.g., "2+2", "10 * 5", "100/4"
        if self.ARITHMETIC_PATTERN.match(query.rstrip("?")):
            return Intent.QUESTION  # Treat as a question that needs answering

        # Check for calculation keywords with numbers
        query_lower = query.lower()
        calc_keywords = [
            "calculate",
            "compute",
            "solve",
            "what is",
            "what's",
            "evaluate",
        ]
        if any(kw in query_lower for kw in calc_keywords) and re.search(r"\d", query):
            return Intent.QUESTION

        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return intent

        return Intent.UNKNOWN

    def format_response(
        self,
        content: str,
        query: str,
        sources: list[SourceReference] | None = None,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        latency_ms: float = 0,
        model_id: str = "unknown",
        confidence: float = 1.0,
        output_format: OutputFormat | None = None,
    ) -> FormattedResponse:
        """
        Format a raw response into production-quality output.

        Args:
            content: Raw response content
            query: Original user query
            sources: Source references for citations
            tokens_prompt: Tokens used in prompt
            tokens_completion: Tokens in completion
            latency_ms: Response latency in milliseconds
            model_id: ID of model used
            confidence: Confidence score (0-1)
            output_format: Desired output format

        Returns:
            FormattedResponse with all metadata
        """
        intent = self.detect_intent(query)
        format_ = output_format or self.default_format

        # Generate summary (first sentence or line)
        summary = self._extract_summary(content)

        # Enhance content based on intent
        enhanced_content = self._enhance_content(content, intent)

        # Add ucontent_domainainty markers if low confidence
        if confidence < 0.7:
            enhanced_content = self._add_ucontent_domainainty_markers(
                enhanced_content, confidence
            )

        metadata = ResponseMetadata(
            intent=intent,
            tokens_used=tokens_prompt + tokens_completion,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            model_id=model_id,
            sources=sources or [],
            confidence=confidence,
        )

        return FormattedResponse(
            content=enhanced_content,
            summary=summary,
            metadata=metadata,
            format=format_,
        )

    def _extract_summary(self, content: str) -> str:
        """Extract a 1-2 line summary from content."""
        lines = content.strip().split("\n")

        # Skip headers
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("```"):
                # Take first sentence
                sentences = re.split(r"[.!?]", line)
                if sentences:
                    summary = sentences[0].strip()
                    if len(summary) > 10:
                        return summary + ("." if not summary.endswith(".") else "")

        # Fallback: first 100 chars
        return content[:100].strip() + ("..." if len(content) > 100 else "")

    def _enhance_content(self, content: str, intent: Intent) -> str:
        """Enhance content structure based on intent."""
        # Ensure code blocks are properly fenced
        content = self._ensure_code_fences(content)

        # Add structure for explanation intents
        if (intent == Intent.EXPLANATION and not content.startswith("#")
                and not any(marker in content for marker in ["##", "**", "1.", "- "])):
            # Don't modify if already structured
            # Simple enhancement: bold key terms
            pass

        return content

    def _ensure_code_fences(self, content: str) -> str:
        """Ensure code blocks have proper language hints."""
        # Pattern for code fences without language
        pattern = r"```\n([^`]+)```"

        def add_language(match):
            code = match.group(1)
            lang = self._detect_code_language(code)
            return f"```{lang}\n{code}```"

        return re.sub(pattern, add_language, content)

    def _detect_code_language(self, code: str) -> str:
        """Detect programming language from code snippet."""
        code_lower = code.lower()

        if "def " in code or "import " in code or "class " in code_lower:
            return "python"
        if "function" in code or "const " in code or "=>" in code:
            return "javascript"
        if "<html" in code_lower or "</div>" in code_lower:
            return "html"
        if "{" in code and ":" in code and ('"' in code or "'" in code):
            return "json"
        if "SELECT " in code.upper() or "INSERT " in code.upper():
            return "sql"

        return ""  # Unknown

    def _add_ucontent_domainainty_markers(self, content: str, confidence: float) -> str:
        """Add markers when response confidence is low."""
        if confidence < 0.5:
            prefix = (
                "⚠️ **Note:** I'm not fully confident about this answer. "
                "Here's what I think, but please verify:\n\n"
            )
            suffix = (
                "\n\n---\n"
                "💡 **How to verify:**\n"
                "- Check official documentation\n"
                "- Test in a safe environment\n"
                f"- Confidence score: {confidence:.0%}"
            )
            return prefix + content + suffix
        elif confidence < 0.7:
            suffix = f"\n\n*Confidence: {confidence:.0%} — consider verifying important details.*"
            return content + suffix

        return content

    def format_code_response(
        self,
        code: str,
        language: str,
        explanation: str = "",
        tests: str | None = None,
        sources: list[SourceReference] | None = None,
    ) -> str:
        """Format a code response with explanation and tests."""
        parts = []

        if explanation:
            parts.append(explanation + "\n")

        parts.append(f"```{language}")
        parts.append(code.strip())
        parts.append("```")

        if tests:
            parts.append("\n**Tests:**\n")
            parts.append(f"```{language}")
            parts.append(tests.strip())
            parts.append("```")

        return "\n".join(parts)

    def format_comparison(
        self,
        items: list[dict[str, Any]],
        criteria: list[str],
    ) -> str:
        """Format a comparison table."""
        if not items or not criteria:
            return ""

        # Build markdown table
        headers = ["Criteria"] + [
            item.get("name", f"Option {i + 1}") for i, item in enumerate(items)
        ]
        separator = "|" + "|".join(["---"] * len(headers)) + "|"
        header_row = "|" + "|".join(headers) + "|"

        rows = [header_row, separator]

        for criterion in criteria:
            row_data = [criterion]
            for item in items:
                value = item.get(criterion, "N/A")
                row_data.append(str(value))
            rows.append("|" + "|".join(row_data) + "|")

        return "\n".join(rows)

    def format_steps(self, steps: list[str], title: str | None = None) -> str:
        """Format numbered steps."""
        parts = []

        if title:
            parts.append(f"## {title}\n")

        for i, step in enumerate(steps, 1):
            parts.append(f"{i}. {step}")

        return "\n".join(parts)

    def format_error_response(
        self,
        error_message: str,
        suggestions: list[str] | None = None,
        context: str | None = None,
    ) -> str:
        """Format an error response with helpful suggestions."""
        parts = [f"❌ **Error:** {error_message}"]

        if context:
            parts.append(f"\n**Context:** {context}")

        if suggestions:
            parts.append("\n**Suggestions:**")
            for sug in suggestions:
                parts.append(f"- {sug}")

        return "\n".join(parts)
