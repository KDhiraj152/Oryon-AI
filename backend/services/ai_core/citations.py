"""
Citation Manager - Source Attribution and Provenance
====================================================

Implements source citations for RAG responses with:
- Document IDs and page/offset references
- Inferred vs quoted distinction
- Confidence scoring
- Citation formatting
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """Types of sources for citations."""

    DOCUMENT = "document"
    CODE = "code"
    WEB = "web"
    DATABASE = "database"
    USER_UPLOAD = "user_upload"
    CURRICULUM = "curriculum"
    MEMORY = "memory"


@dataclass
class Citation:
    """
    A citation reference to a source.

    Attributes:
        source_id: Unique identifier for the source
        source_type: Type of source (document, code, web, etc.)
        title: Human-readable title
        location: Specific location (file:line, page:para, etc.)
        quote: Exact quoted text (if available)
        confidence: Confidence score (0-1)
        is_inferred: True if information is inferred, False if directly quoted
        metadata: Additional source metadata
    """

    source_id: str
    source_type: SourceType
    title: str | None = None
    location: str | None = None
    quote: str | None = None
    confidence: float = 1.0
    is_inferred: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "title": self.title,
            "location": self.location,
            "quote": self.quote,
            "confidence": self.confidence,
            "is_inferred": self.is_inferred,
            "metadata": self.metadata,
            "retrieved_at": self.retrieved_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Citation":
        return cls(
            source_id=data["source_id"],
            source_type=SourceType(data["source_type"]),
            title=data.get("title"),
            location=data.get("location"),
            quote=data.get("quote"),
            confidence=data.get("confidence", 1.0),
            is_inferred=data.get("is_inferred", False),
            metadata=data.get("metadata", {}),
            retrieved_at=datetime.fromisoformat(
                data.get("retrieved_at", datetime.now(UTC).isoformat())
            ),
        )

    def format_markdown(self, index: int) -> str:
        """Format citation as markdown."""
        marker = "📝 (inferred)" if self.is_inferred else "📖 (quoted)"
        title_str = self.title or self.source_id
        location_str = f" — {self.location}" if self.location else ""
        confidence_str = f" [{self.confidence:.0%}]" if self.confidence < 1.0 else ""

        return f"[{index}] {marker} **{title_str}**{location_str}{confidence_str}"

    def format_inline(self, index: int) -> str:
        """Format as inline citation marker."""
        return f"[{index}]"


@dataclass
class CitedResponse:
    """
    A response with citations.

    Attributes:
        content: The main response content
        citations: List of citations
        citation_map: Maps text spans to citation indices
    """

    content: str
    citations: list[Citation] = field(default_factory=list)
    citation_map: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "citations": [c.to_dict() for c in self.citations],
            "citation_map": self.citation_map,
        }

    def format_with_citations(self) -> str:
        """Format response with citation footnotes."""
        if not self.citations:
            return self.content

        parts = [self.content, "\n\n---\n**Sources:**\n"]

        for i, citation in enumerate(self.citations, 1):
            parts.append(citation.format_markdown(i))
            parts.append("\n")

        return "".join(parts)

    def get_top_sources(self, n: int = 3) -> list[Citation]:
        """Get top N most confident sources."""
        sorted_citations = sorted(
            self.citations, key=lambda c: c.confidence, reverse=True
        )
        return sorted_citations[:n]


class CitationManager:
    """
    Manages citations for RAG responses.

    Features:
    - Automatic citation extraction from retrieved chunks
    - Quote vs inference detection
    - Citation deduplication
    - Confidence-based ranking
    """

    # Minimum similarity for direct quote detection
    QUOTE_THRESHOLD = 0.85

    # Maximum number of citations to include
    MAX_CITATIONS = 5

    def __init__(self):
        self._citation_cache: dict[str, list[Citation]] = {}

    def create_citations_from_rag(
        self,
        response_text: str,
        retrieved_chunks: list[dict[str, Any]],
        min_confidence: float = 0.5,
    ) -> list[Citation]:
        """
        Create citations from RAG retrieved chunks.

        Args:
            response_text: The generated response
            retrieved_chunks: Chunks from RAG retrieval
            min_confidence: Minimum confidence to include

        Returns:
            List of Citation objects
        """
        citations = []
        seen_ids = set()

        for chunk in retrieved_chunks:
            # Skip if below confidence threshold
            confidence = chunk.get("score", chunk.get("similarity", 0.7))
            if confidence < min_confidence:
                continue

            # Skip duplicates
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            # Determine if quoted or inferred
            chunk_text = chunk.get("text", chunk.get("content", ""))
            is_inferred = not self._is_directly_quoted(response_text, chunk_text)

            # Extract metadata
            metadata = chunk.get("metadata", {})

            citation = Citation(
                source_id=chunk_id,
                source_type=self._detect_source_type(metadata),
                title=metadata.get("title", metadata.get("filename")),
                location=self._format_location(metadata),
                quote=self._extract_quote(chunk_text),
                confidence=confidence,
                is_inferred=is_inferred,
                metadata=metadata,
            )

            citations.append(citation)

            if len(citations) >= self.MAX_CITATIONS:
                break

        # Sort by confidence
        citations.sort(key=lambda c: c.confidence, reverse=True)

        return citations

    def create_cited_response(
        self,
        response_text: str,
        citations: list[Citation],
    ) -> CitedResponse:
        """
        Create a response with inline citations.

        Args:
            response_text: The generated response
            citations: List of citations to include

        Returns:
            CitedResponse with formatted content
        """
        # Build citation map (which parts of response come from which source)
        citation_map = {}

        for i, citation in enumerate(citations, 1):
            if citation.quote:
                # Find if quote appears in response
                quote_lower = citation.quote.lower()[:50]
                if quote_lower in response_text.lower():
                    citation_map[quote_lower] = i

        return CitedResponse(
            content=response_text,
            citations=citations,
            citation_map=citation_map,
        )

    def add_inline_citations(
        self,
        response_text: str,
        citations: list[Citation],
    ) -> str:
        """
        Add inline citation markers to response text.

        For each sentence that can be attributed to a source,
        adds a citation marker like [1].
        """
        if not citations:
            return response_text

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", response_text)
        cited_sentences = []

        for sentence in sentences:
            # Find best matching citation for this sentence
            best_match = None
            best_score = 0

            for i, citation in enumerate(citations, 1):
                if citation.quote:
                    score = self._compute_similarity(sentence, citation.quote)
                    if score > best_score and score > 0.3:
                        best_score = score
                        best_match = i

            if best_match:
                cited_sentences.append(f"{sentence} [{best_match}]")
            else:
                cited_sentences.append(sentence)

        return " ".join(cited_sentences)

    def _is_directly_quoted(self, response: str, chunk_text: str) -> bool:
        """Check if chunk text is directly quoted in response."""
        # Normalize for comparison
        response_lower = response.lower()

        # Check for substantial overlap
        words = chunk_text.lower().split()
        if len(words) < 5:
            return chunk_text.lower() in response_lower

        # Check for n-gram matches
        for i in range(len(words) - 4):
            phrase = " ".join(words[i : i + 5])
            if phrase in response_lower:
                return True

        return False

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _detect_source_type(self, metadata: dict[str, Any]) -> SourceType:
        """Detect source type from metadata."""
        filename = metadata.get("filename", "").lower()
        source = metadata.get("source", "").lower()

        if any(ext in filename for ext in [".py", ".js", ".ts", ".java", ".cpp"]):
            return SourceType.CODE
        elif any(ext in filename for ext in [".pdf", ".docx", ".txt"]):
            return SourceType.DOCUMENT
        elif "curriculum" in source or "ncert" in source:
            return SourceType.CURRICULUM
        elif metadata.get("user_uploaded"):
            return SourceType.USER_UPLOAD
        elif "http" in source:
            return SourceType.WEB

        return SourceType.DOCUMENT

    def _format_location(self, metadata: dict[str, Any]) -> str | None:
        """Format location string from metadata."""
        parts = []

        if "filename" in metadata:
            parts.append(metadata["filename"])

        if "page" in metadata:
            parts.append(f"page {metadata['page']}")
        elif "line" in metadata:
            parts.append(f"line {metadata['line']}")
        elif "chunk_index" in metadata:
            parts.append(f"section {metadata['chunk_index'] + 1}")

        return " — ".join(parts) if parts else None

    def _extract_quote(self, text: str, max_length: int = 150) -> str:
        """Extract a representative quote from text."""
        text = text.strip()

        if len(text) <= max_length:
            return text

        # Try to find a complete sentence
        sentences = re.split(r"(?<=[.!?])\s+", text)

        if sentences:
            # Take first complete sentence that fits
            for sentence in sentences:
                if len(sentence) <= max_length:
                    return sentence

        # Truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.7:
            truncated = truncated[:last_space]

        return truncated + "..."

    def format_sources_summary(self, citations: list[Citation]) -> str:
        """Format a summary of sources used."""
        if not citations:
            return "No sources used."

        # Group by source type
        by_type: dict[SourceType, list[Citation]] = {}
        for c in citations:
            by_type.setdefault(c.source_type, []).append(c)

        parts = ["**Sources Used:**\n"]

        for source_type, type_citations in by_type.items():
            type_name = source_type.value.replace("_", " ").title()
            parts.append(f"\n*{type_name}:*\n")

            for c in type_citations:
                marker = "📝" if c.is_inferred else "📖"
                title = c.title or c.source_id
                conf = f"({c.confidence:.0%})" if c.confidence < 1.0 else ""
                parts.append(f"- {marker} {title} {conf}\n")

        return "".join(parts)
