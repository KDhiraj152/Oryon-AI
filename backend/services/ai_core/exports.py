"""
Export Service - Multiple Output Formats
=========================================

Provides export capabilities:
- JSON with full provenance
- Markdown formatted
- Plain text
- HTML (for embedding)
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain"
    HTML = "html"


@dataclass
class ExportedResponse:
    """An exported response in a specific format."""

    format: ExportFormat
    content: str
    filename: str
    mime_type: str
    size_bytes: int


class ExportService:
    """
    Service for exporting AI responses in various formats.

    Features:
    - JSON export with full metadata and provenance
    - Markdown with proper formatting
    - Plain text for simple use cases
    - HTML for web embedding
    """

    MIME_TYPES = {
        ExportFormat.JSON: "application/json",
        ExportFormat.MARKDOWN: "text/markdown",
        ExportFormat.PLAIN_TEXT: "text/plain",
        ExportFormat.HTML: "text/html",
    }

    FILE_EXTENSIONS = {
        ExportFormat.JSON: ".json",
        ExportFormat.MARKDOWN: ".md",
        ExportFormat.PLAIN_TEXT: ".txt",
        ExportFormat.HTML: ".html",
    }

    def export(
        self,
        content: str,
        format: ExportFormat,
        metadata: dict[str, Any] | None = None,
        sources: list[dict[str, Any]] | None = None,
        title: str | None = None,
    ) -> ExportedResponse:
        """
        Export content in the specified format.

        Args:
            content: Main content to export
            format: Target format
            metadata: Response metadata (tokens, latency, model, etc.)
            sources: Source citations
            title: Optional title for the export

        Returns:
            ExportedResponse with formatted content
        """
        metadata = metadata or {}
        sources = sources or []
        title = title or "AI Response"

        if format == ExportFormat.JSON:
            exported = self._export_json(content, metadata, sources, title)
        elif format == ExportFormat.MARKDOWN:
            exported = self._export_markdown(content, metadata, sources, title)
        elif format == ExportFormat.PLAIN_TEXT:
            exported = self._export_plain(content, metadata, sources, title)
        elif format == ExportFormat.HTML:
            exported = self._export_html(content, metadata, sources, title)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Generate filename
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", title)[:30]
        filename = f"{safe_title}_{timestamp}{self.FILE_EXTENSIONS[format]}"

        return ExportedResponse(
            format=format,
            content=exported,
            filename=filename,
            mime_type=self.MIME_TYPES[format],
            size_bytes=len(exported.encode("utf-8")),
        )

    def _export_json(
        self,
        content: str,
        metadata: dict[str, Any],
        sources: list[dict[str, Any]],
        title: str,
    ) -> str:
        """Export as JSON with full provenance."""
        data = {
            "title": title,
            "content": content,
            "metadata": {
                "exported_at": datetime.now(UTC).isoformat(),
                "format": "json",
                **metadata,
            },
            "sources": sources,
            "provenance": {
                "generator": "ShikshaSetu AI",
                "version": "1.0.0",
                "model": metadata.get("model", "unknown"),
                "confidence": metadata.get("confidence", 1.0),
            },
        }

        return json.dumps(data, indent=2, ensure_ascii=False)

    def _export_markdown(
        self,
        content: str,
        metadata: dict[str, Any],
        sources: list[dict[str, Any]],
        title: str,
    ) -> str:
        """Export as Markdown."""
        parts = [
            f"# {title}",
            "",
            content,
        ]

        # Add sources section
        if sources:
            parts.extend(
                [
                    "",
                    "---",
                    "",
                    "## Sources",
                    "",
                ]
            )

            for i, source in enumerate(sources, 1):
                title_str = source.get("title", source.get("source_id", f"Source {i}"))
                location = source.get("location", "")
                confidence = source.get("confidence", 1.0)
                is_inferred = source.get("is_inferred", False)

                marker = "📝 (inferred)" if is_inferred else "📖 (quoted)"

                parts.append(
                    f"{i}. {marker} **{title_str}**"
                    f"{f' — {location}' if location else ''}"
                    f" [{confidence:.0%}]"
                )

        # Add metadata footer
        parts.extend(
            [
                "",
                "---",
                "",
                "<details>",
                "<summary>Response Metadata</summary>",
                "",
            ]
        )

        if "tokens" in metadata:
            tokens = metadata["tokens"]
            parts.append(f"- **Tokens:** {tokens.get('total', 'N/A')}")

        if "latency_ms" in metadata:
            parts.append(f"- **Latency:** {metadata['latency_ms']:.0f}ms")

        if "model" in metadata:
            parts.append(f"- **Model:** {metadata['model']}")

        if "confidence" in metadata:
            parts.append(f"- **Confidence:** {metadata['confidence']:.0%}")

        parts.extend(
            [
                f"- **Exported:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
                "",
                "</details>",
            ]
        )

        return "\n".join(parts)

    def _export_plain(
        self,
        content: str,
        metadata: dict[str, Any],
        sources: list[dict[str, Any]],
        title: str,
    ) -> str:
        """Export as plain text (strip markdown formatting)."""
        # Strip markdown formatting
        plain = content

        # Remove code fences
        plain = re.sub(r"```\w*\n", "", plain)
        plain = plain.replace("```", "")

        # Remove headers
        plain = re.sub(r"^#{1,6}\s+", "", plain, flags=re.MULTILINE)

        # Remove bold/italic
        plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
        plain = re.sub(r"\*([^*]+)\*", r"\1", plain)
        plain = re.sub(r"__([^_]+)__", r"\1", plain)
        plain = re.sub(r"_([^_]+)_", r"\1", plain)

        # Remove links but keep text
        plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)

        parts = [
            title.upper(),
            "=" * len(title),
            "",
            plain,
        ]

        # Add sources
        if sources:
            parts.extend(
                [
                    "",
                    "SOURCES",
                    "-" * 7,
                ]
            )

            for i, source in enumerate(sources, 1):
                title_str = source.get("title", source.get("source_id", f"Source {i}"))
                parts.append(f"{i}. {title_str}")

        # Add metadata
        parts.extend(
            [
                "",
                "---",
                "Generated by ShikshaSetu AI",
                f"Exported: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
            ]
        )

        return "\n".join(parts)

    def _export_html(
        self,
        content: str,
        metadata: dict[str, Any],
        sources: list[dict[str, Any]],
        title: str,
    ) -> str:
        """Export as HTML."""
        # Convert markdown to basic HTML
        html_content = self._markdown_to_html(content)

        # Build sources HTML
        sources_html = ""
        if sources:
            sources_items = []
            for i, source in enumerate(sources, 1):
                title_str = source.get("title", source.get("source_id", f"Source {i}"))
                location = source.get("location", "")
                confidence = source.get("confidence", 1.0)
                is_inferred = source.get("is_inferred", False)

                marker = "📝" if is_inferred else "📖"

                sources_items.append(
                    f"<li>{marker} <strong>{title_str}</strong>"
                    f"{f' — {location}' if location else ''}"
                    f' <span class="confidence">[{confidence:.0%}]</span></li>'
                )

            sources_html = f"""
            <section class="sources">
                <h2>Sources</h2>
                <ol>
                    {"".join(sources_items)}
                </ol>
            </section>
            """

        # Build metadata HTML
        meta_items = []
        if "tokens" in metadata:
            meta_items.append(
                f"<li><strong>Tokens:</strong> {metadata['tokens'].get('total', 'N/A')}</li>"
            )
        if "latency_ms" in metadata:
            meta_items.append(
                f"<li><strong>Latency:</strong> {metadata['latency_ms']:.0f}ms</li>"
            )
        if "model" in metadata:
            meta_items.append(f"<li><strong>Model:</strong> {metadata['model']}</li>")
        if "confidence" in metadata:
            meta_items.append(
                f"<li><strong>Confidence:</strong> {metadata['confidence']:.0%}</li>"
            )

        metadata_html = f"""
        <details class="metadata">
            <summary>Response Metadata</summary>
            <ul>
                {"".join(meta_items)}
                <li><strong>Exported:</strong> {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}</li>
            </ul>
        </details>
        """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - ShikshaSetu AI</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
        }}
        h1 {{ color: #2563eb; }}
        h2 {{ color: #1e40af; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.5rem; }}
        pre {{
            background: #f3f4f6;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
        }}
        code {{
            background: #f3f4f6;
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
            font-family: 'SF Mono', Consolas, monospace;
        }}
        .sources {{ margin-top: 2rem; }}
        .sources ol {{ padding-left: 1.5rem; }}
        .sources li {{ margin: 0.5rem 0; }}
        .confidence {{ color: #6b7280; font-size: 0.875rem; }}
        .metadata {{
            margin-top: 2rem;
            padding: 1rem;
            background: #f9fafb;
            border-radius: 0.5rem;
        }}
        .metadata summary {{
            cursor: pointer;
            font-weight: 600;
        }}
        .metadata ul {{
            margin-top: 0.5rem;
            padding-left: 1.5rem;
        }}
        footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
            color: #6b7280;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <article>
        <h1>{title}</h1>
        <div class="content">
            {html_content}
        </div>
        {sources_html}
        {metadata_html}
    </article>
    <footer>
        <p>Generated by ShikshaSetu AI</p>
    </footer>
</body>
</html>"""

    def _markdown_to_html(self, markdown: str) -> str:
        """Simple markdown to HTML conversion."""
        html = markdown

        # Code blocks
        html = re.sub(
            r"```(\w*)\n(.*?)```",
            r'<pre><code class="language-\1">\2</code></pre>',
            html,
            flags=re.DOTALL,
        )

        # Inline code
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)

        # Headers
        html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", html)

        # Lists
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        html = re.sub(r"^(\d+)\. (.+)$", r"<li>\2</li>", html, flags=re.MULTILINE)

        # Paragraphs
        paragraphs = html.split("\n\n")
        html = "".join(
            f"<p>{p}</p>" if not p.startswith("<") else p
            for p in paragraphs
            if p.strip()
        )

        # Links
        html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)

        return html

    def get_supported_formats(self) -> list[dict[str, str]]:
        """Get list of supported export formats."""
        return [
            {
                "format": fmt.value,
                "mime_type": self.MIME_TYPES[fmt],
                "extension": self.FILE_EXTENSIONS[fmt],
            }
            for fmt in ExportFormat
        ]
