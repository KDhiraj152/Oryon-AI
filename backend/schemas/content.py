"""Content processing request/response schemas."""

from pydantic import BaseModel, Field


class ChunkedUploadRequest(BaseModel):
    """Chunked upload metadata."""

    filename: str
    chunk_index: int
    total_chunks: int
    upload_id: str
    checksum: str | None = None

class ProcessRequest(BaseModel):
    """Full pipeline processing request."""

    subject: str | None = None
    target_languages: list[str] = []
    output_format: str = Field(default="both", pattern="^(text|audio|both)$")
    validation_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

class SimplifyRequest(BaseModel):
    """Text simplification request."""

    text: str = Field(min_length=10, max_length=50000)

class TranslateRequest(BaseModel):
    """Translation request."""

    text: str = Field(min_length=10, max_length=50000)
    target_languages: list[str] | None = None
    source_language: str | None = None  # Backward compatibility
    target_language: str | None = None  # Backward compatibility

    def get_target_languages(self) -> list[str]:
        """Get target languages from either format."""
        if self.target_languages:
            return self.target_languages
        elif self.target_language:
            return [self.target_language]
        return ["Hindi"]

class ValidateRequest(BaseModel):
    """Content validation request."""

    text: str | None = Field(None, min_length=10)  # Backward compatibility
    original_text: str | None = Field(None, min_length=10)
    processed_text: str | None = Field(None, min_length=10)
    language: str | None = Field(default="English")

    def get_texts(self) -> tuple[str, str]:
        """Get original and processed texts."""
        if self.text:
            # Use text for both if only text is provided
            return (self.text, self.text)
        return (self.original_text or "", self.processed_text or "")

class TTSRequest(BaseModel):
    """Text-to-speech request."""

    text: str = Field(min_length=10, max_length=10000)
    language: str

class FeedbackRequest(BaseModel):
    """User feedback request."""

    content_id: str
    rating: int = Field(ge=1, le=5)
    feedback_text: str | None = None
    issue_type: str | None = None

class TaskResponse(BaseModel):
    """Task status response."""

    task_id: str
    state: str
    result: dict | None = None
    error: str | None = None
    progress: int | None = Field(None, ge=0, le=100)

class ContentResponse(BaseModel):
    """Content retrieval response."""

    content_id: str
    original_text: str
    simplified_text: str | None = None
    translations: dict | None = None
    validation_score: float | None = None
    audio_files: dict | None = None
    created_at: str
    metadata: dict | None = None

__all__ = [
    "ChunkedUploadRequest",
    "ContentResponse",
    "FeedbackRequest",
    "ProcessRequest",
    "SimplifyRequest",
    "TTSRequest",
    "TaskResponse",
    "TranslateRequest",
    "ValidateRequest",
]
