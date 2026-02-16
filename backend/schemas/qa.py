"""Q&A (RAG) system schemas."""

from pydantic import BaseModel, Field


class QAProcessRequest(BaseModel):
    """Request to process a document for Q&A."""

    file_path: str = Field(description="Path to the uploaded file")
    chunk_size: int | None = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int | None = Field(default=200, ge=0, le=1000)

class QAQueryRequest(BaseModel):
    """Request to ask a question about content."""

    content_id: str = Field(description="ID of the processed content")
    question: str = Field(min_length=5, max_length=1000)
    language: str | None = Field(default="English")
    max_results: int | None = Field(default=3, ge=1, le=10)

class QAResponse(BaseModel):
    """Response from Q&A system."""

    answer: str
    sources: list[str] = Field(description="Source chunk IDs used to generate answer")
    confidence: float = Field(ge=0.0, le=1.0)
    context: list[dict] | None = Field(
        default=None, description="Full context chunks with text and metadata"
    )

class DocumentChunk(BaseModel):
    """Representation of a document chunk."""

    chunk_id: str
    content_id: str
    text: str
    chunk_index: int
    embedding: list[float] | None = None
    metadata: dict | None = None

class QAStatusResponse(BaseModel):
    """Status of Q&A processing."""

    content_id: str
    status: str  # processing, ready, failed
    total_chunks: int | None = None
    processed_chunks: int | None = None
    error: str | None = None

__all__ = [
    "DocumentChunk",
    "QAProcessRequest",
    "QAQueryRequest",
    "QAResponse",
    "QAStatusResponse",
]
