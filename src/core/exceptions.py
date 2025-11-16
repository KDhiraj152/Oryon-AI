"""Custom exception classes for ShikshaSetu."""
from datetime import datetime, timezone


class ShikshaSetuException(Exception):
    """Base exception for all ShikshaSetu errors."""
    
    def __init__(self, detail: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR"):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc).isoformat()
        super().__init__(detail)
    
    def to_dict(self):
        """Convert exception to dictionary for API response."""
        return {
            "error": self.error_code,
            "detail": self.detail,
            "status_code": self.status_code,
            "timestamp": self.timestamp
        }


class ContentNotFoundError(ShikshaSetuException):
    """Raised when content is not found."""
    
    def __init__(self, content_id: str):
        super().__init__(
            detail=f"Content with ID {content_id} not found",
            status_code=404,
            error_code="CONTENT_NOT_FOUND"
        )


class DocumentNotFoundError(ShikshaSetuException):
    """Raised when document is not found."""
    
    def __init__(self, document_id: str):
        super().__init__(
            detail=f"Document with ID {document_id} not found",
            status_code=404,
            error_code="DOCUMENT_NOT_FOUND"
        )


class InvalidFileError(ShikshaSetuException):
    """Raised when uploaded file is invalid."""
    
    def __init__(self, reason: str):
        super().__init__(
            detail=f"Invalid file: {reason}",
            status_code=400,
            error_code="INVALID_FILE"
        )


class TaskNotFoundError(ShikshaSetuException):
    """Raised when task is not found."""
    
    def __init__(self, task_id: str):
        super().__init__(
            detail=f"Task with ID {task_id} not found",
            status_code=404,
            error_code="TASK_NOT_FOUND"
        )


class AuthenticationError(ShikshaSetuException):
    """Raised when authentication fails."""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            detail=detail,
            status_code=401,
            error_code="AUTHENTICATION_FAILED"
        )


class AuthorizationError(ShikshaSetuException):
    """Raised when user is not authorized."""
    
    def __init__(self, detail: str = "Not authorized to access this resource"):
        super().__init__(
            detail=detail,
            status_code=403,
            error_code="AUTHORIZATION_FAILED"
        )


class ValidationError(ShikshaSetuException):
    """Raised when input validation fails."""
    
    def __init__(self, detail: str):
        super().__init__(
            detail=detail,
            status_code=422,
            error_code="VALIDATION_ERROR"
        )


class RateLimitError(ShikshaSetuException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            detail=detail,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED"
        )


class ProcessingError(ShikshaSetuException):
    """Raised when content processing fails."""
    
    def __init__(self, detail: str):
        super().__init__(
            detail=f"Processing failed: {detail}",
            status_code=500,
            error_code="PROCESSING_ERROR"
        )


class DatabaseError(ShikshaSetuException):
    """Raised when database operation fails."""
    
    def __init__(self, detail: str):
        super().__init__(
            detail=f"Database error: {detail}",
            status_code=500,
            error_code="DATABASE_ERROR"
        )
