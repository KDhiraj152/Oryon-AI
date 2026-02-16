"""Application constants and error messages."""

# HTTP Status Messages
CONTENT_NOT_FOUND = "Content not found"
UNAUTHORIZED = "Not authorized"
INVALID_CREDENTIALS = "Invalid credentials"
USER_NOT_FOUND = "User not found"
USER_ALREADY_EXISTS = "User with this email already exists"

# Task Messages
TASK_NOT_FOUND = "Task not found"
TASK_PROCESSING = "Task is still processing"
TASK_FAILED = "Task failed"

# File Messages
FILE_TOO_LARGE = "File size exceeds maximum allowed"
INVALID_FILE_TYPE = "Invalid file type"
FILE_NOT_FOUND = "File not found"

# Validation Messages
INVALID_COMPLEXITY = "Grade level must be between 5 and 12"
INVALID_LANGUAGE = "Unsupported language"
TEXT_TOO_SHORT = "Text is too short for processing"
TEXT_TOO_LONG = "Text exceeds maximum length"

# Q&A Messages
NO_TEXT_AVAILABLE = "No text available for Q&A processing"
QUESTION_TOO_SHORT = "Question is too short"

# Rate Limiting
RATE_LIMIT_EXCEEDED = "Rate limit exceeded. Please try again later."

# Database Messages
DATABASE_ERROR = "Database error occurred"
CONNECTION_ERROR = "Failed to connect to database"

# Default Values
DEFAULT_COMPLEXITY = 8
DEFAULT_SUBJECT = "General"
DEFAULT_LANGUAGE = "English"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_MAX_UPLOAD_SIZE_MB = 100
DEFAULT_RATE_LIMIT_PER_MINUTE = 60

# Complexity Level Bounds
MIN_COMPLEXITY = 1
MAX_COMPLEXITY = 12

# Supported Languages
SUPPORTED_LANGUAGES = [
    "English",
    "Hindi",
    "Tamil",
    "Telugu",
    "Bengali",
    "Marathi",
    "Gujarati",
    "Kannada",
    "Malayalam",
    "Punjabi",
]

# Supported File Types
SUPPORTED_FILE_TYPES = {
    "text/plain": [".txt"],
    "application/pdf": [".pdf"],
    "application/msword": [".doc"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
        ".docx"
    ],
}

# NOTE: Subject constraints removed - system is now unconstrained like ChatGPT/Perplexity
# The multimodel RAG system handles any topic without hardcoded subject lists

# Supported Formats
SUPPORTED_FORMATS = ["text", "audio", "both"]

__all__ = [
    "CONNECTION_ERROR",
    "CONTENT_NOT_FOUND",
    "DATABASE_ERROR",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_COMPLEXITY",
    "DEFAULT_LANGUAGE",
    "DEFAULT_MAX_UPLOAD_SIZE_MB",
    "DEFAULT_RATE_LIMIT_PER_MINUTE",
    "DEFAULT_SUBJECT",
    "FILE_NOT_FOUND",
    "FILE_TOO_LARGE",
    "INVALID_COMPLEXITY",
    "INVALID_CREDENTIALS",
    "INVALID_FILE_TYPE",
    "INVALID_LANGUAGE",
    "MAX_COMPLEXITY",
    "MIN_COMPLEXITY",
    "NO_TEXT_AVAILABLE",
    "QUESTION_TOO_SHORT",
    "RATE_LIMIT_EXCEEDED",
    "SUPPORTED_FILE_TYPES",
    "SUPPORTED_FORMATS",
    "SUPPORTED_LANGUAGES",
    "TASK_FAILED",
    "TASK_NOT_FOUND",
    "TASK_PROCESSING",
    "TEXT_TOO_LONG",
    "TEXT_TOO_SHORT",
    "UNAUTHORIZED",
    "USER_ALREADY_EXISTS",
    "USER_NOT_FOUND",
]
