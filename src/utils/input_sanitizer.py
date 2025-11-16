"""Input validation and sanitization utilities for security."""
import re
import os
import mimetypes
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import magic  # python-magic for file type detection
import hashlib


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputSanitizer:
    """Sanitize and validate user inputs."""
    
    # File upload constraints
    MAX_FILE_SIZE_MB = 10
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'  # .docx
    }
    ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
    
    # Text content constraints
    MAX_TEXT_LENGTH = 100000  # 100KB max text
    MIN_TEXT_LENGTH = 10
    
    # Malicious patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(--|\#|\/\*|\*\/)",  # SQL comments
        r"(;.*\b(DROP|DELETE|UPDATE|INSERT)\b)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"onerror\s*=",
        r"onload\s*=",
        r"onclick\s*=",
        r"<iframe[^>]*>",
        r"<embed[^>]*>",
        r"<object[^>]*>",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.",
        r"%2e%2e",
        r"\.\.\\",
    ]
    
    # Supported languages and parameters
    SUPPORTED_LANGUAGES = {'English', 'Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi'}
    # ISO 639-1 language code mapping
    LANGUAGE_CODE_MAP = {
        'en': 'English',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'bn': 'Bengali',
        'mr': 'Marathi'
    }
    SUPPORTED_SUBJECTS = {'Mathematics', 'Science', 'Social Studies', 'English', 'History', 'Geography'}
    SUPPORTED_FORMATS = {'text', 'audio', 'both'}
    GRADE_RANGE = (5, 12)
    
    @classmethod
    def sanitize_text_input(cls, text: str) -> str:
        """
        Sanitize text input for processing.
        
        Args:
            text: Raw text input
            
        Returns:
            Sanitized text
            
        Raises:
            ValidationError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValidationError("Text input must be a string")
        
        # Check length
        if len(text) < cls.MIN_TEXT_LENGTH:
            raise ValidationError(f"Text must be at least {cls.MIN_TEXT_LENGTH} characters")
        
        if len(text) > cls.MAX_TEXT_LENGTH:
            raise ValidationError(f"Text must not exceed {cls.MAX_TEXT_LENGTH} characters")
        
        # Check for malicious patterns
        cls._check_sql_injection(text)
        cls._check_xss(text)
        cls._check_path_traversal(text)
        
        # Strip dangerous characters but preserve educational content
        sanitized = text.strip()
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        return sanitized
    
    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """
        Alias for sanitize_text_input for backwards compatibility.
        
        Args:
            text: Raw text input
            
        Returns:
            Sanitized text
        """
        return cls.sanitize_text_input(text)
    
    @classmethod
    def validate_grade_level(cls, grade: int) -> int:
        """
        Validate grade level.
        
        Args:
            grade: Grade level to validate
            
        Returns:
            Validated grade level
            
        Raises:
            ValidationError: If grade is invalid
        """
        if not isinstance(grade, int):
            raise ValidationError("grade_level must be an integer")
        if not (cls.GRADE_RANGE[0] <= grade <= cls.GRADE_RANGE[1]):
            raise ValidationError(
                f"grade_level must be between {cls.GRADE_RANGE[0]} and {cls.GRADE_RANGE[1]}"
            )
        return grade
    
    @classmethod
    def validate_language(cls, language: str) -> str:
        """
        Validate target language. Accepts both ISO codes (hi, ta) and full names (Hindi, Tamil).
        
        Args:
            language: Language to validate (ISO code or full name)
            
        Returns:
            Validated language (full name)
            
        Raises:
            ValidationError: If language is not supported
        """
        if not isinstance(language, str):
            raise ValidationError("Language must be a string")
        
        lang_clean = language.strip().lower()
        
        # Check if it's an ISO code first
        if lang_clean in cls.LANGUAGE_CODE_MAP:
            return cls.LANGUAGE_CODE_MAP[lang_clean]
        
        # Normalize to title case for comparison
        lang_normalized = language.strip().title()
        
        # Check if it's in supported languages (case-insensitive)
        if lang_normalized not in cls.SUPPORTED_LANGUAGES:
            raise ValidationError(
                f"Unsupported language '{language}'. "
                f"Supported: {', '.join(cls.SUPPORTED_LANGUAGES)} or ISO codes: {', '.join(cls.LANGUAGE_CODE_MAP.keys())}"
            )
        return lang_normalized
    
    @classmethod
    def validate_subject(cls, subject: str) -> str:
        """
        Validate subject.
        
        Args:
            subject: Subject to validate
            
        Returns:
            Validated subject
            
        Raises:
            ValidationError: If subject is not supported
        """
        if not isinstance(subject, str):
            raise ValidationError("Subject must be a string")
        
        # Normalize to title case for comparison
        subject_normalized = subject.strip().title()
        
        # Check if it's in supported subjects (case-insensitive)
        if subject_normalized not in cls.SUPPORTED_SUBJECTS:
            raise ValidationError(
                f"Unsupported subject '{subject}'. "
                f"Supported: {', '.join(cls.SUPPORTED_SUBJECTS)}"
            )
        return subject_normalized
    
    @classmethod
    def validate_file_upload(
        cls,
        filename: str,
        file_bytes: bytes,
        max_size_mb: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate uploaded file for security when content is provided in-memory.
        
        Args:
            filename: Original filename
            file_bytes: Raw file bytes to inspect
            max_size_mb: Optional override for maximum file size constraint
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ValidationError: If file is invalid or malicious
        """
        file_size = len(file_bytes)
        max_size_bytes = (max_size_mb or cls.MAX_FILE_SIZE_MB) * 1024 * 1024
        validation_result = {
            'valid': False,
            'filename': filename,
            'size': file_size,
            'mime_type': None,
            'extension': None,
            'hash': None,
            'errors': []
        }
        
        # Check file size
        if file_size > max_size_bytes:
            raise ValidationError(
                f"File size exceeds maximum allowed size of {max_size_mb or cls.MAX_FILE_SIZE_MB}MB"
            )
        
        if file_size == 0:
            raise ValidationError("File is empty")
        
        # Check filename
        cls._validate_filename(filename)
        
        # Check extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in cls.ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"File extension '{file_ext}' not allowed. "
                f"Allowed: {', '.join(cls.ALLOWED_EXTENSIONS)}"
            )
        validation_result['extension'] = file_ext
        
        # Check MIME type using magic bytes
        try:
            detected_mime = magic.from_buffer(file_bytes, mime=True)
            validation_result['mime_type'] = detected_mime
            
            if detected_mime not in cls.ALLOWED_MIME_TYPES:
                raise ValidationError(
                    f"File type '{detected_mime}' not allowed. "
                    f"File extension and content must match."
                )
        except Exception as e:
            raise ValidationError(f"Failed to detect file type: {str(e)}")
        
        # Calculate file hash for deduplication
        validation_result['hash'] = cls._calculate_file_hash(file_bytes)
        
        # Scan for malicious content in text files
        if detected_mime == 'text/plain':
            cls._scan_text_file(file_bytes)
        
        validation_result['valid'] = True
        return validation_result
    
    @classmethod
    def validate_request_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate API request parameters.
        
        Args:
            params: Dictionary of request parameters
            
        Returns:
            Validated and sanitized parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        validated = {}
        
        # Validate target_language
        validated.update(cls._validate_language(params))
        
        # Validate grade_level
        validated.update(cls._validate_grade(params))
        
        # Validate subject
        validated.update(cls._validate_subject(params))
        
        # Validate output_format
        validated.update(cls._validate_format(params))
        
        # Validate input_data (text content)
        if 'input_data' in params:
            validated['input_data'] = cls.sanitize_text_input(params['input_data'])
        
        return validated
    
    @classmethod
    def _validate_language(cls, params: Dict[str, Any]) -> Dict[str, str]:
        """Validate target_language parameter."""
        if 'target_language' not in params:
            return {}
        
        lang = params['target_language']
        if not isinstance(lang, str):
            raise ValidationError("target_language must be a string")
        if lang not in cls.SUPPORTED_LANGUAGES:
            raise ValidationError(
                f"Unsupported language '{lang}'. "
                f"Supported: {', '.join(cls.SUPPORTED_LANGUAGES)}"
            )
        return {'target_language': lang}
    
    @classmethod
    def _validate_grade(cls, params: Dict[str, Any]) -> Dict[str, int]:
        """Validate grade_level parameter."""
        if 'grade_level' not in params:
            return {}
        
        grade = params['grade_level']
        if not isinstance(grade, int):
            raise ValidationError("grade_level must be an integer")
        if not (cls.GRADE_RANGE[0] <= grade <= cls.GRADE_RANGE[1]):
            raise ValidationError(
                f"grade_level must be between {cls.GRADE_RANGE[0]} and {cls.GRADE_RANGE[1]}"
            )
        return {'grade_level': grade}
    
    @classmethod
    def _validate_subject(cls, params: Dict[str, Any]) -> Dict[str, str]:
        """Validate subject parameter."""
        if 'subject' not in params:
            return {}
        
        subject = params['subject']
        if not isinstance(subject, str):
            raise ValidationError("subject must be a string")
        if subject not in cls.SUPPORTED_SUBJECTS:
            raise ValidationError(
                f"Unsupported subject '{subject}'. "
                f"Supported: {', '.join(cls.SUPPORTED_SUBJECTS)}"
            )
        return {'subject': subject}
    
    @classmethod
    def _validate_format(cls, params: Dict[str, Any]) -> Dict[str, str]:
        """Validate output_format parameter."""
        if 'output_format' not in params:
            return {}
        
        fmt = params['output_format']
        if not isinstance(fmt, str):
            raise ValidationError("output_format must be a string")
        if fmt not in cls.SUPPORTED_FORMATS:
            raise ValidationError(
                f"Unsupported output format '{fmt}'. "
                f"Supported: {', '.join(cls.SUPPORTED_FORMATS)}"
            )
        return {'output_format': fmt}
    
    @classmethod
    def sanitize_query_string(cls, query: str) -> str:
        """
        Sanitize search query string.
        
        Args:
            query: Raw query string
            
        Returns:
            Sanitized query
        """
        if not isinstance(query, str):
            raise ValidationError("Query must be a string")
        
        # Check length
        if len(query) > 500:
            raise ValidationError("Query too long (max 500 characters)")
        
        # Check for malicious patterns
        cls._check_sql_injection(query)
        cls._check_xss(query)
        
        # Remove special characters but keep spaces and basic punctuation
        sanitized = re.sub(r'[^\w\s\-.,!?]', '', query)
        
        return sanitized.strip()
    
    @classmethod
    def validate_content_ids(cls, content_ids: List[str], max_count: int = 50) -> List[str]:
        """
        Validate list of content IDs (UUIDs).
        
        Args:
            content_ids: List of content ID strings
            max_count: Maximum number of IDs allowed
            
        Returns:
            List of validated UUIDs
            
        Raises:
            ValidationError: If IDs are invalid
        """
        if not isinstance(content_ids, list):
            raise ValidationError("content_ids must be a list")
        
        if len(content_ids) == 0:
            raise ValidationError("content_ids cannot be empty")
        
        if len(content_ids) > max_count:
            raise ValidationError(f"Maximum {max_count} content IDs allowed")
        
        # Validate UUID format
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        
        validated = []
        for content_id in content_ids:
            if not isinstance(content_id, str):
                raise ValidationError("Each content_id must be a string")
            
            if not uuid_pattern.match(content_id):
                raise ValidationError(f"Invalid UUID format: {content_id}")
            
            validated.append(content_id.lower())
        
        return validated
    
    @classmethod
    def validate_uuid(cls, uuid_string: str) -> str:
        """
        Validate a single UUID string.
        
        Args:
            uuid_string: UUID string to validate
            
        Returns:
            Validated UUID in lowercase
            
        Raises:
            ValidationError: If UUID is invalid
        """
        if not isinstance(uuid_string, str):
            raise ValidationError("UUID must be a string")
        
        # Validate UUID format
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        
        if not uuid_pattern.match(uuid_string):
            raise ValidationError(f"Invalid UUID format: {uuid_string}")
        
        return uuid_string.lower()
    
    # Private helper methods
    
    @classmethod
    def _check_sql_injection(cls, text: str) -> None:
        """Check for SQL injection patterns."""
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValidationError("Potential SQL injection detected")
    
    @classmethod
    def _check_xss(cls, text: str) -> None:
        """Check for XSS patterns."""
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValidationError("Potential XSS attack detected")
    
    @classmethod
    def _check_path_traversal(cls, text: str) -> None:
        """Check for path traversal patterns."""
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValidationError("Potential path traversal detected")
    
    @classmethod
    def _validate_filename(cls, filename: str) -> None:
        """Validate filename for security."""
        if not filename:
            raise ValidationError("Filename cannot be empty")
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise ValidationError("Invalid filename: path components not allowed")
        
        # Check for null bytes
        if '\x00' in filename:
            raise ValidationError("Invalid filename: null bytes not allowed")
        
        # Check length
        if len(filename) > 255:
            raise ValidationError("Filename too long (max 255 characters)")
        
        # Check for dangerous characters
        if re.search(r'[<>:"|?*]', filename):
            raise ValidationError("Filename contains invalid characters")
    
    @classmethod
    def _calculate_file_hash(cls, file_bytes: bytes) -> str:
        """Calculate SHA-256 hash from in-memory bytes."""
        sha256_hash = hashlib.sha256()
        sha256_hash.update(file_bytes)
        return sha256_hash.hexdigest()
    
    @classmethod
    def _scan_text_file(cls, file_bytes: bytes) -> None:
        """Scan text bytes for malicious content."""
        try:
            content = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            raise ValidationError("File encoding not supported. Use UTF-8.")
        snippet = content[:cls.MAX_TEXT_LENGTH]
        cls._check_sql_injection(snippet)
        cls._check_xss(snippet)
        cls._check_path_traversal(snippet)


class RateLimitValidator:
    """Validate rate limits and request quotas."""
    
    # In-memory storage (use Redis in production)
    _request_counts: Dict[str, Dict[str, int]] = {}
    
    # Rate limit rules
    LIMITS = {
        'process_content': {'requests': 10, 'window': 60},  # 10 per minute
        'get_content': {'requests': 100, 'window': 60},  # 100 per minute
        'search': {'requests': 100, 'window': 60},  # 100 per minute
        'batch_download': {'requests': 5, 'window': 60},  # 5 per minute
    }
    
    @classmethod
    def check_rate_limit(cls, client_id: str, endpoint: str) -> Tuple[bool, int]:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_id: Client identifier (IP address, API key, etc.)
            endpoint: API endpoint name
            
        Returns:
            Tuple of (allowed: bool, remaining_requests: int)
        """
        if endpoint not in cls.LIMITS:
            return True, -1
        
        limit_config = cls.LIMITS[endpoint]
        max_requests = limit_config['requests']
        
        # Initialize tracking for client+endpoint
        key = f"{client_id}:{endpoint}"
        if key not in cls._request_counts:
            cls._request_counts[key] = {'count': 0, 'reset_at': 0}
        
        import time
        current_time = int(time.time())
        
        # Reset counter if window expired
        if current_time >= cls._request_counts[key]['reset_at']:
            cls._request_counts[key] = {
                'count': 0,
                'reset_at': current_time + limit_config['window']
            }
        
        # Check if limit exceeded
        current_count = cls._request_counts[key]['count']
        if current_count >= max_requests:
            return False, 0
        
        # Increment counter
        cls._request_counts[key]['count'] += 1
        remaining = max_requests - cls._request_counts[key]['count']
        
        return True, remaining


# Export main classes
__all__ = ['InputSanitizer', 'RateLimitValidator', 'ValidationError']
