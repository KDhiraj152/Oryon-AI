# API Documentation

This directory contains the API implementations for the multilingual education content pipeline.

## Overview

The system provides two API implementations:
- **Flask API** (`flask_app.py`): Traditional REST API with comprehensive error handling
- **FastAPI** (`fastapi_app.py`): High-performance async API with automatic OpenAPI documentation

Both APIs provide the same endpoints with consistent functionality.

## Endpoints

### 1. Process Content
**POST** `/api/process-content` (Flask) or `/api/v1/process-content` (FastAPI)

Process educational content through the four-stage pipeline (simplification → translation → validation → speech generation).

**Rate Limit:** 10 requests per minute

**Request Body:**
```json
{
  "input_data": "Educational content text to process",
  "target_language": "Hindi",
  "grade_level": 8,
  "subject": "Mathematics",
  "output_format": "both"
}
```

**Parameters:**
- `input_data` (string, required): Text content to process
- `target_language` (string, required): Target language (Hindi, Tamil, Telugu, Bengali, Marathi)
- `grade_level` (integer, required): Grade level (5-12)
- `subject` (string, required): Subject area (Mathematics, Science, Social Studies, English, History, Geography)
- `output_format` (string, required): Output format (text, audio, both)

**Response (201 Created):**
```json
{
  "id": "uuid",
  "simplified_text": "Simplified content...",
  "translated_text": "Translated content...",
  "language": "Hindi",
  "grade_level": 8,
  "subject": "Mathematics",
  "audio_url": "/api/content/{id}/audio",
  "ncert_alignment_score": 0.85,
  "audio_accuracy_score": 0.92,
  "validation_status": "passed",
  "created_at": "2024-01-01T00:00:00",
  "metadata": {
    "output_format": "both",
    "total_processing_time_ms": 5000
  }
}
```

### 2. Get Content
**GET** `/api/content/{content_id}` (Flask) or `/api/v1/content/{content_id}` (FastAPI)

Retrieve processed content by ID.

**Rate Limit:** 100 requests per minute

**Query Parameters:**
- `compress` (boolean, optional): Return gzip-compressed content for bandwidth optimization

**Response (200 OK):**
```json
{
  "id": "uuid",
  "original_text": "Original content...",
  "simplified_text": "Simplified content...",
  "translated_text": "Translated content...",
  "language": "Hindi",
  "grade_level": 8,
  "subject": "Mathematics",
  "audio_url": "/api/content/{id}/audio",
  "ncert_alignment_score": 0.85,
  "audio_accuracy_score": 0.92,
  "created_at": "2024-01-01T00:00:00",
  "metadata": {}
}
```

### 3. Get Audio
**GET** `/api/content/{content_id}/audio` (Flask) or `/api/v1/content/{content_id}/audio` (FastAPI)

Retrieve audio file for processed content.

**Rate Limit:** 50 requests per minute

**Response:** Audio file (MP3 format)

### 4. Batch Download
**POST** `/api/batch-download` (Flask) or `/api/v1/batch-download` (FastAPI)

Create batch download package for offline access (up to 50 items).

**Rate Limit:** 5 requests per minute

**Request Body:**
```json
{
  "content_ids": ["uuid1", "uuid2", "uuid3"],
  "include_audio": true
}
```

**Response (200 OK):**
```json
{
  "package_id": "batch_3_items",
  "item_count": 3,
  "include_audio": true,
  "total_size_bytes": 1048576,
  "total_size_mb": 1.0,
  "performance": {
    "content_size_bytes": 1048576,
    "estimated_load_time_seconds": 2.5,
    "meets_requirement": true
  },
  "contents": [
    {
      "id": "uuid1",
      "simplified_text": "...",
      "translated_text": "...",
      "language": "Hindi",
      "grade_level": 8,
      "subject": "Mathematics",
      "audio_url": "/api/content/uuid1/audio",
      "ncert_alignment_score": 0.85,
      "metadata": {}
    }
  ]
}
```

### 5. Search Content
**GET** `/api/content/search` (Flask) or `/api/v1/content/search` (FastAPI)

Search content with filters and pagination.

**Rate Limit:** 100 requests per minute

**Query Parameters:**
- `language` (string, optional): Filter by language
- `grade` (integer, optional): Filter by grade level (5-12)
- `subject` (string, optional): Filter by subject
- `limit` (integer, optional): Maximum results (default 20, max 100)
- `offset` (integer, optional): Pagination offset (default 0)

**Response (200 OK):**
```json
{
  "total_count": 100,
  "limit": 20,
  "offset": 0,
  "results": [
    {
      "id": "uuid",
      "language": "Hindi",
      "grade_level": 8,
      "subject": "Mathematics",
      "translated_text_preview": "First 200 characters...",
      "ncert_alignment_score": 0.85,
      "audio_available": true,
      "created_at": "2024-01-01T00:00:00"
    }
  ],
  "filters": {
    "language": "Hindi",
    "grade": 8,
    "subject": "Mathematics"
  }
}
```

## Error Responses

### 400 Bad Request
Invalid request parameters or validation failure.

```json
{
  "error": "Validation failed",
  "details": "target_language must be one of ['Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi']"
}
```

### 404 Not Found
Resource not found.

```json
{
  "error": "Content not found"
}
```

### 429 Too Many Requests
Rate limit exceeded.

```json
{
  "error": "Rate limit exceeded",
  "details": "10 per 1 minute"
}
```

### 500 Internal Server Error
Server-side processing error.

```json
{
  "error": "Processing failed",
  "details": "Stage translation failed after 3 attempts"
}
```

## Rate Limiting

Rate limiting is implemented to prevent API abuse:

| Endpoint | Rate Limit |
|----------|------------|
| Process Content | 10 per minute |
| Get Content | 100 per minute |
| Get Audio | 50 per minute |
| Batch Download | 5 per minute |
| Search Content | 100 per minute |

Global limits:
- 200 requests per day
- 50 requests per hour

## Running the APIs

### Flask API
```bash
# Development
python src/api/flask_app.py

# Production
gunicorn src.api.flask_app:app --bind 0.0.0.0:5000
```

### FastAPI
```bash
# Development
python src/api/fastapi_app.py

# Production
uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

## Environment Variables

```bash
# Flask Configuration
FLASK_SECRET_KEY=your-secret-key
FLASK_DEBUG=false
FLASK_PORT=5000

# FastAPI Configuration
FASTAPI_PORT=8000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/education_content

# Rate Limiting (optional)
REDIS_URL=redis://localhost:6379/0

# Caching
CACHE_DIR=data/cache
```

## API Documentation

FastAPI provides automatic interactive documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Run API tests:
```bash
pytest tests/test_api_endpoints.py -v
```

## Features

### Request Validation
- Automatic parameter validation
- Type checking
- Range validation for grade levels
- Enum validation for languages, subjects, and formats

### Error Handling
- Comprehensive error messages
- Proper HTTP status codes
- Detailed error logging
- Retry logic for transient failures

### Performance Optimization
- Gzip compression for bandwidth optimization
- Content caching for offline access
- 2G network performance validation
- Lazy loading for audio files

### Security
- Rate limiting to prevent abuse
- Input validation and sanitization
- CORS configuration
- Request size limits (16MB max)

### Monitoring
- Request logging
- Performance metrics
- Error tracking
- Pipeline stage metrics
