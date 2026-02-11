# API Reference

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## Overview

The Shiksha Setu API follows RESTful conventions with streaming support via Server-Sent Events. All endpoints are versioned under `/api/v2/` and require authentication unless otherwise noted.

**Base URL:** `http://localhost:8000/api/v2`
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)

---

## Authentication

### Register New User

```http
POST /auth/register
Content-Type: application/json
```

**Request:**
```json
{
  "name": "Student Name",
  "email": "student@example.com",
  "password": "securepassword123",
  "grade": 10,
  "preferred_language": "Hindi"
}
```

**Response (201):**
```json
{
  "id": "usr_abc123",
  "name": "Student Name",
  "email": "student@example.com",
  "grade": 10,
  "preferred_language": "Hindi",
  "created_at": "2025-12-05T10:30:00Z"
}
```

### Login

```http
POST /auth/login
Content-Type: application/json
```

**Request:**
```json
{
  "email": "student@example.com",
  "password": "securepassword123"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "usr_abc123",
    "name": "Student Name",
    "email": "student@example.com"
  }
}
```

### Refresh Token

```http
POST /auth/refresh
Content-Type: application/json
```

**Request:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 3600
}
```

---

## Question Answering

### Ask Question (Streaming)

```http
POST /qa/ask
Authorization: Bearer {token}
Content-Type: application/json
Accept: text/event-stream
```

**Request:**
```json
{
  "question": "न्यूटन का पहला नियम क्या है?",
  "language": "hi",
  "grade": 10,
  "stream": true,
  "include_citations": true
}
```

**Response (SSE Stream):**
```
event: message
data: {"type": "token", "content": "न्यूटन"}

event: message
data: {"type": "token", "content": " का"}

event: message
data: {"type": "token", "content": " पहला"}

... (token stream continues)

event: message
data: {"type": "citations", "sources": [
  {"id": "doc_123", "title": "NCERT Physics Ch3", "score": 0.92},
  {"id": "doc_456", "title": "Motion Laws", "score": 0.87}
]}

event: message
data: {"type": "done"}
```

### Ask Question (Non-Streaming)

```http
POST /qa/ask
Authorization: Bearer {token}
Content-Type: application/json
```

**Request:**
```json
{
  "question": "What is photosynthesis?",
  "language": "en",
  "grade": 8,
  "stream": false
}
```

**Response (200):**
```json
{
  "answer": "Photosynthesis is the process by which green plants convert light energy...",
  "language": "en",
  "citations": [
    {
      "id": "doc_789",
      "title": "NCERT Biology Chapter 1",
      "excerpt": "Plants use chlorophyll to capture sunlight...",
      "score": 0.94
    }
  ],
  "processing_time_ms": 1250,
  "model_used": "Qwen3-8B"
}
```

---

## Voice Processing

### Transcribe Audio

```http
POST /voice/transcribe
Authorization: Bearer {token}
Content-Type: multipart/form-data
```

**Request:**
```
audio: (binary file - WebM, MP3, WAV, M4A)
language: auto  (optional, auto-detect if not specified)
```

**Response (200):**
```json
{
  "transcript": "गुरुत्वाकर्षण बल क्या है?",
  "detected_language": "hi",
  "confidence": 0.96,
  "duration_seconds": 3.2,
  "processing_time_ms": 420
}
```

### Synthesize Speech

```http
POST /voice/synthesize
Authorization: Bearer {token}
Content-Type: application/json
```

**Request:**
```json
{
  "text": "Newton's First Law states that an object at rest...",
  "language": "en",
  "voice": "en-US-JennyNeural",
  "format": "mp3"
}
```

**Response (200):**
```
Content-Type: audio/mpeg
Content-Disposition: attachment; filename="speech.mp3"

(binary audio data)
```

### Voice-to-Voice Query

```http
POST /voice/query
Authorization: Bearer {token}
Content-Type: multipart/form-data
Accept: audio/mpeg
```

**Request:**
```
audio: (binary file)
output_format: mp3
```

**Response (200):**
```
Content-Type: audio/mpeg

(binary audio response - complete answer as speech)
```

---

## Document Management

### Upload Document

```http
POST /documents/upload
Authorization: Bearer {token}
Content-Type: multipart/form-data
```

**Request:**
```
file: (binary file - PDF, DOCX, TXT, images)
title: "NCERT Physics Class 11 Chapter 3"  (optional)
subject: "Physics"  (optional)
grade: 11  (optional)
```

**Response (201):**
```json
{
  "id": "doc_abc123",
  "filename": "NCERT_Physics_Ch3.pdf",
  "title": "NCERT Physics Class 11 Chapter 3",
  "size_bytes": 2560000,
  "pages": 45,
  "chunks_created": 245,
  "processing_time_ms": 3400,
  "status": "indexed"
}
```

### List Documents

```http
GET /documents
Authorization: Bearer {token}
```

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 20)
- `subject` (string): Filter by subject
- `grade` (int): Filter by grade

**Response (200):**
```json
{
  "documents": [
    {
      "id": "doc_abc123",
      "title": "NCERT Physics Class 11 Chapter 3",
      "subject": "Physics",
      "grade": 11,
      "chunks": 245,
      "created_at": "2025-12-05T10:30:00Z"
    }
  ],
  "total": 15,
  "page": 1,
  "pages": 1
}
```

### Delete Document

```http
DELETE /documents/{document_id}
Authorization: Bearer {token}
```

**Response (204):** No content

---

## Content Processing

### Simplify Text

```http
POST /content/simplify
Authorization: Bearer {token}
Content-Type: application/json
```

**Request:**
```json
{
  "text": "The mitochondria are membrane-bound organelles found in the cytoplasm...",
  "target_grade": 6,
  "language": "en"
}
```

**Response (200):**
```json
{
  "original": "The mitochondria are membrane-bound organelles...",
  "simplified": "Mitochondria are tiny parts inside cells that make energy...",
  "grade_level": 6,
  "readability_score": 72.5
}
```

### Translate Text

```http
POST /content/translate
Authorization: Bearer {token}
Content-Type: application/json
```

**Request:**
```json
{
  "text": "Newton's First Law of Motion states that...",
  "source_language": "en",
  "target_language": "hi"
}
```

**Response (200):**
```json
{
  "original": "Newton's First Law of Motion states that...",
  "translated": "न्यूटन का गति का पहला नियम बताता है कि...",
  "source_language": "en",
  "target_language": "hi"
}
```

---

## Chat Conversations

### Create Conversation

```http
POST /chat/conversations
Authorization: Bearer {token}
Content-Type: application/json
```

**Request:**
```json
{
  "title": "Physics Homework Help",
  "subject": "Physics"
}
```

**Response (201):**
```json
{
  "id": "conv_xyz789",
  "title": "Physics Homework Help",
  "subject": "Physics",
  "message_count": 0,
  "created_at": "2025-12-05T10:30:00Z"
}
```

### Send Message

```http
POST /chat/conversations/{conversation_id}/messages
Authorization: Bearer {token}
Content-Type: application/json
Accept: text/event-stream
```

**Request:**
```json
{
  "content": "Can you explain the second law of thermodynamics?",
  "stream": true
}
```

**Response:** SSE stream (same format as `/qa/ask`)

### Get Conversation History

```http
GET /chat/conversations/{conversation_id}/messages
Authorization: Bearer {token}
```

**Response (200):**
```json
{
  "messages": [
    {
      "id": "msg_001",
      "role": "user",
      "content": "What is entropy?",
      "timestamp": "2025-12-05T10:30:00Z"
    },
    {
      "id": "msg_002",
      "role": "assistant",
      "content": "Entropy is a measure of disorder or randomness...",
      "citations": [...],
      "timestamp": "2025-12-05T10:30:02Z"
    }
  ],
  "total": 2
}
```

---

## Health & Status

### Health Check

```http
GET /health
```

**Response (200):**
```json
{
  "status": "healthy",
  "version": "4.0.0",
  "environment": "production",
  "timestamp": "2025-12-05T10:30:00Z"
}
```

### Readiness Check

```http
GET /health/ready
```

**Response (200):**
```json
{
  "ready": true,
  "checks": {
    "database": "ok",
    "redis": "ok",
    "models": {
      "qwen": "loaded",
      "bge-m3": "loaded",
      "whisper": "standby"
    }
  }
}
```

### System Metrics

```http
GET /health/metrics
Authorization: Bearer {admin_token}
```

**Response (200):**
```json
{
  "memory": {
    "used_gb": 8.5,
    "total_gb": 16.0,
    "pressure": "normal"
  },
  "models": {
    "loaded": ["qwen", "bge-m3", "bge-reranker"],
    "standby": ["whisper", "indictrans2", "mms-tts"]
  },
  "requests": {
    "total_today": 1250,
    "avg_latency_ms": 1340
  }
}
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": "error_code",
  "message": "Human-readable error description",
  "request_id": "req_abc123",
  "details": {}  // Optional additional context
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `unauthorized` | 401 | Missing or invalid authentication |
| `forbidden` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `validation_error` | 422 | Invalid request parameters |
| `rate_limited` | 429 | Too many requests |
| `model_overload` | 503 | AI models are overloaded |
| `internal_error` | 500 | Unexpected server error |

### Rate Limit Headers

All responses include rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1733400000
```

---

## Supported Languages

| Code | Language | Native Name |
|------|----------|-------------|
| `en` | English | English |
| `hi` | Hindi | हिन्दी |
| `ta` | Tamil | தமிழ் |
| `te` | Telugu | తెలుగు |
| `bn` | Bengali | বাংলা |
| `mr` | Marathi | मराठी |
| `gu` | Gujarati | ગુજરાતી |
| `kn` | Kannada | ಕನ್ನಡ |
| `ml` | Malayalam | മലയാളം |
| `pa` | Punjabi | ਪੰਜਾਬੀ |
| `or` | Odia | ଓଡ଼ିଆ |

---

## SDK Examples

### Python

```python
import requests

class ShikshaSetuClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def ask(self, question: str, language: str = "en") -> dict:
        response = requests.post(
            f"{self.base_url}/qa/ask",
            headers=self.headers,
            json={"question": question, "language": language, "stream": False}
        )
        return response.json()

# Usage
client = ShikshaSetuClient("http://localhost:8000/api/v2", "your_token")
result = client.ask("What is photosynthesis?")
print(result["answer"])
```

### JavaScript/TypeScript

```typescript
async function askQuestion(question: string, language = 'en') {
  const response = await fetch('/api/v2/qa/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({ question, language, stream: false }),
  });

  return response.json();
}

// Streaming example
async function askStreaming(question: string, onToken: (token: string) => void) {
  const eventSource = new EventSource(`/api/v2/qa/ask?question=${encodeURIComponent(question)}`);

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'token') {
      onToken(data.content);
    } else if (data.type === 'done') {
      eventSource.close();
    }
  };
}
```

---

*For model pipeline details, see Section 7: Model Pipeline.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com
