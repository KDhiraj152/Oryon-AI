# 🔧 ShikshaSetu API Reference

**FastAPI Production API • JWT Auth • Rate Limited**

Base URL: `http://localhost:8000`  
Interactive Docs: `http://localhost:8000/docs`

---

## 🔐 Authentication

All protected endpoints require JWT authentication via Bearer token.

**Header Format:**
```http
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Token Expiry:**
- Access Token: 30 minutes
- Refresh Token: 7 days

---

## 📋 Endpoints Overview

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get tokens
- `POST /api/v1/auth/refresh` - Refresh access token
- `GET /api/v1/auth/me` - Get user profile

### Content Processing
- `POST /api/v1/upload` - Upload file
- `POST /api/v1/process` - Full AI pipeline
- `POST /api/v1/simplify` - Simplify text
- `POST /api/v1/translate` - Translate text
- `POST /api/v1/validate` - Validate content
- `POST /api/v1/tts` - Generate audio

### Q&A System (RAG)
- `POST /api/v1/qa/process` - Process document for Q&A
- `POST /api/v1/qa/ask` - Ask questions
- `GET /api/v1/qa/history/{id}` - Get chat history

### Management
- `GET /api/v1/tasks/{task_id}` - Get task status
- `DELETE /api/v1/tasks/{task_id}` - Cancel task
- `GET /api/v1/content/{content_id}` - Get content
- `GET /api/v1/audio/{content_id}` - Stream audio
- `POST /api/v1/feedback` - Submit feedback

### Monitoring
- `GET /health` - Basic health check
- `GET /health/detailed` - System diagnostics

---

## 🔒 Authentication Endpoints

### 1. Register User
**POST** `/api/v1/auth/register`

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "full_name": "John Doe"
}
```

**Response:** `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

**Errors:**
- `400` - Invalid email or weak password
- `409` - Email already registered

---

### 2. Login
**POST** `/api/v1/auth/login`

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Response:** Same as registration

**Errors:**
- `401` - Invalid credentials

---

### 3. Refresh Token
**POST** `/api/v1/auth/refresh`

**Request:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

**Response:** New access and refresh tokens

---

### 4. Get User Profile
**GET** `/api/v1/auth/me`

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "created_at": "2025-11-16T10:00:00Z"
}
```

---

## 📄 Content Processing Endpoints

### 1. Upload File
**POST** `/api/v1/upload`

**Request:** Multipart form data
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "file_id": "abc123",
  "file_path": "data/uploads/2025/11/16/abc123_document.pdf",
  "original_filename": "document.pdf"
}
```

**Constraints:**
- Max size: 100MB
- Allowed: PDF, TXT, DOCX

---

### 2. Process Content
**POST** `/api/v1/process`

Full AI pipeline (simplify + translate + validate + TTS).

**Request:**
```json
{
  "file_path": "data/uploads/abc123_doc.pdf",
  "grade_level": 8,
  "subject": "Science",
  "target_languages": ["Hindi", "Tamil"],
  "output_format": "both"
}
```

**Response:**
```json
{
  "task_id": "task_abc123",
  "status": "processing"
}
```

---

### 3. Simplify Text
**POST** `/api/v1/simplify`

**Request:**
```json
{
  "text": "Complex scientific text...",
  "grade_level": 8
}
```

**Response:**
```json
{
  "task_id": "task_xyz",
  "status": "queued"
}
```

---

### 4. Translate Text
**POST** `/api/v1/translate`

**Request:**
```json
{
  "text": "Text to translate",
  "target_language": "Hindi"
}
```

**Supported Languages:** Hindi, Tamil, Telugu, Bengali, Marathi

---

### 5. Validate Content
**POST** `/api/v1/validate`

**Request:**
```json
{
  "text": "Educational content",
  "subject": "Science",
  "grade_level": 8
}
```

---

### 6. Generate Audio (TTS)
**POST** `/api/v1/tts`

**Request:**
```json
{
  "text": "Text to convert",
  "language": "Hindi"
}
```

---

## 💬 Q&A System Endpoints

### 1. Process Document
**POST** `/api/v1/qa/process`

**Request:**
```json
{
  "file_path": "data/uploads/abc123_doc.pdf"
}
```

**Response:**
```json
{
  "content_id": "123",
  "status": "processed",
  "chunks_created": 15
}
```

---

### 2. Ask Question
**POST** `/api/v1/qa/ask`

**Request:**
```json
{
  "content_id": "123",
  "question": "What is photosynthesis?"
}
```

**Response:**
```json
{
  "answer": "Photosynthesis is...",
  "sources": ["chunk_1", "chunk_5"],
  "confidence": 0.92
}
```

---

### 3. Get Chat History
**GET** `/api/v1/qa/history/{content_id}`

**Response:**
```json
{
  "content_id": "123",
  "history": [
    {
      "question": "What is photosynthesis?",
      "answer": "...",
      "timestamp": "2025-11-16T10:00:00Z"
    }
  ]
}
```

---

## 🔍 Task & Content Management

### 1. Get Task Status
**GET** `/api/v1/tasks/{task_id}`

**Response:**
```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "result": {
    "content_id": "123",
    "simplified_text": "...",
    "translations": {...}
  }
}
```

**Status Values:** `queued`, `processing`, `completed`, `failed`

---

### 2. Cancel Task
**DELETE** `/api/v1/tasks/{task_id}`

**Response:**
```json
{
  "message": "Task cancelled",
  "task_id": "task_abc123"
}
```

---

### 3. Get Content
**GET** `/api/v1/content/{content_id}`

**Response:**
```json
{
  "id": "123",
  "original_text": "...",
  "simplified_text": "...",
  "translations": {...},
  "audio_available": true
}
```

---

### 4. Stream Audio
**GET** `/api/v1/audio/{content_id}`

Returns audio file stream (WAV format).

---

### 5. Submit Feedback
**POST** `/api/v1/feedback`

**Request:**
```json
{
  "content_id": "123",
  "rating": 5,
  "comment": "Very helpful!"
}
```

---

## 🏥 Health Endpoints

### 1. Basic Health Check
**GET** `/health`

**Response:**
```json
{
  "status": "healthy"
}
```

---

### 2. Detailed Diagnostics
**GET** `/health/detailed`

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "celery": "running",
  "workers": 2,
  "uptime": "2h 15m"
}
```

---

## ⚠️ Error Responses

### Standard Format
```json
{
  "error": "ERROR_CODE",
  "message": "Human-readable error message",
  "timestamp": "2025-11-16T10:45:00Z"
}
```

### HTTP Status Codes
- `200 OK` - Success
- `201 Created` - Resource created
- `202 Accepted` - Task queued
- `400 Bad Request` - Invalid input
- `401 Unauthorized` - Authentication required
- `404 Not Found` - Resource not found
- `409 Conflict` - Duplicate resource
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

---

## 🚦 Rate Limiting

### Current Limits
- Upload: 1000 requests/minute
- Process: 1000 requests/minute
- Auth: 1000 requests/minute

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1700140800
```

---

## 📝 Example Usage

### cURL Examples

**Register:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123!","full_name":"John Doe"}'
```

**Upload File:**
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf"
```

**Process Content:**
```bash
curl -X POST http://localhost:8000/api/v1/process \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path":"data/uploads/abc123_document.pdf",
    "grade_level":8,
    "subject":"Science",
    "target_languages":["Hindi","Tamil"],
    "output_format":"both"
  }'
```

**Check Task:**
```bash
curl http://localhost:8000/api/v1/tasks/$TASK_ID \
  -H "Authorization: Bearer $TOKEN"
```

---

## 🔗 Useful Links

- **Interactive Docs:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc
- **Health Dashboard:** http://localhost:8000/health/detailed

---

## 👨‍💻 Created By

**K Dhiraj**  
📧 k.dhiraj.srihari@gmail.com  
💼 LinkedIn: [linkedin.com/in/k-dhiraj]
