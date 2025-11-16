# 📚 ShikshaSetu Developer Guide

**AI-Powered Multilingual Education Platform**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- PostgreSQL 16
- Redis

### Start the System
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Start backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 3. Start Celery worker (new terminal)
celery -A src.tasks.celery_app worker --loglevel=info
```

**Access:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

## 🏗️ Project Structure

```
shiksha_setu/
├── src/
│   ├── core/                 # Configuration, security, exceptions
│   ├── api/                  # FastAPI routes & middleware
│   │   ├── routes/           # Modular route handlers
│   │   ├── main.py           # Main application
│   │   └── middleware.py     # Security headers, logging
│   ├── models.py             # Database models
│   ├── database.py           # DB connection & session
│   ├── tasks/                # Celery background tasks
│   ├── services/             # Business logic
│   └── utils/                # Helpers & utilities
│
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
│
├── scripts/                  # Utility scripts
├── docs/                     # Documentation
├── frontend/                 # React application
└── alembic/                  # Database migrations
```

---

## ✨ Features

### Core Capabilities
- **Text Simplification** - AI-powered simplification for grades 5-12
- **Multilingual Translation** - Hindi, Tamil, Telugu, Bengali, Marathi
- **Speech Synthesis** - Natural audio generation
- **Content Validation** - Curriculum alignment checking
- **Document Q&A** - RAG-powered question answering

### Technical Features
- **Modular Architecture** - Clear separation of concerns (core, api, services)
- **JWT Authentication** - Secure token-based auth with refresh tokens
- **Async Task Processing** - Celery for background ML operations
- **Database Optimization** - Composite indexes for fast queries
- **Security Middleware** - Headers, request timing, error handling
- **Centralized Logging** - Rotating file handlers with structured logs

---

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/shiksha_setu

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=your-secret-key
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Quick test
./scripts/quick_test.sh
```

---

## 📊 API Overview

### Authentication
- `POST /api/v1/auth/register` - Register user
- `POST /api/v1/auth/login` - Login
- `POST /api/v1/auth/refresh` - Refresh token
- `GET /api/v1/auth/me` - Get user profile

### Content Processing
- `POST /api/v1/upload` - Upload file
- `POST /api/v1/simplify` - Simplify text
- `POST /api/v1/translate` - Translate text
- `POST /api/v1/validate` - Validate content
- `POST /api/v1/tts` - Generate audio

### Q&A System
- `POST /api/v1/qa/process` - Process document
- `POST /api/v1/qa/ask` - Ask question
- `GET /api/v1/qa/history/{id}` - Get chat history

### Monitoring
- `GET /health` - Basic health check
- `GET /health/detailed` - System diagnostics

Full API documentation: [API.md](API.md)

---

## 🔒 Security

### Implemented Protections
- ✅ Input sanitization (SQL injection, XSS, path traversal)
- ✅ File upload validation (type, size, magic bytes)
- ✅ Rate limiting (Redis or in-memory)
- ✅ Secure password hashing
- ✅ Environment variable validation
- ✅ CORS configuration
- ✅ Content Security Policy headers

### Security Features
- Input validation (XSS, SQL injection protection)
- Security headers (CSP, X-Frame-Options, etc.)
- Rate limiting (1000/min, 10000/hour)
- JWT token-based authentication

---

## 🚀 Deployment

```bash
# Docker deployment
docker-compose up -d

# Production checklist
- Update JWT_SECRET_KEY in .env
- Configure DATABASE_URL for production
- Enable Redis for caching (optional)
- Set up SSL/HTTPS
```

---

## 🐛 Troubleshooting

### Database Connection Error
```bash
# Check PostgreSQL status
brew services list

# Restart if needed
brew services restart postgresql@16

# Verify connection
psql -U postgres -d education_content
```

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Model Download Issues
```bash
# Manual model download
python scripts/download_models.py
```

### Frontend Build Errors
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

---

## 📦 Dependencies

### Backend (Python)
- FastAPI 0.104.1 - Modern API framework
- PyTorch 2.6.0 - ML framework
- Transformers 4.35.2 - HuggingFace models
- SQLAlchemy 2.0.35 - Database ORM
- Redis 5.0.1 - Caching layer
- Psutil 5.9.6 - System monitoring

### Frontend (Node.js)
- React 18.2.0 - UI framework
- Vite 5.0.7 - Build tool
- TailwindCSS 3.3.6 - Styling
- React Router 6.20.1 - Routing

---

## 📚 Additional Documentation

- [API Reference](API.md) - Complete endpoint documentation
- [Changelog](../CHANGELOG.md) - Version history

---

👨‍💻 Created By

K Dhiraj  
📧 k.dhiraj.srihari@gmail.com  
💼 LinkedIn: [linkedin.com/in/k-dhiraj]

 

