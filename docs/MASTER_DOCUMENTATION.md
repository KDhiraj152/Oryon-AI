# Shiksha Setu v4.0 - Master Documentation

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.0.0 (Universal Mode)
**Last Updated:** December 5, 2025

---

## Overview

Shiksha Setu is an AI-powered universal education platform designed for India. The platform delivers personalized learning experiences across 10 Indian languages, serving students from diverse socioeconomic backgrounds with quality educational content.

**Core Capabilities:**
- Question answering with RAG-enhanced accuracy
- Real-time translation across 10 Indian languages
- Speech-to-text for voice queries
- Text-to-speech for audio responses
- Curriculum-aligned content delivery
- Adaptive learning paths

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 17 with pgvector extension
- Redis 7+
- 16GB RAM minimum (32GB recommended)

### Installation

```bash
# Clone and setup
git clone https://github.com/kdhiraj/shiksha-setu.git
cd shiksha-setu

# Run setup script
chmod +x setup.sh
./setup.sh

# Start services
./start.sh
```

### Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000/api/v2 |
| API Documentation | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SHIKSHA SETU v4.0                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   React UI  │  │   Voice UI  │  │  Mobile PWA │  │ Teacher UI │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
│         └─────────────────┴─────────────────┴──────────────┘        │
│                                │                                     │
│                    ┌───────────┴───────────┐                        │
│                    │      API Gateway      │                        │
│                    │   (FastAPI + Nginx)   │                        │
│                    └───────────┬───────────┘                        │
│                                │                                     │
│  ┌─────────────────────────────┴─────────────────────────────┐      │
│  │                    Service Layer                           │      │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │      │
│  │  │   RAG    │ │ Translate│ │   STT    │ │     TTS      │  │      │
│  │  │ Service  │ │ Service  │ │ Service  │ │   Service    │  │      │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │      │
│  └───────┴────────────┴────────────┴──────────────┴──────────┘      │
│                                │                                     │
│  ┌─────────────────────────────┴─────────────────────────────┐      │
│  │                    AI Model Layer                          │      │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐ │      │
│  │  │ Qwen2.5-3B   │ │   BGE-M3     │ │   IndicTrans2-1B   │ │      │
│  │  │  (INT4)      │ │  Embeddings  │ │    Translation     │ │      │
│  │  └──────────────┘ └──────────────┘ └────────────────────┘ │      │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐ │      │
│  │  │BGE-Reranker  │ │Whisper Turbo │ │  MMS-TTS/Edge-TTS  │ │      │
│  │  │    v2-M3     │ │     V3       │ │                    │ │      │
│  │  └──────────────┘ └──────────────┘ └────────────────────┘ │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                │                                     │
│  ┌─────────────────────────────┴─────────────────────────────┐      │
│  │                    Data Layer                              │      │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐ │      │
│  │  │ PostgreSQL   │ │    Redis     │ │   File Storage     │ │      │
│  │  │ + pgvector   │ │    Cache     │ │                    │ │      │
│  │  └──────────────┘ └──────────────┘ └────────────────────┘ │      │
│  └───────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- FastAPI v2 with async/await
- PostgreSQL 17 + pgvector 0.7
- Redis 7 for caching
- Python 3.11

**Frontend:**
- React 18 with TypeScript 5
- Vite 5 build system
- Zustand 4 state management
- Tailwind CSS + shadcn/ui

**AI Models:**
| Component | Model | Size |
|-----------|-------|------|
| LLM | Qwen2.5-3B-Instruct | 3B params (INT4) |
| Embeddings | BGE-M3 | 568M params |
| Reranker | BGE-Reranker-v2-M3 | 568M params |
| Translation | IndicTrans2-1B | 1B params |
| STT | Whisper V3 Turbo | 809M params |
| TTS | MMS-TTS / Edge-TTS | Variable |

---

## API Reference

### Authentication

```bash
# Login
POST /api/v2/auth/login
{
  "email": "user@example.com",
  "password": "password"
}

# Response
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Question Answering

```bash
# Ask question
POST /api/v2/qa/ask
Authorization: Bearer <token>
{
  "question": "What is photosynthesis?",
  "language": "hindi",
  "stream": false
}

# Response
{
  "answer": "प्रकाश संश्लेषण वह प्रक्रिया है...",
  "sources": [...],
  "confidence": 0.92
}
```

### Translation

```bash
# Translate text
POST /api/v2/translate
Authorization: Bearer <token>
{
  "text": "The mitochondria is the powerhouse of the cell",
  "source_lang": "english",
  "target_lang": "tamil"
}

# Response
{
  "translated_text": "மைட்டோகாண்ட்ரியா உயிரணுவின் ஆற்றல் மையம்",
  "source_lang": "english",
  "target_lang": "tamil"
}
```

### Health Check

```bash
# System health
GET /api/v2/health

# Response
{
  "status": "healthy",
  "version": "4.0.0",
  "models": {
    "llm": "loaded",
    "embedder": "loaded",
    "translator": "loaded"
  },
  "database": "connected",
  "cache": "connected"
}
```

---

## Configuration

### Environment Variables

```bash
# Application
APP_ENV=production
APP_VERSION=4.0.0
DEBUG=false

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/shiksha_setu
REDIS_URL=redis://localhost:6379

# AI Models
DEVICE=auto
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
EMBEDDING_MODEL=BAAI/bge-m3
TRANSLATION_MODEL=ai4bharat/indictrans2-en-indic-1B

# Security
SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

### Model Configuration

```python
# backend/core/config.py
class Settings(BaseSettings):
    # Model settings
    LLM_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    LLM_QUANTIZATION: str = "int4"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024

    # Device settings
    DEVICE: str = "auto"  # auto, cuda, mps, cpu
    MAX_MEMORY_GB: float = 16.0

    # Supported languages
    SUPPORTED_LANGUAGES: list = [
        "hindi", "tamil", "telugu", "bengali", "marathi",
        "gujarati", "kannada", "malayalam", "punjabi", "odia"
    ]
```

---

## Deployment

### Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f backend

# Scale workers
docker-compose up -d --scale backend=3
```

### Production Checklist

- [ ] Set `APP_ENV=production`
- [ ] Configure SSL/TLS certificates
- [ ] Set strong `SECRET_KEY`
- [ ] Enable rate limiting
- [ ] Configure log aggregation
- [ ] Set up monitoring dashboards
- [ ] Configure backup schedules
- [ ] Test failover procedures

---

## Performance

### Benchmarks

| Operation | Latency (p50) | Latency (p95) | Throughput |
|-----------|---------------|---------------|------------|
| Question answering | 450ms | 850ms | 50 req/s |
| Translation | 120ms | 250ms | 200 req/s |
| Speech-to-text | 1.2s | 2.5s | 20 req/s |
| Text-to-speech | 200ms | 400ms | 100 req/s |
| Vector search | 15ms | 35ms | 1000 req/s |

### Resource Requirements

| Configuration | RAM | GPU VRAM | Storage |
|---------------|-----|----------|---------|
| Minimum | 16GB | 8GB | 50GB |
| Recommended | 32GB | 16GB | 100GB |
| Production | 64GB | 24GB | 500GB |

---

## Documentation Index

| Section | File | Description |
|---------|------|-------------|
| 01 | `01_EXECUTIVE_SUMMARY.md` | Project overview and mission |
| 02 | `02_ARCHITECTURE_DIAGRAM.md` | Visual system architecture |
| 03 | `03_BACKEND_ARCHITECTURE.md` | Backend services and APIs |
| 04 | `04_FRONTEND_ARCHITECTURE.md` | React application structure |
| 05 | `05_DATA_FLOW.md` | Request/response flows |
| 06 | `06_API_DOCUMENTATION.md` | Complete API reference |
| 07 | `07_MODEL_PIPELINE.md` | AI model integration |
| 08 | `08_DEPLOYMENT.md` | Deployment procedures |
| 09 | `09_CODE_QUALITY.md` | Quality standards and testing |
| 10 | `10_FUTURE_IMPROVEMENTS.md` | Development roadmap |
| 11 | `11_CONTRIBUTION_SUMMARY.md` | Author contributions |

---

## Support

For questions or issues:

**K Dhiraj**
k.dhiraj.srihari@gmail.com

---

*Built for India. Built for learning.*
