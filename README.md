# Shiksha Setu

**Safe, Open AI for Education & Noble Purposes**

A local-first, unrestricted AI platform that empowers learning, research, creativity, and noble causes—while maintaining essential safety guardrails.

---

## Vision

Shiksha Setu is evolving beyond education into a **general-purpose AI** that:
- 🎓 **Educates** — STEM-aligned content, multilingual support, grade adaptation
- 🔬 **Researches** — Unrestricted knowledge exploration for academic work
- 🎨 **Creates** — Assists with writing, coding, analysis, and creative tasks
- 🌍 **Serves Noble Purposes** — Healthcare, accessibility, social good

### Philosophy

> **Safe without being restricted. Powerful without being harmful.**

We block only genuinely dangerous content (weapons, malware, real harm) while trusting users with good intent for everything else.

---

## Overview

Shiksha Setu is a production-grade AI platform that runs entirely locally on Apple Silicon, with no cloud dependencies. It simplifies content, translates to Indian languages, answers questions, and generates audio—all through a unified AI pipeline.

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Text Simplification** | Grade-level adaptation using Qwen3-8B (MLX 4-bit) |
| **Translation** | 10 Indian languages via IndicTrans2-1B |
| **OCR** | Document extraction with GOT-OCR2.0 (95%+ accuracy on Indian scripts) |
| **Validation** | NCERT curriculum alignment using Qwen3-8B (shared LLM, ≥80% threshold) |
| **Text-to-Speech** | Dual TTS: Edge TTS (online) + MMS-TTS (offline, 1100+ languages) |
| **Speech-to-Text** | Whisper Large V3 Turbo (8x faster, 99 languages) |
| **RAG Q&A** | Intelligent question answering with BGE-M3 embeddings |
| **Reranking** | Improved retrieval with BGE-Reranker-v2-M3 |
| **Universal File Upload** | Process any file: images, PDFs, audio, video, spreadsheets |
| **A/B Testing** | Experiment framework for content optimization |
| **Multi-Tenancy** | Organization-level isolation and management |
| **Learning Recommendations** | Personalized content suggestions |
| **Question Generation** | Auto-generate quizzes from content |
| **Teacher Evaluation** | Content review and approval workflows |

### Universal File Processing

Upload **any file type** and get intelligent AI processing:

| File Type | Extensions | AI Processing |
|-----------|-----------|---------------|
| **Audio** | mp3, wav, m4a, ogg, flac, aac, wma | Whisper V3 transcription |
| **Video** | mp4, webm, mov, avi, mkv | Audio extraction + STT |
| **Documents** | pdf (multi-page), docx | GOT-OCR2 + Tesseract OCR |
| **Images** | png, jpg, jpeg, tiff, bmp, webp, gif, heic | GOT-OCR2 text extraction |
| **Spreadsheets** | csv, xls, xlsx | Direct parsing + analysis |
| **Text** | txt, md, json, xml, yaml | Direct content extraction |

### Supported Languages

Hindi • Tamil • Telugu • Bengali • Marathi • Gujarati • Kannada • Malayalam • Punjabi • Odia

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Frontend (React + Vite)                     │
│              TypeScript • TailwindCSS • Shadcn/UI               │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Backend (FastAPI)                          │
│     REST API • JWT Auth • Rate Limiting • Multi-Tier Cache      │
└─────────────────────────────────────────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐
│   PostgreSQL    │    │   Multi-Tier     │    │  Unified Pipeline │
│ pgvector + HNSW │    │     Cache        │    │   (Optimized)     │
└─────────────────┘    │  L1: Memory      │    └───────────────────┘
                       │  L2: Redis       │              │
                       │  L3: SQLite      │              ▼
                       └──────────────────┘    ┌───────────────────┐
                                               │   Device Router   │
                                               │  GPU│MPS│ANE│CPU  │
                                               └───────────────────┘
                                                         │
                    ┌────────────────┬──────────────────┼──────────────────┐
                    ▼                ▼                  ▼                  ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │     MLX      │ │   CoreML     │ │     MPS      │ │   vLLM/HF    │
           │  (Apple M4)  │ │ (ANE 38TOPS) │ │   (Metal)    │ │   (CUDA)     │
           └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
                    │                │                │                │
                    ▼                ▼                ▼                ▼
           ┌──────────────────────────────────────────────────────────────┐
           │                        ML Models                             │
           │  Qwen3-8B • IndicTrans2 • GOT-OCR • BGE-M3 • BGE-Reranker      │
           │  Whisper V3 Turbo • Edge TTS • MMS-TTS                          │
           └──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18 • TypeScript 5 • Vite 5 • TailwindCSS • Shadcn/UI |
| **Backend** | FastAPI • SQLAlchemy 2.0 • Pydantic v2 • Celery |
| **Database** | PostgreSQL 17 • pgvector • HNSW indexes |
| **Cache** | Multi-Tier: L1 (LRU) → L2 (Redis) → L3 (SQLite) |
| **ML/AI** | PyTorch • MLX (Apple Silicon) • CoreML • Transformers • vLLM |
| **Inference** | DeviceRouter: MLX/CoreML/MPS/CUDA with auto-selection |
| **Resilience** | Circuit Breakers • Graceful Degradation |
| **Observability** | OpenTelemetry • Prometheus • Grafana • Sentry |
| **Infrastructure** | Docker • Kubernetes |

---

## Quick Start

### Prerequisites

- **Python 3.11** (recommended) — See [Python Version Note](#python-version-note) below
- Node.js 20+
- Redis 7+
- PostgreSQL 17+ (or Supabase)

### Setup

```bash
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd shiksha_setu
./setup.sh
```

The setup script:
- Creates Python virtual environment
- Installs backend dependencies
- Installs frontend dependencies
- Generates secure JWT secret
- Initializes database schema
- Creates required directories

### Run

```bash
./start.sh
```

Starts:
- Backend API (port 8000)
- AI Pipeline (7 models ready)
- Frontend (port 3000)

Access: http://localhost:3000

### Stop

```bash
./stop.sh
```

---

## Python Version Note

**Why Python 3.11?**

This project requires **Python 3.11** specifically (not newer versions) for optimal ML/AI stack compatibility:

| Reason | Explanation |
|--------|-------------|
| **Pre-built Wheels** | All ML packages (PyTorch, MLX, Transformers, etc.) have pre-built wheels for 3.11, avoiding compilation |
| **Proven Stability** | Python 3.11 is mature and thoroughly tested with production ML frameworks |
| **Package Support** | Some packages don't yet support Python 3.13+ (e.g., verovio requires compilation on 3.14) |
| **Performance** | Python 3.11 includes significant performance improvements (~25% faster than 3.10) |
| **Apple Silicon** | MLX and CoreML tools are optimized and tested for Python 3.11 |

**Tested Package Versions (Python 3.11):**
- PyTorch 2.9.1, Transformers 4.57.3, MLX 0.30.0
- Sentence-Transformers 3.4.1, FastAPI 0.123.2
- Edge-TTS 7.2.3, Verovio 5.6.0

**Installation (macOS):**
```bash
brew install python@3.11
```

---

## Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Chat Interface | http://localhost:3000/chat |
| Settings | http://localhost:3000/settings |
| Backend API (V2) | http://localhost:8000/api/v2 |
| Health Check | http://localhost:8000/api/v2/health |
| Hardware Status | http://localhost:8000/api/v2/hardware/status |
| Models Status | http://localhost:8000/api/v2/models/status |
| API Documentation | http://localhost:8000/docs |
| Prometheus Metrics | http://localhost:8000/metrics |

### V2 API Quick Reference

```bash
# Guest chat (no auth required)
curl -X POST http://localhost:8000/api/v2/chat/guest \
  -H "Content-Type: application/json" \
  -d '{"message": "What is photosynthesis?", "language": "hi", "grade_level": 5}'

# Streaming chat with conversation history (v2.3.1+)
curl -X POST http://localhost:8000/api/v2/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you explain more?",
    "history": [
      {"role": "user", "content": "What is AI?"},
      {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}
    ]
  }'

# Content simplification
curl -X POST http://localhost:8000/api/v2/content/simplify \
  -H "Content-Type: application/json" \
  -d '{"text": "Complex text here", "target_grade": 5}'
```

---

## Scripts

### Start/Stop

```bash
# Start all services
./start.sh                    # Full start with Docker
./start.sh --skip-docker      # Skip Docker (use existing containers)
./start.sh --quick            # Quick start (minimal checks)
./start.sh --monitoring       # Include Prometheus + Grafana

# Stop all services
./stop.sh                     # Graceful stop (keeps Docker containers)
./stop.sh --all               # Stop everything including Docker
./stop.sh --force             # Force kill immediately
./stop.sh --status            # Show optimization metrics before stopping
```

### Model Management

```bash
# Download ML models
./download_models.sh           # Download essential models
./download_models.sh --all     # Download all models
./download_models.sh --list    # List available models
./download_models.sh --check   # Check cached models
```

### Validation & Testing

```bash
# Run tests
source venv/bin/activate
pytest tests/                  # Full test suite
pytest tests/unit/             # Unit tests only
pytest tests/ --cov=backend    # With coverage

# Validation scripts
python scripts/validation/validate_setup.py
python scripts/validation/validate.py
```

---

## Project Structure

```
shiksha-setu/
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT License
├── pyproject.toml               # Python project config (Ruff, Pytest, MyPy, Coverage)
├── requirements.txt             # Python dependencies
├── requirements.lock.txt        # Pinned dependency versions
├── docker-compose.yml           # Docker orchestration (PostgreSQL + Redis)
├── alembic.ini                  # Database migration config
├── setup.sh                     # One-step project setup
├── start.sh                     # Start all services
├── stop.sh                      # Stop all services
├── download_models.sh           # Download ML models from HuggingFace
│
├── backend/                     # FastAPI application (Python package)
│   ├── database.py              # SQLAlchemy engine & session management
│   ├── api/                     # HTTP layer
│   │   ├── main.py              # FastAPI app, lifespan, startup
│   │   ├── deps.py              # Shared lazy-loaded singletons
│   │   ├── documentation.py     # OpenAPI customization
│   │   ├── metrics.py           # Prometheus /metrics endpoint
│   │   ├── middleware.py        # Exception handlers
│   │   ├── unified_middleware.py    # Consolidated middleware chain
│   │   ├── validation_middleware.py # Request validation
│   │   ├── version_middleware.py    # API versioning headers
│   │   └── routes/              # Domain-organized endpoints
│   │       ├── auth.py          # Authentication (register, login, refresh)
│   │       ├── chat.py          # Chat & streaming (SSE)
│   │       ├── content.py       # Content processing (simplify, translate, TTS, OCR)
│   │       ├── batch.py         # Batch processing
│   │       ├── health.py        # Health checks, system status, admin
│   │       └── agents.py        # Multi-agent system endpoints
│   │
│   ├── core/                    # Infrastructure & configuration
│   │   ├── config.py            # Application settings (env-backed)
│   │   ├── constants.py         # Application-wide constants
│   │   ├── types.py             # Canonical enums (ModelTier, ModelType, TaskType)
│   │   ├── exceptions.py        # Custom exceptions with retry decorators
│   │   ├── circuit_breaker.py   # Fault tolerance pattern
│   │   ├── security.py          # JWT, password hashing
│   │   ├── storage.py           # Redis/Memory storage backends
│   │   ├── policy.py            # Content policy engine
│   │   ├── model_config.py      # Hot-reloadable model configuration
│   │   ├── correlation.py       # Request correlation ID logging
│   │   ├── tracing.py           # OpenTelemetry distributed tracing
│   │   ├── hal.py               # Hardware abstraction layer
│   │   └── optimized/           # Apple Silicon M4 optimizations (lazy-loaded)
│   │       ├── device_router.py       # MPS/CUDA/CPU routing
│   │       ├── model_manager.py       # High-performance model lifecycle
│   │       ├── memory_coordinator.py  # Global memory management
│   │       ├── memory_pool.py         # Buffer pool management
│   │       ├── async_optimizer.py     # Async-first patterns & batching
│   │       ├── gpu_pipeline.py        # GPU queue pipelining
│   │       ├── core_affinity.py       # P/E core routing
│   │       ├── apple_silicon.py       # M4-specific optimizations
│   │       ├── rate_limiter.py        # Unified rate limiting
│   │       ├── quantization.py        # INT4/INT8 quantization
│   │       ├── benchmark.py           # Performance benchmarking
│   │       └── ...                    # (22 modules total)
│   │
│   ├── models/                  # SQLAlchemy ORM models
│   │   ├── auth.py              # User, APIKey, Token models
│   │   ├── chat.py              # Conversation, Message models
│   │   ├── content.py           # ProcessedContent, Translation, Audio
│   │   ├── progress.py          # StudentProgress, Quiz, Achievement
│   │   ├── rag.py               # DocumentChunk, Embedding, ChatHistory
│   │   └── student.py           # StudentProfile, LearningStyle
│   │
│   ├── schemas/                 # Pydantic request/response schemas
│   │   ├── auth.py              # Auth DTOs (UserCreate, Token, etc.)
│   │   ├── content.py           # Content DTOs (ProcessRequest, etc.)
│   │   └── qa.py                # Q&A DTOs (QAQueryRequest, etc.)
│   │
│   ├── services/                # Business logic layer
│   │   ├── ai_core/             # AI engine (intent, routing, safety, prompts)
│   │   ├── pipeline/            # Content processing pipeline & orchestration
│   │   ├── inference/           # ML backends (MLX, CoreML, unified engine)
│   │   ├── evaluation/          # Semantic accuracy evaluation & refinement
│   │   ├── translate/           # Translation (IndicTrans2 engine & service)
│   │   ├── tts/                 # Text-to-Speech (Edge TTS, MMS-TTS)
│   │   ├── validate/            # Curriculum validation (NCERT, CBSE)
│   │   ├── rag.py               # RAG Q&A with BGE-M3 embeddings
│   │   ├── ocr.py               # Document OCR (GOT-OCR2)
│   │   ├── simplifier.py        # Content simplification
│   │   ├── speech_generator.py  # Speech generation
│   │   ├── speech_processor.py  # Speech processing
│   │   ├── safety_pipeline.py   # 3-pass safety verification
│   │   ├── cultural_context.py  # Indian cultural context adaptation
│   │   ├── curriculum_validation.py  # Curriculum alignment
│   │   ├── grade_adaptation.py  # Grade-level content adaptation
│   │   ├── student_profile.py   # Student personalization
│   │   ├── review_queue.py      # Teacher review workflow
│   │   └── error_tracking.py    # Sentry integration
│   │
│   ├── cache/                   # Multi-tier caching
│   │   ├── multi_tier_cache.py  # L1 (LRU) → L2 (Redis) → L3 (SQLite)
│   │   ├── redis_cache.py       # Redis cache backend
│   │   ├── embedding_cache.py   # Embedding-specific cache
│   │   ├── response_cache.py    # Response cache
│   │   ├── kv_cache.py          # Key-value cache
│   │   └── fast_serializer.py   # msgpack serialization
│   │
│   ├── agents/                  # Multi-agent system
│   │   ├── base.py              # BaseAgent protocol & registry
│   │   ├── orchestrator.py      # Request routing & coordination
│   │   ├── model_execution.py   # ML model lifecycle & inference
│   │   ├── hardware_optimizer.py # Dynamic hardware tuning
│   │   ├── evaluation.py        # Quality measurement
│   │   ├── resource_monitor.py  # Memory, GPU, latency tracking
│   │   └── self_improvement.py  # Closed-loop optimization
│   │
│   ├── monitoring/              # Observability
│   │   ├── metrics.py           # Prometheus metrics
│   │   └── oom_alerts.py        # OOM detection & alerting
│   │
│   ├── tasks/                   # Celery background tasks
│   │   ├── celery_app.py        # Celery application
│   │   ├── celery_config.py     # Worker configuration
│   │   ├── embedding_tasks.py   # Embedding generation tasks
│   │   ├── ocr_tasks.py         # OCR processing tasks
│   │   ├── rag_tasks.py         # RAG pipeline tasks
│   │   ├── translate_tasks.py   # Translation tasks
│   │   └── simplify_tasks.py    # Simplification tasks
│   │
│   └── utils/                   # Shared utilities
│       ├── auth.py              # Auth helpers (get_current_user)
│       ├── logging.py           # Structured logging setup
│       ├── hashing.py           # Hashing utilities
│       ├── cancellation.py      # Task cancellation
│       └── memory_guard.py      # Memory guard utilities
│
├── frontend/                    # React + TypeScript + Vite
│   ├── package.json             # Node.js dependencies
│   ├── vite.config.ts           # Vite build config
│   ├── tsconfig.json            # TypeScript config
│   ├── tailwind.config.js       # TailwindCSS config
│   └── src/
│       ├── main.tsx             # App entry point
│       ├── App.tsx              # Root component with routing
│       ├── api/                 # Backend API client layer
│       │   ├── client.ts        # HTTP client with interceptors
│       │   ├── auth.ts          # Auth endpoints
│       │   ├── chat.ts          # Chat endpoints
│       │   ├── content.ts       # Content processing
│       │   ├── aiCore.ts        # AI engine endpoints
│       │   ├── audio.ts         # Audio endpoints
│       │   ├── conversations.ts # Conversation management
│       │   ├── progress.ts      # Student progress
│       │   ├── qa.ts            # Q&A endpoints
│       │   ├── system.ts        # System status
│       │   ├── profileReview.ts # Profile & review
│       │   ├── types.ts         # Shared API types
│       │   └── v2.ts            # V2 API helpers
│       ├── pages/               # Top-level route components
│       │   ├── LandingPage.tsx  # Landing page
│       │   ├── Auth.tsx         # Authentication page
│       │   ├── Chat.tsx         # Chat interface
│       │   └── Settings.tsx     # Settings page
│       ├── components/          # Reusable UI components
│       │   ├── chat/            # Chat UI (ChatMessage, ChatInput, Sidebar, etc.)
│       │   ├── landing/         # Landing page (OmLogo)
│       │   ├── layout/          # App layout (AppLayout)
│       │   ├── system/          # System status (SystemStatusCard)
│       │   ├── ui/              # Base UI primitives (Skeleton, Toast)
│       │   ├── ErrorBoundary.tsx
│       │   ├── LightRays.tsx
│       │   └── LogoLoop.tsx
│       ├── context/             # React Context providers
│       │   ├── SystemStatusContext.tsx
│       │   └── ThemeContext.tsx
│       ├── hooks/               # Custom React hooks
│       │   └── useChat.ts
│       ├── store/               # Zustand state management
│       │   └── index.ts         # Auth, chat, settings stores
│       ├── lib/                 # Utility libraries
│       │   └── accessibility.tsx
│       └── utils/               # Utility functions
│           └── secureTokens.ts  # XSS-safe token management
│
├── tests/                       # Test suite
│   ├── conftest.py              # Shared fixtures & test setup
│   ├── unit/                    # Unit tests (fast, isolated)
│   ├── integration/             # Integration tests (DB, services)
│   ├── e2e/                     # End-to-end pipeline tests
│   ├── performance/             # Benchmarks & load tests
│   ├── manual/                  # Manual testing endpoints
│   └── fixtures/                # Test data (policy configs)
│
├── alembic/                     # Database migrations
│   ├── env.py                   # Migration environment config
│   └── versions/                # Migration scripts (001–018)
│
├── scripts/                     # Development & operations scripts
│   ├── setup/                   # Setup scripts (DB init, model download, auth)
│   ├── deployment/              # Deployment scripts (start, backup, verify)
│   ├── testing/                 # Test runners & quality checks
│   ├── benchmarks/              # Performance benchmark scripts
│   ├── demo/                    # Demo scripts & data seeders
│   ├── validation/              # System validation scripts
│   └── utils/                   # Utility scripts (cleanup, status check)
│
├── infrastructure/              # DevOps & deployment configs
│   ├── docker/                  # Dockerfiles & compose overrides
│   ├── kubernetes/              # K8s manifests (base + overlays)
│   ├── monitoring/              # Prometheus, Grafana, Alertmanager configs
│   └── nginx/                   # Reverse proxy configuration
│
├── docs/                        # Project documentation
│   ├── 01-executive-summary.md
│   ├── 02-architecture.md
│   ├── 03-backend.md
│   ├── 04-frontend.md
│   ├── 05-api-reference.md
│   ├── 06-model-pipeline.md
│   ├── 07-deployment.md
│   ├── 08-code-quality.md
│   ├── 09-hardware-optimization.md
│   ├── 10-roadmap.md
│   └── 11-contributing.md
│
├── storage/                     # Runtime data storage
│   ├── audio/                   # Generated audio files
│   ├── cache/                   # SQLite cache databases
│   ├── captions/                # Caption files
│   ├── cultural_context/        # Indian cultural context data
│   ├── curriculum/              # NCERT/CBSE standards data
│   ├── models/                  # ML model cache
│   └── uploads/                 # User uploads
│
├── data/                        # Model cache & uploads (env-configurable)
│   ├── models/                  # ML model storage (MODEL_CACHE_DIR)
│   └── uploads/                 # User uploads (UPLOAD_DIR)
│
├── policy/                      # Content policy configuration
│   └── config.default.json      # Default policy settings
│
└── .github/                     # CI/CD
    └── workflows/               # GitHub Actions (ci.yml, build.yml)
```

---

## Environment Configuration

Key variables in `.env`:

```bash
# Application
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/shiksha_setu

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=<auto-generated>

# ML Models (2025 Optimal Stack)
DEVICE=auto                    # auto | cuda | mps | cpu
USE_QUANTIZATION=true

# Model IDs
SIMPLIFICATION_MODEL_ID=mlx-community/Qwen3-8B-4bit
TRANSLATION_MODEL_ID=ai4bharat/indictrans2-en-indic-1B
VALIDATION_MODEL_ID=mlx-community/Qwen3-8B-4bit
EMBEDDING_MODEL_ID=BAAI/bge-m3
RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3
TTS_MODEL_ID=facebook/mms-tts-hin
WHISPER_MODEL_ID=openai/whisper-large-v3-turbo

# TTS Configuration
EDGE_TTS_ENABLED=true          # Use Edge TTS as primary (online)
MMS_TTS_FALLBACK=true          # Use MMS-TTS as fallback (offline)
```

See `.env.example` for complete configuration.

---

## API Overview

### V2 API (Current - Recommended)

All endpoints are consolidated under `/api/v2/` with full hardware optimization.

#### Authentication
- `POST /api/v2/auth/register` — Create account
- `POST /api/v2/auth/login` — Get tokens
- `POST /api/v2/auth/refresh` — Refresh access token
- `GET /api/v2/auth/me` — Get current user

#### Chat
- `POST /api/v2/chat` — Authenticated chat
- `POST /api/v2/chat/stream` — Streaming chat (SSE)
- `POST /api/v2/chat/guest` — Guest chat (no auth)
- `GET /api/v2/chat/conversations` — List conversations
- `POST /api/v2/chat/conversations` — Create conversation
- `GET /api/v2/chat/conversations/{id}` — Get conversation
- `GET /api/v2/chat/conversations/{id}/messages` — Get messages
- `DELETE /api/v2/chat/conversations/{id}` — Delete conversation

#### Content Processing
- `POST /api/v2/content/process` — Full pipeline (simplify + translate + validate + TTS)
- `POST /api/v2/content/process/stream` — Full pipeline with streaming progress
- `POST /api/v2/content/simplify` — Simplify text (Qwen3-8B)
- `POST /api/v2/content/translate` — Translate (IndicTrans2)
- `POST /api/v2/content/tts` — Text-to-Speech (MMS-TTS/Edge TTS)
- `GET /api/v2/content/tts/voices` — List TTS voices

#### Speech-to-Text (Whisper V3 Turbo)
- `POST /api/v2/stt/transcribe` — Transcribe audio
- `GET /api/v2/stt/languages` — List supported languages

#### OCR (GOT-OCR2)
- `POST /api/v2/ocr/extract` — Extract text from images
- `GET /api/v2/ocr/capabilities` — Get OCR capabilities

#### Embeddings & Reranking (BGE-M3)
- `POST /api/v2/embeddings/generate` — Generate embeddings
- `POST /api/v2/embeddings/rerank` — Rerank documents

#### Q&A (RAG)
- `POST /api/v2/qa/process` — Process document for Q&A
- `POST /api/v2/qa/ask` — Ask questions

#### Progress & Quizzes
- `GET /api/v2/progress/stats` — User progress
- `POST /api/v2/progress/quiz/generate` — Generate quiz
- `POST /api/v2/progress/quiz/submit` — Submit answers

#### Embeddings
- `POST /api/v2/embeddings/generate` — Generate embeddings (BGE-M3)
- `POST /api/v2/embeddings/rerank` — Rerank documents (BGE-Reranker-v2-M3)
- `POST /api/v2/embed` — Generate embeddings (alternative)

#### Teacher Review
- `GET /api/v2/review/pending` — Get pending reviews
- `GET /api/v2/review/{response_id}` — Get flagged response
- `POST /api/v2/review/{response_id}/submit` — Submit review
- `GET /api/v2/review/stats` — Review statistics

#### Student Profile
- `GET /api/v2/profile/me` — Get student profile
- `PUT /api/v2/profile/me` — Update profile

#### AI Core
- `POST /api/v2/ai/explain` — Explain content
- `GET /api/v2/ai/prompts` — List prompts
- `POST /api/v2/ai/safety/check` — Safety check

#### Admin
- `POST /api/v2/admin/backup` — Create backup
- `GET /api/v2/admin/backups` — List backups

#### System
- `GET /api/v2/health` — Health check with device info
- `GET /api/v2/health/detailed` — Detailed health check
- `GET /api/v2/stats` — API statistics
- `GET /health` — Basic health check
- `GET /metrics` — Prometheus metrics

---

## Testing

```bash
# Activate environment
source venv/bin/activate

# All tests
pytest tests/

# Specific test categories
pytest tests/unit/           # Unit tests (fast)
pytest tests/integration/    # Integration tests (needs DB)
pytest tests/e2e/            # End-to-end tests
pytest tests/performance/    # Benchmarks

# With coverage
pytest tests/ --cov=backend --cov-report=html

# Frontend
cd frontend && npm run lint
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Redis connection failed | Start Redis: `redis-server` |
| Database connection error | Check `DATABASE_URL` in `.env` |
| Model loading slow | First run downloads models (~10GB) |
| CUDA out of memory | Set `USE_QUANTIZATION=true` |
| Port already in use | Run `./stop.sh` first |

---

## License

MIT License — see [LICENSE](LICENSE)

---

⸻

Created by: **K Dhiraj**
Email: k.dhiraj.srihari@gmail.com


