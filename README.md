<p align="center">
  <h1 align="center">शिक्षा सेतु — Shiksha Setu</h1>
  <p align="center"><strong>AI-Powered Education for Every Student in India</strong></p>
  <p align="center">
    <em>6 AI models. 10 languages. Zero cloud. Runs on a laptop.</em>
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#why-this-exists">Why</a> •
  <a href="#what-it-does">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#performance">Performance</a> •
  <a href="docs/">Documentation</a>
</p>

---

## Why This Exists

**800 million students in India lack access to quality AI tutoring in their own language.**

Every major EdTech platform requires internet, charges subscriptions, and speaks English first. A student in rural Tamil Nadu gets a worse education than one in Mumbai — not because they're less capable, but because the tools don't work for them.

Shiksha Setu fixes this with three architectural constraints:

| Problem | How We Solve It |
|---------|:-----------------|
| **Language exclusion** | Native support for **10 Indian languages** via IndicTrans2 — not translation-as-afterthought, but multilingual from the ground up |
| **Internet dependency** | **Fully offline** after a one-time ~10GB model download. No APIs. No subscriptions. No metered usage. |
| **Privacy concerns** | **Zero data leaves the device.** No telemetry, no analytics, no cloud calls. Students explore freely. |

> A school in a remote village runs the same AI stack as elite urban institutions.

---

## What It Does

### Core AI Pipeline

| Capability | Model | What Happens |
|:-----------|:------|:-------------|
| **Question Answering** | Qwen3-8B (MLX 4-bit) | RAG-enhanced answers aligned to NCERT curriculum |
| **Translation** | IndicTrans2-1B | Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia |
| **Voice Input** | Whisper V3 Turbo | Speak in any language, get AI responses — 8x faster than Whisper V3 |
| **Voice Output** | MMS-TTS + Edge-TTS | Natural speech synthesis with automatic failover |
| **Smart Search** | BGE-M3 + BGE-Reranker-v2-M3 | Multilingual semantic search with cross-encoder reranking |
| **Document Processing** | GOT-OCR2 + PyMuPDF | Upload PDFs, images, spreadsheets — AI extracts and understands |

### Beyond Tutoring

- **Adaptive Learning** — Personalized content based on grade level and learning profile
- **Question Generation** — Auto-generate quizzes from any uploaded material
- **Teacher Dashboard** — Content review and approval workflows
- **A/B Testing Framework** — Experiment with content optimization strategies
- **Multi-Agent System** — Specialized agents for hardware optimization, quality evaluation, and self-improvement
- **Multi-Tenancy** — Organization-level isolation for schools and institutions

### Universal File Upload

Drop **any file** and get intelligent processing:

```
Audio  (mp3, wav, m4a, flac, ogg)     → Whisper V3 transcription
Video  (mp4, webm, mov, avi, mkv)     → Audio extraction + STT
Docs   (pdf, docx)                     → OCR + text extraction
Images (png, jpg, tiff, webp, heic)   → GOT-OCR2 text extraction
Data   (csv, xlsx, json, xml, yaml)   → Direct parsing + analysis
```

---

## Performance

Benchmarked on Apple Silicon M4 Pro (16GB unified memory):

| Metric | Result |
|:-------|:-------|
| LLM Inference | **50 tokens/sec** (Qwen3-8B, INT4) |
| Embedding Throughput | **348 texts/sec** (BGE-M3) |
| Text-to-Speech | **31x realtime** (MMS-TTS) |
| Speech-to-Text | **2x realtime** (Whisper V3 Turbo) |
| Reranking Latency | **2.6ms/document** (BGE-Reranker-v2-M3) |
| Q&A Latency (p50) | **450ms** |
| Translation Latency | **120ms** |
| Memory Efficiency | **75% reduction** vs FP16 (INT4 quantization) |

**Voice-to-voice end-to-end** (speak a question in Hindi → hear the answer): **under 4 seconds.**

### vs. Cloud Solutions

| | Cloud AI Tutors | Shiksha Setu |
|:--|:----------------|:-------------|
| **Privacy** | Data sent to foreign servers | All processing on-device |
| **Offline** | Requires internet | Full offline after setup |
| **Languages** | English-first | Native multilingual |
| **Cost** | ₹500-2000/month subscription | Free. Forever. |
| **Curriculum** | Generic responses | NCERT/CBSE aligned |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         SHIKSHA SETU v4.1                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   React 18 + TypeScript + Vite + Tailwind + shadcn/ui                   │
│   ─────────────────────────────────────────────────                      │
│              SSE Streaming • Zustand State • Web Audio API               │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   FastAPI (async) + JWT Auth + Rate Limiting + Circuit Breakers          │
│   ──────────────────────────────────────────────────────────             │
│   REST API │ Multi-Tier Cache (Memory→Redis→SQLite) │ OpenTelemetry     │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│   │  Qwen3-8B  │  │IndicTrans2 │  │   BGE-M3   │  │Whisper V3  │       │
│   │   (LLM)    │  │(Translate) │  │(Embeddings)│  │  (STT)     │       │
│   └────────────┘  └────────────┘  └────────────┘  └────────────┘       │
│   ┌────────────┐  ┌────────────┐                                        │
│   │BGE-Reranker│  │  MMS-TTS   │    Device Router: MLX│MPS│CUDA│CPU    │
│   │ (Search)   │  │  (Voice)   │    Memory Coordinator │ GPU Scheduler  │
│   └────────────┘  └────────────┘                                        │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   PostgreSQL 17 + pgvector (HNSW)  │  Redis 7  │  File Storage          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technologies |
|:------|:-------------|
| **Frontend** | React 18, TypeScript 5, Vite 5, Tailwind CSS, shadcn/ui, Zustand |
| **Backend** | FastAPI, SQLAlchemy 2.0, Pydantic v2, Celery, Alembic |
| **Database** | PostgreSQL 17 + pgvector (HNSW indexes) |
| **Cache** | L1 Memory (LRU) → L2 Redis → L3 SQLite |
| **AI/ML** | PyTorch, MLX (Apple Silicon), CoreML, Transformers, vLLM |
| **Observability** | OpenTelemetry, Prometheus, Grafana, Sentry |
| **Infrastructure** | Docker Compose, Kubernetes, Nginx |

### Hardware Optimization

The system auto-detects and optimizes for the available hardware:

- **Apple Silicon (M1–M4)** — MLX backend with unified memory coordination, thermal monitoring, P/E core routing
- **NVIDIA CUDA** — vLLM/HuggingFace with INT4/INT8 quantization
- **CPU-only** — Graceful fallback with optimized batch sizes

A custom **Memory Coordinator** orchestrates 6+ concurrent AI models in shared memory with LRU eviction, preventing OOM while maximizing throughput.

---

## Quick Start

### Prerequisites

- Python 3.11 ([why 3.11?](#python-311-requirement))
- Node.js 20+
- PostgreSQL 17+ with pgvector
- Redis 7+
- ~30GB disk space (models + data)

### Install & Run

```bash
# Clone
git clone https://github.com/KDhiraj152/Shiksha-setu.git
cd Shiksha-setu

# Setup (creates venv, installs deps, downloads models, runs migrations)
./setup.sh

# Start everything
./start.sh
```

| Service | URL |
|:--------|:----|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000/api/v2 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/api/v2/health |

### Try It

```bash
# Ask a question in Hindi (no auth required)
curl -X POST http://localhost:8000/api/v2/chat/guest \
  -H "Content-Type: application/json" \
  -d '{"message": "प्रकाश संश्लेषण क्या है?", "language": "hi", "grade_level": 8}'

# Simplify text for a 5th grader
curl -X POST http://localhost:8000/api/v2/content/simplify \
  -H "Content-Type: application/json" \
  -d '{"text": "Mitochondria are membrane-bound organelles...", "target_grade": 5}'

# Streaming chat with SSE
curl -N -X POST http://localhost:8000/api/v2/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain Newton'"'"'s laws", "language": "en"}'
```

### Stop

```bash
./stop.sh
```

---

## API Overview

All endpoints under `/api/v2/`. Full reference in [docs/05-api-reference.md](docs/05-api-reference.md).

| Category | Key Endpoints |
|:---------|:-------------|
| **Auth** | `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh` |
| **Chat** | `POST /chat`, `POST /chat/stream` (SSE), `POST /chat/guest` |
| **Content** | `POST /content/process` (full pipeline), `/simplify`, `/translate`, `/tts` |
| **Speech** | `POST /stt/transcribe`, `POST /content/tts` |
| **Q&A (RAG)** | `POST /qa/process` (ingest document), `POST /qa/ask` |
| **Search** | `POST /embeddings/generate`, `POST /embeddings/rerank` |
| **System** | `GET /health`, `GET /health/detailed`, `GET /metrics` |

---

## Project Structure

```
Shiksha-setu/
├── backend/                 # FastAPI application
│   ├── api/                 #   Routes, middleware, metrics
│   ├── agents/              #   Multi-agent system (7 specialized agents)
│   ├── core/                #   Config, security, circuit breakers
│   │   └── optimized/       #   Apple Silicon optimizations (22 modules)
│   ├── services/            #   AI pipeline, RAG, translation, TTS, safety
│   │   ├── ai_core/         #     Intent routing, prompts, safety
│   │   ├── inference/       #     MLX, CoreML, unified engine
│   │   ├── pipeline/        #     Content orchestration
│   │   └── translate/       #     IndicTrans2 integration
│   ├── models/              #   SQLAlchemy ORM models
│   ├── cache/               #   Multi-tier caching (L1/L2/L3)
│   └── monitoring/          #   Prometheus, OOM alerts
├── frontend/                # React + TypeScript + Vite
│   └── src/
│       ├── api/             #   Backend API client layer
│       ├── pages/           #   Landing, Auth, Chat, Settings
│       ├── components/      #   Chat UI, layout, system status
│       └── store/           #   Zustand state management
├── docs/                    # 11 documentation files
├── tests/                   # Unit, integration, e2e, performance
├── alembic/                 # 18 database migrations
├── infrastructure/          # Docker, Kubernetes, Nginx, monitoring
├── scripts/                 # Setup, deployment, testing, benchmarks
└── storage/                 # Audio, cache, models, uploads
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Core
ENVIRONMENT=development
DATABASE_URL=postgresql://postgres:password@localhost:5432/shiksha_setu
REDIS_URL=redis://localhost:6379/0

# Device (auto-detected if not set)
DEVICE=auto                              # auto | cuda | mps | cpu

# Models (2025 Optimal Stack)
SIMPLIFICATION_MODEL_ID=mlx-community/Qwen3-8B-4bit
TRANSLATION_MODEL_ID=ai4bharat/indictrans2-en-indic-1B
EMBEDDING_MODEL_ID=BAAI/bge-m3
RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3
WHISPER_MODEL_ID=openai/whisper-large-v3-turbo
```

See [.env.example](.env.example) for all options.

---

## Testing

```bash
source venv/bin/activate

pytest tests/                         # Full suite
pytest tests/unit/                    # Unit tests (fast)
pytest tests/integration/             # Integration (needs DB)
pytest tests/e2e/                     # End-to-end pipelines
pytest tests/ --cov=backend           # With coverage report
```

---

## Python 3.11 Requirement

This project requires **Python 3.11** specifically — not newer — for full ML stack compatibility:

- All ML packages (PyTorch, MLX, Transformers) ship pre-built wheels for 3.11
- MLX and CoreML tools are optimized and tested for 3.11 on Apple Silicon
- Some dependencies don't yet support Python 3.13+ without compilation

```bash
brew install python@3.11    # macOS
```

---

## Documentation

Detailed documentation in [`docs/`](docs/):

| Doc | Content |
|:----|:--------|
| [Executive Summary](docs/01-executive-summary.md) | Project vision, market gap, strategic positioning |
| [Architecture](docs/02-architecture.md) | System diagrams, data flow traces, component responsibilities |
| [Backend](docs/03-backend.md) | FastAPI services, directory structure, middleware chain |
| [Frontend](docs/04-frontend.md) | React architecture, state management, component design |
| [API Reference](docs/05-api-reference.md) | Complete endpoint documentation with examples |
| [Model Pipeline](docs/06-model-pipeline.md) | AI model integration, benchmarks, configurations |
| [Deployment](docs/07-deployment.md) | Docker, Kubernetes, production checklist |
| [Code Quality](docs/08-code-quality.md) | Linting, testing, security scanning standards |
| [Hardware Optimization](docs/09-hardware-optimization.md) | Apple Silicon, CUDA, CPU optimization guide |
| [Roadmap](docs/10-roadmap.md) | Development plan through 2026 |
| [Contributing](docs/11-contributing.md) | Project history, key decisions, contributions |

---

## Troubleshooting

| Issue | Fix |
|:------|:----|
| Redis connection failed | `redis-server` or `brew services start redis` |
| Database connection error | Check `DATABASE_URL` in `.env`, ensure PostgreSQL is running |
| Models loading slowly | First run downloads ~10GB — subsequent starts are instant |
| CUDA out of memory | Set `USE_QUANTIZATION=true` in `.env` |
| Port already in use | `./stop.sh` then `./start.sh` |
| MPS errors on macOS | Ensure PyTorch 2.x with MPS support: `pip install torch>=2.0` |

---

## License

[MIT](LICENSE) — use it, modify it, deploy it.

---

<p align="center">
  <strong>Built for India. Built for learning. Built to run anywhere.</strong>
  <br><br>
  <sub>Created by <strong>K Dhiraj</strong> · <a href="mailto:k.dhiraj.srihari@gmail.com">k.dhiraj.srihari@gmail.com</a> · <a href="https://github.com/KDhiraj152">GitHub</a> · <a href="https://linkedin.com/in/kdhiraj">LinkedIn</a></sub>
</p>
