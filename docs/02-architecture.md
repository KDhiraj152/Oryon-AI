# Architecture & Data Flow

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## System Overview

The Oryon AI architecture is designed to handle the complexity of running multiple AI models while remaining responsive and resource-efficient. Every component is deliberately positioned to minimize latency and maximize throughput on consumer hardware.

The following diagram represents the complete data flow—from user interaction to final response delivery.

---

## Complete System Architecture

```
                                    ┌─────────────────────┐
                                    │    USER DEVICE      │
                                    │  (Browser/Mobile)   │
                                    └──────────┬──────────┘
                                               │
                                        HTTPS / WSS
                                    (TLS 1.3 Encrypted)
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              LOAD BALANCER & REVERSE PROXY                               │
│                                                                                          │
│     ┌─────────────────────────────────────────────────────────────────────────────┐     │
│     │                              NGINX (Production)                              │     │
│     │  • SSL Termination        • Rate Limiting           • Gzip Compression      │     │
│     │  • Static Asset Caching   • WebSocket Upgrade       • Health Check Routing  │     │
│     └─────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│     ┌─────────────────────────────────────────────────────────────────────────────┐     │
│     │                         UVICORN (Development Mode)                           │     │
│     │  • Hot Reload             • Direct ASGI Access      • Debug Logging          │     │
│     └─────────────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                    FRONTEND LAYER                                        │
│                              React 18 + Vite + TypeScript                                │
│                                                                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐                  │
│  │   REACT SPA        │  │   STATE LAYER      │  │   REAL-TIME I/O    │                  │
│  │                    │  │                    │  │                    │                  │
│  │  • Pages:          │  │  • Zustand Stores: │  │  • SSE Handler:    │                  │
│  │    - LandingPage   │  │    - useAuthStore  │  │    - Token stream  │                  │
│  │    - ChatInterface │  │    - useChatStore  │  │    - Auto-reconnect│                  │
│  │    - Settings      │  │    - useSettingsStore│ │                    │                  │
│  │    - Auth          │  │                    │  │  • Audio Processor:│                  │
│  │                    │  │  • Persist:        │  │    - Web Audio API │                  │
│  │  • Components:     │  │    - localStorage  │  │    - MediaRecorder │                  │
│  │    - ChatMessage   │  │    - Session sync  │  │    - Audio Playback│                  │
│  │    - AudioPlayer   │  │                    │  │                    │                  │
│  │    - FileUploader  │  │                    │  │                    │                  │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘                  │
│                                                                                          │
│                              Tailwind CSS + shadcn/ui Components                         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                       REST / JSON / SSE
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                     BACKEND LAYER                                        │
│                                    FastAPI (ASGI)                                        │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                REQUEST PIPELINE                                      │ │
│  │                                                                                      │ │
│  │  ┌───────────────┐    ┌───────────────────┐    ┌─────────────────────────────────┐  │ │
│  │  │  API GATEWAY  │───▶│  MIDDLEWARE CHAIN │───▶│     TASK ORCHESTRATOR           │  │ │
│  │  │               │    │                   │    │                                 │  │ │
│  │  │ • JWT Auth    │    │ • Request Logging │    │ • Priority Queue                │  │ │
│  │  │ • API Version │    │ • Rate Limiter    │    │   (High/Normal/Low)             │  │ │
│  │  │ • CORS        │    │ • Circuit Breaker │    │                                 │  │ │
│  │  │ • OpenAPI     │    │ • Age Consent     │    │ • Batch Processor               │  │ │
│  │  │               │    │ • Timing Header   │    │   (Groups similar requests)     │  │ │
│  │  └───────────────┘    └───────────────────┘    │                                 │  │ │
│  │                                                │ • Async Task Queue              │  │ │
│  │                                                │   (Background processing)       │  │ │
│  │                                                └─────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                               │                                          │
│                                               ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              SERVICE ORCHESTRATION                                   │ │
│  │                                                                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │                        BUSINESS LOGIC SERVICES                                │   │ │
│  │  │                                                                               │   │ │
│  │  │  • RAGService            • TranslationService      • EdgeTTSService          │   │ │
│  │  │  • UserProfileService • SafetyPipeline          • OCRService              │   │ │
│  │  │  • ContentValidation  • GradeLevelAdaptation    • UnifiedPipelineService  │   │ │
│  │  └──────────────────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                    AI CORE ENGINE                                        │
│                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          MEMORY & DEVICE COORDINATION                             │   │
│  │                                                                                   │   │
│  │  ┌─────────────────────┐ ◄───► ┌─────────────────────┐ ◄───► ┌────────────────┐  │   │
│  │  │  MEMORY COORDINATOR │       │   GPU SCHEDULER     │       │ MODEL REGISTRY │  │   │
│  │  │                     │       │                     │       │                │  │   │
│  │  │  • RAM Monitor      │       │  • Thermal Aware    │       │ • LRU Eviction │  │   │
│  │  │  • VRAM Budget      │       │  • Device Router    │       │ • Lazy Loading │  │   │
│  │  │  • OOM Prevention   │       │    (MPS/CUDA/CPU)   │       │ • Version Mgmt │  │   │
│  │  │  • Cache Eviction   │       │  • Batch Optimizer  │       │ • Health Check │  │   │
│  │  └─────────────────────┘       └─────────────────────┘       └────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                              │
│          ┌────────────────────────────────┼────────────────────────────────┐            │
│          ▼                                ▼                                ▼            │
│  ┌──────────────────┐         ┌──────────────────────┐         ┌───────────────────┐   │
│  │   RAG PIPELINE   │         │   INFERENCE ENGINE   │         │  SAFETY PIPELINE  │   │
│  │                  │         │                      │         │                   │   │
│  │  ┌────────────┐  │         │  ┌────────────────┐  │         │  ┌─────────────┐  │   │
│  │  │  BGE-M3    │  │         │  │    Qwen3-8B    │  │         │  │  Semantic   │  │   │
│  │  │  Embedder  │  │         │  │  (Reasoning)   │  │         │  │   Check     │  │   │
│  │  └────────────┘  │         │  └────────────────┘  │         │  └─────────────┘  │   │
│  │                  │         │                      │         │         │         │   │
│  │  ┌────────────┐  │         │  ┌────────────────┐  │         │         ▼         │   │
│  │  │   HNSW     │  │         │  │  IndicTrans2   │  │         │  ┌─────────────┐  │   │
│  │  │   Index    │  │         │  │ (Translation)  │  │         │  │  Logical    │  │   │
│  │  └────────────┘  │         │  └────────────────┘  │         │  │   Check     │  │   │
│  │                  │         │                      │         │  └─────────────┘  │   │
│  │  ┌────────────┐  │         │  ┌────────────────┐  │         │         │         │   │
│  │  │    BGE     │  │         │  │  Whisper V3    │  │         │         ▼         │   │
│  │  │  Reranker  │  │         │  │    (STT)       │  │         │  ┌─────────────┐  │   │
│  │  └────────────┘  │         │  └────────────────┘  │         │  │   Policy    │  │   │
│  │                  │         │                      │         │  │   Engine    │  │   │
│  │  ┌────────────┐  │         │  ┌────────────────┐  │         │  └─────────────┘  │   │
│  │  │  Semantic  │  │         │  │   MMS-TTS      │  │         │                   │   │
│  │  │ Validator  │  │         │  │ (Edge Fallback)│  │         │                   │   │
│  │  └────────────┘  │         │  └────────────────┘  │         │                   │   │
│  └──────────────────┘         └──────────────────────┘         └───────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                     DATA LAYER                                           │
│                                                                                          │
│  ┌────────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │     POSTGRESQL 17      │  │    REDIS 7 CACHE    │  │      FILE STORAGE           │   │
│  │                        │  │                     │  │                             │   │
│  │  • SQLAlchemy ORM      │  │  • Multi-Tier Cache │  │  • Audio Files              │   │
│  │  • pgvector Extension  │  │    - L1: In-memory  │  │  • Document Uploads         │   │
│  │  • HNSW Vector Index   │  │    - L2: Redis      │  │  • Model Weights            │   │
│  │  • Alembic Migrations  │  │    - L3: Disk       │  │  • Generated Content        │   │
│  │                        │  │  • Session Store    │  │                             │   │
│  │  Tables:               │  │  • Rate Limit       │  │  Directories:               │   │
│  │  • users               │  │    Counters         │  │  • storage/audio/           │   │
│  │  • conversations       │  │                     │  │  • storage/uploads/         │   │
│  │  • messages            │  │  Fast Serializer:   │  │  • storage/models/          │   │
│  │  • documents           │  │  • msgpack          │  │  • storage/cache/           │   │
│  │  • embeddings          │  │  • numpy arrays     │  │                             │   │
│  │  • feedback            │  │                     │  │                             │   │
│  └────────────────────────┘  └─────────────────────┘  └─────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### Frontend Layer

| Component | Responsibility |
|-----------|----------------|
| **React SPA** | Single-page application with client-side routing |
| **Zustand Stores** | Lightweight state management with persistence |
| **SSE Handler** | Server-Sent Events for streaming responses |
| **Audio Processor** | Web Audio API for recording and playback |

### Backend Layer

| Component | Responsibility |
|-----------|----------------|
| **API Gateway** | Authentication, versioning, CORS, OpenAPI |
| **Middleware Chain** | Logging, rate limiting, circuit breaking |
| **Task Orchestrator** | Priority queuing, batch processing |
| **Service Layer** | Business logic encapsulation |

### AI Core Engine

| Component | Responsibility |
|-----------|----------------|
| **Memory Coordinator** | Global memory budget management |
| **GPU Scheduler** | Device routing with thermal awareness |
| **Model Registry** | LRU-based model lifecycle management |
| **RAG Pipeline** | Embedding, retrieval, reranking |
| **Inference Engine** | LLM, translation, STT, TTS |
| **Safety Pipeline** | 3-pass content safety verification |

### Data Layer

| Component | Responsibility |
|-----------|----------------|
| **PostgreSQL + pgvector** | Structured data + vector similarity search |
| **Redis Cache** | Multi-tier caching with msgpack serialization |
| **File Storage** | Audio, documents, models, generated content |

---

## Data Flow Traces

This section traces the complete data flow through Oryon AI for the primary use cases: Question-Answering, Voice Interaction, and Document Processing. Each trace shows the exact path data takes from user input to final response.

---

### Text Question-Answering

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        TEXT Q&A DATA FLOW                                        │
└──────────────────────────────────────────────────────────────────────────────────┘

User types: "न्यूटन का पहला नियम क्या है?" (What is Newton's First Law? - in Hindi)

┌─────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Browser   │────▶│  Zustand    │────▶│  API Client     │────▶│   FastAPI   │
│   Input     │     │  Store      │     │  (Axios)        │     │   /qa/ask   │
└─────────────┘     └─────────────┘     └─────────────────┘     └──────┬──────┘
                                                                        │
                    ┌───────────────────────────────────────────────────┘
                    ▼
           ┌────────────────┐
           │   MIDDLEWARE   │
           │                │
           │ 1. JWT Auth    │
           │ 2. Rate Limit  │
           │ 3. Request ID  │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │ LANGUAGE       │
           │ DETECTION      │
           │                │
           │ Input: Hindi   │
           │ Script: Devanagari │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  TRANSLATION   │────▶│   IndicTrans2-1B   │
           │  (if needed)   │     │   Hindi → English  │
           └───────┬────────┘     └────────────────────┘
                   │
                   │ Translated: "What is Newton's First Law?"
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  EMBEDDING     │────▶│     BGE-M3         │
           │  GENERATION    │     │   1024-dim vector  │
           └───────┬────────┘     └────────────────────┘
                   │
                   │ Query vector: [0.023, -0.156, ...]
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  VECTOR        │────▶│   PostgreSQL +     │
           │  SEARCH        │     │   pgvector HNSW    │
           └───────┬────────┘     └────────────────────┘
                   │
                   │ Top-15 candidate chunks
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  RERANKING     │────▶│   BGE-Reranker     │
           │                │     │   Cross-Encoder    │
           └───────┬────────┘     └────────────────────┘
                   │
                   │ Top-5 reranked chunks with scores
                   ▼
           ┌────────────────┐
           │  CONTEXT       │
           │  ASSEMBLY      │
           │                │
           │  Build prompt  │
           │  with sources  │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  LLM           │────▶│   Qwen3-8B         │
           │  GENERATION    │     │   Streaming output │
           └───────┬────────┘     └────────────────────┘
                   │
                   │ English response (streaming)
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  TRANSLATION   │────▶│   IndicTrans2-1B   │
           │  (back to Hindi) │   │   English → Hindi  │
           └───────┬────────┘     └────────────────────┘
                   │
                   ▼
           ┌────────────────┐     ┌─────────────┐     ┌─────────────┐
           │  SSE STREAM    │────▶│  React SSE  │────▶│   Browser   │
           │  Response      │     │  Handler    │     │   Display   │
           └────────────────┘     └─────────────┘     └─────────────┘
```

#### Timing Breakdown (M4 Pro, 16GB)

| Stage | Latency | Cumulative |
|-------|---------|------------|
| Language Detection | 5ms | 5ms |
| Translation (Hindi→English) | 120ms | 125ms |
| Embedding Generation | 15ms | 140ms |
| Vector Search (HNSW) | 8ms | 148ms |
| Reranking (5 docs) | 45ms | 193ms |
| Context Assembly | 3ms | 196ms |
| LLM First Token | 180ms | 376ms |
| LLM Full Generation | 800ms | 1,176ms |
| Translation (English→Hindi) | 120ms | 1,296ms |
| **Total** | | **~1.3 seconds** |

---

### Voice Question-Answering

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       VOICE Q&A DATA FLOW                                        │
└──────────────────────────────────────────────────────────────────────────────────┘

User speaks: "गुरुत्वाकर्षण बल क्या है?" (What is gravitational force? - in Hindi)

┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Microphone │────▶│  MediaRecorder  │────▶│  Audio Blob     │
│  Input      │     │  (WebM/Opus)    │     │  (Binary)       │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                      │
                                              POST /voice/transcribe
                                                      │
                                                      ▼
                                             ┌────────────────┐
                                             │   FastAPI      │
                                             │   Audio Upload │
                                             └───────┬────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐
                                             │  AUDIO         │
                                             │  PREPROCESSING │
                                             │                │
                                             │  • Format: WAV │
                                             │  • Sample: 16kHz│
                                             │  • Channels: 1 │
                                             └───────┬────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐     ┌──────────────────┐
                                             │  SPEECH-TO-    │────▶│  Whisper V3      │
                                             │  TEXT          │     │  Turbo           │
                                             │                │     │  (Multilingual)  │
                                             │  Auto-detect:  │     └──────────────────┘
                                             │  Language: hi  │
                                             └───────┬────────┘
                                                     │
                                                     │ Transcript: "गुरुत्वाकर्षण बल क्या है?"
                                                     │ Detected Language: Hindi
                                                     │
                                                     ▼
                                         ┌────────────────────────┐
                                         │                        │
                                         │   RAG PIPELINE         │
                                         │   (Same as Text Q&A)   │
                                         │                        │
                                         │   Translation →        │
                                         │   Embedding →          │
                                         │   Retrieval →          │
                                         │   Reranking →          │
                                         │   Generation →         │
                                         │   Translation          │
                                         │                        │
                                         └───────────┬────────────┘
                                                     │
                                                     │ Hindi text response
                                                     ▼
                                             ┌────────────────┐     ┌──────────────────┐
                                             │  TEXT-TO-      │────▶│  MMS-TTS         │
                                             │  SPEECH        │     │  (Hindi voice)   │
                                             │                │     │                  │
                                             │  Streaming     │     │  OR Edge-TTS     │
                                             │  audio chunks  │     │  (fallback)      │
                                             └───────┬────────┘     └──────────────────┘
                                                     │
                                                     │ Audio stream (MP3)
                                                     ▼
                                             ┌────────────────┐     ┌─────────────────┐
                                             │  HTTP Response │────▶│  Web Audio API  │
                                             │  (audio/mpeg)  │     │  Playback       │
                                             └────────────────┘     └─────────────────┘
```

#### Timing Breakdown (M4 Pro, 16GB)

| Stage | Latency | Cumulative |
|-------|---------|------------|
| Audio Upload | 50ms | 50ms |
| Preprocessing | 20ms | 70ms |
| STT (Whisper V3) | 400ms | 470ms |
| RAG Pipeline | 1,200ms | 1,670ms |
| TTS Generation | 600ms | 2,270ms |
| Audio Streaming | 100ms | 2,370ms |
| **Total** | | **~2.4 seconds** |

---

### Document Upload and Processing

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     DOCUMENT PROCESSING DATA FLOW                                │
└──────────────────────────────────────────────────────────────────────────────────┘

User uploads: "content_domain_Physics_Class11_Chapter3.pdf" (2.5MB)

┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  File       │────▶│  FormData       │────▶│  FastAPI        │
│  Picker     │     │  Upload         │     │  /documents/    │
└─────────────┘     └─────────────────┘     └───────┬─────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐
                                             │  FILE          │
                                             │  VALIDATION    │
                                             │                │
                                             │  • Type: PDF   │
                                             │  • Size: 2.5MB │
                                             │  • Virus scan  │
                                             └───────┬────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐
                                             │  STORAGE       │
                                             │                │
                                             │  Save to:      │
                                             │  storage/      │
                                             │  uploads/      │
                                             └───────┬────────┘
                                                     │
                                          ┌──────────┴──────────┐
                                          ▼                     ▼
                                   ┌────────────┐        ┌────────────┐
                                   │  PDF TEXT  │        │  OCR       │
                                   │  EXTRACTION│        │  (if scan) │
                                   │            │        │            │
                                   │  PyMuPDF   │        │  OCRService│
                                   └─────┬──────┘        └─────┬──────┘
                                         │                     │
                                         └──────────┬──────────┘
                                                    │
                                                    │ Raw text (50 pages)
                                                    ▼
                                             ┌────────────────┐
                                             │  TEXT          │
                                             │  CHUNKING      │
                                             │                │
                                             │  • Chunk: 512  │
                                             │  • Overlap: 50 │
                                             │  • Total: 245  │
                                             └───────┬────────┘
                                                     │
                                                     │ 245 text chunks
                                                     ▼
                                             ┌────────────────┐     ┌──────────────────┐
                                             │  BATCH         │────▶│  BGE-M3          │
                                             │  EMBEDDING     │     │  batch_size=32   │
                                             │                │     │                  │
                                             │  8 batches     │     │  Memory-coordinated│
                                             └───────┬────────┘     └──────────────────┘
                                                     │
                                                     │ 245 × 1024-dim vectors
                                                     ▼
                                             ┌────────────────┐     ┌──────────────────┐
                                             │  VECTOR        │────▶│  PostgreSQL      │
                                             │  STORAGE       │     │  pgvector        │
                                             │                │     │                  │
                                             │  Bulk insert   │     │  HNSW index      │
                                             │  with metadata │     │  update          │
                                             └───────┬────────┘     └──────────────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐
                                             │  DATABASE      │
                                             │  RECORD        │
                                             │                │
                                             │  documents:    │
                                             │  • id          │
                                             │  • filename    │
                                             │  • chunk_count │
                                             │  • created_at  │
                                             └───────┬────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐     ┌─────────────────┐
                                             │  JSON Response │────▶│  UI Update      │
                                             │  {success, id} │     │  Document list  │
                                             └────────────────┘     └─────────────────┘
```

#### Timing Breakdown (M4 Pro, 16GB)

| Stage | Latency | Cumulative |
|-------|---------|------------|
| Upload & Validation | 200ms | 200ms |
| Text Extraction | 800ms | 1,000ms |
| Chunking | 50ms | 1,050ms |
| Embedding (245 chunks) | 1,800ms | 2,850ms |
| Vector Storage | 400ms | 3,250ms |
| Index Update | 150ms | 3,400ms |
| **Total** | | **~3.4 seconds** |

---

### Streaming Response (SSE)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     SSE STREAMING DATA FLOW                                      │
└──────────────────────────────────────────────────────────────────────────────────┘

LLM generates token-by-token:

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Qwen3-8B        │────▶│  Token Buffer   │────▶│  SSE Encoder    │
│  generate()     │     │  (5 tokens)     │     │  event: message │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
Token stream: "Newton" → "'s" → " First" → " Law" → ...   │
                                                          ▼
                                                 ┌────────────────┐
                                                 │  HTTP Stream   │
                                                 │                │
                                                 │  data: Newton  │
                                                 │  data: 's      │
                                                 │  data:  First  │
                                                 │  data:  Law    │
                                                 │  ...           │
                                                 │  data: [DONE]  │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  EventSource   │
                                                 │  (Browser)     │
                                                 │                │
                                                 │  onmessage()   │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  Zustand       │
                                                 │  appendChunk() │
                                                 │                │
                                                 │  Real-time UI  │
                                                 │  update        │
                                                 └────────────────┘
```

---

### Error Handling

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     ERROR HANDLING DATA FLOW                                     │
└──────────────────────────────────────────────────────────────────────────────────┘

Error occurs during LLM inference:

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  LLM Inference  │──X──│  CUDA OOM       │────▶│  Circuit        │
│  Exception      │     │  Error          │     │  Breaker        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                 Records failure
                                                 (5 failures = trip)
                                                          │
                                                          ▼
                                                 ┌────────────────┐
                                                 │  Memory        │
                                                 │  Coordinator   │
                                                 │                │
                                                 │  Evict models  │
                                                 │  Clear cache   │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  Exception     │
                                                 │  Handler       │
                                                 │                │
                                                 │  Format error  │
                                                 │  for client    │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  HTTP 503      │
                                                 │                │
                                                 │  {             │
                                                 │    error:      │
                                                 │    "model_     │
                                                 │     overload", │
                                                 │    retry_after:│
                                                 │    30          │
                                                 │  }             │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  React Error   │
                                                 │  Boundary      │
                                                 │                │
                                                 │  Show retry    │
                                                 │  message       │
                                                 └────────────────┘
```

---

### Multi-Tier Cache

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-TIER CACHE DATA FLOW                                   │
└──────────────────────────────────────────────────────────────────────────────────┘

Embedding lookup request:

┌─────────────────┐
│  Cache.get()    │
│  key: "hash123" │
└────────┬────────┘
         │
         ▼
┌────────────────────┐     ┌─────────────────┐
│  L1: Memory        │────▶│  HIT?           │──── YES ──▶ Return (1ms)
│  (LRU Dict)        │     │                 │
└────────────────────┘     └────────┬────────┘
                                    │ MISS
                                    ▼
                           ┌────────────────────┐     ┌─────────────────┐
                           │  L2: Redis         │────▶│  HIT?           │──── YES ──▶ Return (5ms)
                           │  (msgpack)         │     │                 │            + Populate L1
                           └────────────────────┘     └────────┬────────┘
                                                               │ MISS
                                                               ▼
                                                      ┌────────────────────┐
                                                      │  L3: Disk          │
                                                      │  (msgpack files)   │
                                                      └────────┬───────────┘
                                                               │
                                                      ┌────────┴────────┐
                                                      ▼                 ▼
                                               HIT (20ms)         MISS (Compute)
                                               Populate L1+L2     Generate embedding
                                                                  Populate all tiers
```

---

## Data Flow Summary

```
User Input → Frontend → API Gateway → Middleware → Service Layer
                                                        ↓
                                                  AI Core Engine
                                                        ↓
                                    Memory Coordinator ←→ GPU Scheduler
                                                        ↓
                                              Model Inference
                                                        ↓
                                                  Data Layer
                                                        ↓
                                              Response Assembly
                                                        ↓
                                    SSE Stream → Frontend → User
```

---

*For detailed component implementations, see Backend (03) and Frontend (04) documentation.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com
