# Section 2: High-Level Architecture Diagram

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.0.0 (Universal Mode)
**Last Updated:** December 5, 2025

---

## System Overview

The Shiksha Setu architecture is designed to handle the complexity of running multiple AI models while remaining responsive and resource-efficient. Every component is deliberately positioned to minimize latency and maximize throughput on consumer hardware.

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
│  │  │  • StudentProfileService • SafetyPipeline          • OCRService              │   │ │
│  │  │  • CurriculumValidation  • GradeLevelAdaptation    • UnifiedPipelineService  │   │ │
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
│  │  │  BGE-M3    │  │         │  │   Qwen2.5-3B   │  │         │  │  Semantic   │  │   │
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

*For detailed component implementations, see the Backend Architecture (Section 3) and Frontend Architecture (Section 4) documentation.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com
