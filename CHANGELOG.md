# Shiksha Setu - Changelog

All notable changes to this project will be documented in this file.

---

## [2.7.0] - 2025-12-04

### PROJECT X: Autonomous Codebase Transformation

Comprehensive automated codebase transformation with static analysis, architecture refactoring, and test compatibility fixes.

#### Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | ENV PREP | ✅ Complete |
| Phase 1 | DISCOVERY | ✅ 77,863 lines indexed |
| Phase 2 | STATIC ANALYSIS | ✅ 59 lint fixes |
| Phase 3 | ARCHITECTURE REFACTOR | ✅ All imports fixed |
| Phase 4 | HARDWARE OPTIMIZATION | ✅ Verified working |
| Phase 5 | ML PIPELINE | ✅ Verified working |
| Phase 6 | FRONTEND | ✅ TS/ESLint clean |
| Phase 7 | DEVOPS | ✅ Verified working |
| Phase 8 | BENCHMARKING | ✅ Baselines established |
| Phase 9-10 | FINALIZATION | ✅ Complete |

#### Performance Benchmarks (M4 Apple Silicon)

| Metric | Value | Notes |
|--------|-------|-------|
| GPU TFLOPS | 3.72 | FP16 2048x2048 GEMM |
| SIMD Throughput | 54.7M/sec | Cosine similarity |
| Memory | 16GB | Unified memory detected |
| Import Latency | 1.6s | Full FastAPI stack |
| Device Detection | 18.7ms | M4 correctly identified |

#### Architecture Fixes

- **simplify_simplifier.py**: Fixed relative imports → absolute backend.* imports
- **model_collaboration.py**: Redirected imports from missing `collaboration` package
- **version_middleware.py**: Created new API versioning module
- **orchestrator.py**: Added `ModelCollaborator.get_metrics()` with all required fields
- **test_refinement_integration.py**: Fixed patching to use `_get_refinement_pipeline`
- **test_services.py**: Updated `CurriculumValidationService` tests
- **test_all_features.py**: Fixed syntax errors (extra parentheses)

#### Frontend Fixes

- **ChatMessage.tsx**: Fixed syntax error (orphaned brace)
- **chat.ts**: Fixed `while(true)` ESLint no-constant-condition error
- **client.ts**: Fixed `while(true)` ESLint error
- **chatUtils.ts**: Fixed two `while(true)` ESLint errors

#### Test Results

- **165 tests passing**
- **4 tests skipped** (require optional dependencies)
- **1 xpassed** (bcrypt compatibility)

---

## [2.6.0] - 2025-12-03

### Frontend-Backend Integration & Production Polish

Complete frontend-backend integration with real-time system status, improved scripts, and bug fixes.

#### Performance Achievements

| Metric | Value | Notes |
|--------|-------|-------|
| Backend Benchmarks | **864% improvement** | After dead code removal |
| Embeddings | **348 texts/sec** | BGE-M3 with MLX |
| LLM Inference | **50 tokens/sec** | Qwen2.5-3B |
| TTS | **31x realtime** | MMS-TTS on Apple Silicon |
| STT | **2x realtime** | Whisper V3 Turbo |
| Reranking | **2.6ms/doc** | BGE-Reranker-v2-M3 |

#### New Features

**Real-Time System Status**
- Added `SystemStatusContext` for global hardware/model status
- `useSystemStatus` hook for components
- Header connection indicator (green/yellow/red dot)
- Hardware info display (chip name, memory, GPU cores)

**Improved Chat Experience**
- Fixed blank screen bug when sending messages
- Support for both JSON and SSE response formats
- `parseJSONResponse()` for guest endpoint (JSON)
- `parseSSEStream()` for authenticated streaming (SSE)
- Automatic format detection based on Content-Type

**Unified Scripts (v3.3)**
- Port unified to 3000 (was inconsistent between 3000/3002)
- Extended backend timeout (15s → 45s for model loading)
- Improved process management with nohup
- Better error handling and status display
- Session stats in stop script

#### Files Modified

**Frontend:**
- `frontend/src/api/system.ts` - Updated types to match backend response
- `frontend/src/context/SystemStatusContext.tsx` - Added normalization helpers
- `frontend/src/components/system/SystemStatusCard.tsx` - Fixed type usage
- `frontend/src/components/chat/Header.tsx` - Added connection status indicator
- `frontend/src/lib/chatUtils.ts` - Added `parseJSONResponse()` function
- `frontend/src/pages/Chat.tsx` - Fixed JSON vs SSE detection
- `frontend/vite-env.d.ts` - Added Vite environment types

**Backend:**
- `backend/api/routes/v2_api.py` - Fixed FormattedResponse attribute access

**Scripts:**
- `start.sh` (v3.3) - Unified port 3000, extended timeouts
- `stop.sh` (v3.3) - Unified port 3000, improved cleanup

#### Bug Fixes

- **Chat Blank Screen**: Fixed issue where chat went blank after sending message
  - Root cause: Frontend used SSE parser for JSON responses
  - Solution: Detect Content-Type and use appropriate parser

- **API Type Mismatch**: Frontend types didn't match backend response
  - Updated `HardwareStatus` and `ModelsStatus` interfaces
  - Added normalization functions for API responses

- **FormattedResponse Attributes**: Fixed `.sources` → `.metadata.sources`

#### API Endpoints (V2)

All endpoints consolidated under `/api/v2/*`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/health` | GET | Health check with device info |
| `/api/v2/hardware/status` | GET | Hardware & optimization info |
| `/api/v2/models/status` | GET | Model loading status |
| `/api/v2/chat/guest` | POST | Guest chat (JSON response) |
| `/api/v2/chat/stream` | POST | Authenticated chat (SSE) |
| `/api/v2/content/simplify` | POST | Text simplification |
| `/api/v2/content/translate` | POST | Translation |
| `/api/v2/content/tts` | POST | Text-to-speech |
| `/api/v2/content/ocr` | POST | OCR extraction |
| `/api/v2/profile/me` | GET/PUT | Student profile |

#### TypeScript Improvements

- Removed all unused imports and variables
- Clean production build with no warnings
- Proper Vite environment type definitions

---

## [2.5.0] - 2025-12-02

### M4 Hardware Optimization - 5-Phase Architecture

Complete Apple Silicon M4 optimization with benchmark-driven implementation.

#### Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| I/O Operations | Sequential | Async parallel | **19.4x** speedup |
| Serialization | Pickle only | Type-optimized | **2.2ms** cycle |
| GPU Task Queue | Single queue | Priority queues | **0.3μs** overhead |
| Core Routing | OS default | P/E affinity | **0.1μs** lookup |
| Buffer Reuse | 0% | Pool-based | **94.3%** reuse |

#### New Optimization Modules

**Phase 1: Async-First Architecture** (`async_optimizer.py`)
- `AsyncTaskRunner` with semaphore-based concurrency control
- `gather_with_concurrency()` for parallel I/O
- Replaces threading (0.71x slower due to GIL) with async (19x faster)

**Phase 2: Fast Serialization** (`fast_serializer.py`)
- Type-optimized serialization paths
- msgpack for dicts, struct for primitives, numpy fast path
- Integrated with L2/L3 cache tiers

**Phase 3: GPU Queue Pipelining** (`gpu_pipeline.py`)
- `GPUPipelineScheduler` with priority queues
- Per-task command queues (LLM, embedding, TTS, validation)
- Async stage processing with preload support

**Phase 4: Core Affinity** (`core_affinity.py`)
- `CoreAffinityManager` for M4 P-core/E-core routing
- `TaskQoS` enum: USER_INTERACTIVE, UTILITY, BACKGROUND
- P-cores (4): inference, GPU compute, audio synthesis
- E-cores (6): I/O, logging, cache ops, monitoring

**Phase 5: Memory Pool Management** (`memory_pool.py`)
- `SizeClassAllocator` with 10 size classes (64B - 16MB)
- `TensorPool` for pre-allocated inference tensors
- `MemoryMappedWeights` for shared model weights
- `UnifiedMemoryPool` singleton for 16GB budget

#### Files Added

```
backend/core/optimized/
├── async_optimizer.py    # Phase 1
├── gpu_pipeline.py       # Phase 3
├── core_affinity.py      # Phase 4
└── memory_pool.py        # Phase 5

backend/cache/unified/
└── fast_serializer.py    # Phase 2
```

#### Files Removed (Cleanup)

- `backend/core/_archive/` - Unused archived code
- `backend/core/performance.py` - Migrated to `optimized/performance.py`

#### Documentation Updates

- `README.md` - Updated project structure with optimization modules
- `docs/BACKEND.md` - Added M4 Optimization section with usage examples

#### Tests

All 199 tests passing after optimization integration.

---

## [2.4.0] - 2025-12-02

### Python 3.11 Standardization & Environment Cleanup

#### Breaking Change: Python 3.11 Required

**Why This Change:**
- Python 3.11 is now the **required** version (not 3.11+)
- Ensures all ML packages have pre-built wheels (no compilation needed)
- Proven stability with PyTorch, Transformers, MLX, and other ML frameworks
- Some packages (verovio, certain transformers versions) don't fully support Python 3.13+

**Benefits:**
- ✅ Faster installation (pre-built wheels for all packages)
- ✅ Guaranteed compatibility across ML stack
- ✅ No build tools or compilation required
- ✅ Consistent environment across development machines

#### Updated Package Versions (requirements.txt)

All packages updated with version bounds for Python 3.11 compatibility:

| Package | Version | Notes |
|---------|---------|-------|
| PyTorch | 2.9.1 | Full Python 3.11 support |
| Transformers | 4.57.3 | Latest stable |
| MLX | 0.30.0 | Apple Silicon optimized |
| MLX-LM | 0.28.3 | Language model inference |
| Sentence-Transformers | 3.4.1 | Embedding generation |
| FastAPI | 0.123.2 | REST API framework |
| Pydantic | 2.12.5 | Data validation |
| SQLAlchemy | 2.0.44 | ORM |
| Verovio | 5.6.0 | Music notation (GOT-OCR2) |
| Edge-TTS | 7.2.3 | Text-to-speech |

#### Environment Cleanup

- Removed legacy `.venv` directory (Python 3.13.5)
- Standardized on `venv` directory for virtual environment
- Updated `setup.sh` to enforce Python 3.11 and clean up old environments

#### Updated Files

- `requirements.txt` - Version bounds for Python 3.11
- `setup.sh` - Python 3.11 requirement, v3.1 with smarter venv handling
- `README.md` - Prerequisites and Python Version Note section
- `docs/BACKEND.md` - Updated tech stack versions

#### Installation

```bash
# Install Python 3.11 (macOS)
brew install python@3.11

# Run setup (will use Python 3.11 automatically)
./setup.sh

# Or use the dedicated script
./scripts/setup_python311.sh
```

---

## [2.3.1] - 2025-12-02

### Chat History Support & Dependency Fixes

#### Backend: Chat Schema Enhancement

**v2_api.py - ChatMessage Schema Updated:**
- Added `history` field to accept conversation history
- New `HistoryMessage` model for structured history entries
- All chat endpoints now support multi-turn conversations

```python
class HistoryMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str   # Message content

class ChatMessage(BaseModel):
    message: str
    language: str = "en"
    subject: str = "General"
    grade_level: int = 8
    conversation_id: Optional[str] = None
    history: Optional[List[HistoryMessage]] = None  # NEW
    stream: bool = False
```

**Endpoints Updated:**
- `/api/v2/chat` - Now passes history to LLM
- `/api/v2/chat/stream` - SSE streaming with history context
- `/api/v2/chat/guest` - Guest chat with history support

#### Dependency: verovio for GOT-OCR2

**Added to requirements.txt:**
- `verovio>=5.6.0` - Music score rendering library required by GOT-OCR2
- Enables proper document processing including music notation
- Without this, OCR falls back to Tesseract for certain documents

**Installation:**
```bash
pip install verovio
```

#### Bug Fixes

- Fixed 422 Validation Error on chat endpoints when frontend sends history
- Fixed GOT-OCR2 "No module named 'verovio'" error
- Frontend-backend schema alignment for chat requests

---

## [2.3.0] - 2025-12-02

### Universal File Upload & Processing

Complete overhaul of file handling to support all file types with intelligent AI processing.

#### Frontend: Universal File Support

**ChatInput.tsx**:
- Changed file accept from limited types to `accept="*/*"` for ALL file types
- Added color-coded file icons:
  - 🔵 Images (blue)
  - 🟣 Audio (purple)
  - 🔴 Video (red)
  - 🟢 Spreadsheets (green)
  - ⚫ Documents (default)
- Updated tooltip: "Attach files: Images, PDFs, Documents, Audio, Video, Spreadsheets"

**chatUtils.ts** - New File Processing Functions:

| Function | File Types | Processing Method |
|----------|-----------|-------------------|
| `processAudioFile()` | mp3, wav, m4a, ogg, flac, aac, wma | Whisper V3 STT transcription |
| `processVideoFile()` | mp4, webm, mov, avi, mkv, m4v, wmv | Audio extraction + STT |
| `processSpreadsheetFile()` | csv, xls, xlsx | Direct text parsing |
| `processDocumentWithOCR()` | pdf, docx, images | GOT-OCR2 extraction |
| `processTextFile()` | txt, md, json, xml, yaml | Direct file read |

**New File Type Detection Utilities**:
```typescript
const AUDIO_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma'];
const VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v', '.wmv'];
const SPREADSHEET_EXTENSIONS = ['.csv', '.xls', '.xlsx'];
const OCR_EXTENSIONS = ['.pdf', '.docx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp', '.gif', '.heic'];
```

#### Backend: Enhanced File Processing APIs

**STT Endpoint** (`/api/v2/stt/transcribe`):
- Now accepts video files (mp4, webm, mov, avi, mkv)
- Automatic audio extraction from video
- Returns: text, language, confidence, processing_time_ms

**OCR Endpoint** (`/api/v2/ocr/extract`):
- Added HEIC, GIF, TIFF format support
- Multi-page PDF processing via PDFProcessor
- Formula and table detection

---

## [2.2.0] - 2025-12-02

### Backend 10/10 Improvements

Production-hardening improvements for enterprise-grade reliability.

#### New Core Modules

**Circuit Breakers** (`backend/core/circuit_breaker.py`):
- Resilience pattern for external service calls
- States: CLOSED → OPEN → HALF_OPEN → CLOSED
- Pre-configured for: database, redis, ml_model, external_api

```python
from backend.core.circuit_breaker import circuit_breaker

@circuit_breaker("redis")
async def get_cached_value(key: str):
    return await redis.get(key)
```

**OpenTelemetry Tracing** (`backend/core/tracing.py`):
- Distributed tracing for request flow through pipeline stages
- NoOp fallback when OTEL not installed
- `@trace_span` decorator for easy function tracing

```python
from backend.core.tracing import trace_span

@trace_span("process_content")
async def process_content(text: str):
    ...
```

**Validation Middleware** (`backend/api/validation_middleware.py`):
- Consistent validation error format across all endpoints
- Parses Pydantic errors into structured responses
- Field-level error details with constraints

**API Version Middleware** (`backend/api/version_middleware.py`):
- `X-API-Version: 2.0.0` on all responses
- `Sunset` header for deprecated v1 API
- Migration support with deprecation warnings

#### New Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health/models` | ML model availability status |
| `/health/circuits` | Circuit breaker states |
| `/health/dependencies` | All dependency checks (DB, Redis, ML, disk, memory) |

#### Device Router Enhancements

- Added `unified_memory_gb` alias property
- Added `has_mlx` alias property
- Added `get_optimal_backends()` method

#### Unit Tests

New test file: `tests/unit/test_core_components.py`
- 29 tests covering all new components
- Circuit breaker behavior tests
- Tracing NoOp tests
- Validation middleware tests
- API versioning tests
- Performance overhead tests

---

## [2.1.0] - 2025-12-02

### Full Model Integration & Frontend V2 Migration

Complete integration of all AI models into the UnifiedPipelineService and migration of frontend to V2 API.

#### Backend: Full Model Integration

**UnifiedPipelineService** (`backend/services/pipeline/unified_pipeline.py`) now uses ALL specialized models:

| Method | Model | Purpose |
|--------|-------|---------|
| `_simplify()` | Qwen2.5-3B-Instruct | Text simplification (MLX) |
| `_translate()` | IndicTrans2-1B | Translation (10 Indian languages) |
| `_validate()` | ValidationModule + BERT | NCERT curriculum validation |
| `_generate_audio()` | MMS-TTS / Edge TTS | Text-to-speech |
| `embed()` | BGE-M3 | Semantic embeddings for RAG |
| `rerank()` | BGE-Reranker-v2-M3 | Document reranking |
| `ocr()` | GOT-OCR2 | Image text extraction |
| `transcribe()` | Whisper V3 Turbo | Speech-to-text |
| `chat()` | Qwen2.5-3B-Instruct | Conversational AI |

**New V2 API Endpoints:**

| Endpoint | Method | Model |
|----------|--------|-------|
| `/api/v2/stt/transcribe` | POST | Whisper V3 Turbo |
| `/api/v2/stt/languages` | GET | - |
| `/api/v2/ocr/extract` | POST | GOT-OCR2 |
| `/api/v2/ocr/capabilities` | GET | - |
| `/api/v2/embeddings/generate` | POST | BGE-M3 |
| `/api/v2/embeddings/rerank` | POST | BGE-Reranker-v2-M3 |

**Key Changes:**
- `_translate()` now uses IndicTrans2 via TranslationEngine with LLM fallback
- `_validate()` now uses ValidationModule with BERTClient for semantic scoring
- All services are lazy-loaded for minimal startup time
- Added `asyncio.to_thread()` wrappers for sync model calls

#### Frontend: V2 API Migration

**API Client Updated** (`frontend/src/api/index.ts`):
- Changed `API_BASE` from `/api/v1` to `/api/v2`
- Added new API modules: `ocr`, `embeddings`, `stt`, `tts`

**New Frontend API Modules:**
```typescript
export const ocr;        // extractText(), getCapabilities()
export const embeddings; // generate(), rerank()
export const stt;        // transcribe(), getLanguages()
export const tts;        // synthesize(), getVoices()
```

**ChatInput Updated** (`frontend/src/components/chat/ChatInput.tsx`):
- Voice input now uses V2 STT API (`sttApi.transcribe()`)
- Uses Whisper V3 Turbo for transcription

**Chat Utils Updated** (`frontend/src/lib/chatUtils.ts`):
- Image uploads now processed with OCR (GOT-OCR2)
- Automatic text extraction from images in chat

**Updated Paths:**
- `Chat.tsx`: Guest fallback uses `/api/v2/chat/guest`
- `chatUtils.ts`: `getChatEndpoint()` returns V2 paths

#### Documentation Updated
- `README.md`: Updated API overview with V2 endpoints
- `docs/BACKEND.md`: Added STT, OCR, Embeddings endpoint documentation
- `docs/FRONTEND.md`: Rewrote for React (was incorrectly showing SvelteKit)
- `docs/ai_pipeline.md`: Already up-to-date

---

## [2.0.0] - 2025-12-02

### BREAKING: Complete V2 API Migration

Migrated from fragmented v1 API (109 routes across 15 files) to consolidated optimized V2 API.

#### Deleted V1 Files (BREAKING CHANGES)
- `backend/api/routes/auth.py` → Moved to `/api/v2/auth/*`
- `backend/api/routes/chat.py` → Moved to `/api/v2/chat/*`
- `backend/api/routes/admin.py` → Moved to `/api/v2/admin/*`
- `backend/api/routes/ai_core.py` → Moved to `/api/v2/ai/*`
- `backend/api/routes/audio_upload.py` → Moved to `/api/v2/audio/*`
- `backend/api/routes/experiments.py` → Removed (use `/api/v2/admin/experiments`)
- `backend/api/routes/optimized.py` → Consolidated into v2_api.py
- `backend/api/routes/progress.py` → Moved to `/api/v2/progress/*`
- `backend/api/routes/qa.py` → Moved to `/api/v2/ai/*`
- `backend/api/routes/review.py` → Removed (use admin endpoints)
- `backend/api/routes/streaming.py` → Moved to `/api/v2/chat/stream`
- `backend/api/routes/content/` → Entire directory removed, consolidated to `/api/v2/content/*`
- `backend/api/endpoints/` → Entire directory removed

#### New Consolidated V2 API (`backend/api/routes/v2_api.py`)

| Endpoint Group | Path | Description |
|---------------|------|-------------|
| Authentication | `/api/v2/auth/*` | Register, login, logout, refresh, me |
| Chat | `/api/v2/chat/*` | Guest chat, streaming, TTS, conversations |
| Content | `/api/v2/content/*` | Process, simplify, translate, TTS |
| Progress | `/api/v2/progress/*` | Stats, sessions, quizzes |
| Admin | `/api/v2/admin/*` | Backup, restore, system management |
| AI Core | `/api/v2/ai/*` | Explain, prompts, safety, sandbox |
| Health | `/health`, `/api/v2/health` | Basic and detailed health checks |

#### Fixed: MLX Backend for mlx_lm v0.28.3
- Updated `backend/services/inference/mlx_backend.py` for new MLX API
- Changed deprecated `temp=` parameter to new `sampler=make_sampler()` API
- Affects both `_generate_sync()` and `generate_stream()` methods

```python
# Old (broken with mlx_lm 0.28.3):
response = generate(model, tokenizer, prompt, temp=0.7, top_p=0.9)

# New (fixed):
from mlx_lm.sample_utils import make_sampler
sampler = make_sampler(temp=0.7, top_p=0.9)
response = generate(model, tokenizer, prompt, max_tokens=512, sampler=sampler)
```

#### API Version Bump
- Backend version: 2.0.0
- All endpoints now at `/api/v2/*`
- Health endpoints remain at `/health` for load balancer compatibility

#### Migration Guide

**Before (v1):**
```bash
curl -X POST http://localhost:8000/api/v1/chat/guest \
  -d '{"message": "Hello", "language": "en"}'
```

**After (v2):**
```bash
curl -X POST http://localhost:8000/api/v2/chat/guest \
  -d '{"message": "Hello", "language": "en", "grade_level": 5}'
```

#### Testing
- All v2 endpoints tested and verified working
- English, Hindi, Telugu chat tested successfully
- Streaming endpoint working
- Auth endpoints working
- Content simplification working

---

## [1.3.4] - 2025-12-01

### Semantic Accuracy Refinement Pipeline - Target 8.2+

#### New: Iterative Refinement for Semantic Accuracy
Added `SemanticRefinementPipeline` (`backend/services/evaluation/refinement_pipeline.py`) that enables iterative content improvement to achieve semantic accuracy ≥8.2.

**Key Features**:
- **Iterative Refinement Loop**: Up to 3 iterations with feedback-driven improvement
- **Task-Aware Evaluation Weights**: Different weight distributions for simplification, translation, and summarization
- **Multi-Dimensional Scoring**: Factual accuracy, educational clarity, semantic preservation, language appropriateness
- **Automatic Regeneration**: Re-generates content with specific feedback when score < 8.2

**Evaluation Dimensions by Task**:

| Task | FACTUAL_ACCURACY | EDUCATIONAL_CLARITY | SEMANTIC_PRESERVATION | LANGUAGE_APPROPRIATENESS |
|------|-----------------|--------------------|-----------------------|-------------------------|
| Simplification | 0.35 | 0.35 | 0.15 | 0.15 |
| Translation | 0.40 | 0.10 | 0.40 | 0.10 |
| Summarization | 0.30 | 0.20 | 0.35 | 0.15 |

**Refinement Prompts** (per dimension):
- `low_FACTUAL_ACCURACY`: Preserve all key facts without introducing errors
- `low_EDUCATIONAL_CLARITY`: Make explanations clearer for target grade level
- `low_SEMANTIC_PRESERVATION`: Keep more of the original meaning
- `low_LANGUAGE_APPROPRIATENESS`: Adjust vocabulary for target audience

#### Updated: TextSimplifier with Refinement
`backend/services/simplify/simplifier.py` now integrates refinement:

```python
# Initialize with refinement (default)
simplifier = TextSimplifier(
    enable_refinement=True,        # Enable iterative refinement
    target_semantic_score=8.2      # Target score
)

# Results include refinement metrics
result = await simplifier.simplify_text(content, grade_level, subject)
print(result.semantic_score)        # 8.5
print(result.refinement_iterations) # 1
print(result.dimension_scores)      # {'FACTUAL_ACCURACY': 8.8, ...}
```

**New `SimplifiedText` Fields**:
- `semantic_score: float` — Final evaluated semantic accuracy
- `refinement_iterations: int` — Number of iterations used
- `dimension_scores: Dict[str, float]` — Per-dimension scores

#### Updated: TranslationEngine with Refinement
`backend/services/translate/engine.py` also supports refinement:

```python
engine = TranslationEngine(
    enable_refinement=True,
    target_semantic_score=8.2
)

result = engine.translate(text, target_language, subject)
# result.semantic_score, result.refinement_iterations, etc.
```

**New `TranslatedText` Fields**:
- `refinement_iterations: int`
- `dimension_scores: Dict[str, float]`
- `target_reached: bool`

#### Technical Details

**Refinement Pipeline Flow**:
```
1. Generate initial output (single-pass)
2. Evaluate with SemanticAccuracyEvaluator
3. If score < 8.2:
   a. Identify lowest-scoring dimension
   b. Build refinement prompt with feedback
   c. Re-generate with refined prompt
   d. Repeat until target reached or max_iterations
4. Return best result with full metrics
```

**Configuration Options** (`RefinementConfig`):
- `target_score: float = 8.2` — Minimum acceptable score
- `max_iterations: int = 3` — Maximum refinement attempts
- `min_acceptable: float = 6.5` — Floor score (abort if below)
- `improvement_threshold: float = 0.15` — Min improvement to continue

**Fallback Behavior**:
- If refinement pipeline fails, returns single-pass output with error in metadata
- If `REFINEMENT_AVAILABLE = False` (import fails), uses single-pass only

#### New: Test Suite
`tests/unit/test_refinement_integration.py`:
- Mock tests for simplifier/translator integration
- Unit tests for RefinementConfig, RefinementTask
- Benchmark contracts for iteration count and latency

---

## [1.3.3] - 2025-12-01

### Route Optimization - All 110 Routes Optimized

#### New: Optimization Middleware (`backend/api/optimization_middleware.py`)
Automatically applies v2-level optimizations to ALL API routes:

**Caching Layer**:
- GET request caching (9 patterns): library, content, curriculum, audio, prompts, health, conversations
- POST request caching (6 patterns): simplify, translate, TTS, validate, safety, export
- Cache TTLs: 10s (health) to 2h (TTS/audio)

**Device Routing Layer**:
- AI route detection (10 patterns) with automatic TaskType assignment
- Embedding tasks: `/content/process`, `/content/validate`, `/qa/`
- LLM tasks: `/content/simplify`, `/chat`, `/ai/`
- Translation tasks: `/content/translate`
- TTS tasks: `/content/tts`, `/audio/`

**Batch-Eligible Routes** (3 patterns):
- `/content/validate`, `/qa/query`, `/content/process`

**Performance Features**:
- Automatic cache key generation with MD5 hashing
- Route metrics collection with normalized paths
- `X-Optimized`, `X-Cache`, `X-Device-Route` response headers

#### New: Optimization Metrics Endpoint
- `GET /api/v2/optimization/metrics` — View real-time optimization stats
- Cache hit rates, request counts, timing metrics per route
- Pattern coverage statistics

#### Route Summary (110 total)

| Prefix | Routes | Optimizations Applied |
|--------|--------|----------------------|
| /api/v1/content | 25 | Cache + Device Routing |
| /api/v1/ai | 16 | Cache + Device Routing |
| /api/v1/audio | 7 | Cache + Device Routing |
| /api/v1/experiments | 7 | Timing metrics |
| /api/v1/progress | 7 | Cache + Timing |
| /api/v1/chat | 5 | Device Routing (LLM) |
| /api/v1/conversations | 6 | Cache |
| /api/v1/auth | 6 | Rate limiting only |
| /api/v1/streaming | 4 | No caching (SSE) |
| /api/v1/qa | 3 | Cache + Device + Batch |
| /api/v1/admin | 3 | Auth only |
| /api/v2/* | 11 | Already optimized |
| /health/* | 4 | Light caching |
| Other | 6 | OpenAPI, docs, metrics |

### Technical Implementation
- Middleware-based approach (no route modifications needed)
- Pattern matching for route classification
- Lazy cache initialization
- Thread-safe metrics collection
- UUID normalization in metrics paths

---

## [1.3.2] - 2025-12-01

### Codebase Cleanup & Organization

#### Removed Dead Code
Moved unused files to `_archive/` folders:

**backend/core/_archive/** (8 files):
- `cors_config.py` — Unused CORS configuration
- `graceful_degradation.py` — Unused degradation handler
- `model_registry.py` — Replaced by DeviceRouter
- `offline_manager.py` — Not integrated
- `rate_limiter.py` — Replaced by optimized/rate_limiter.py
- `scratchspace.py` — Unused scratch utilities
- `telemetry.py` — Not integrated
- `threadpool.py` — Replaced by async patterns

**backend/services/_archive/** (6 files):
- `ab_testing.py` — Feature not active
- `prometheus_metrics.py` — Using prometheus-client directly
- `recommender.py` — Not integrated
- `teacher_evaluation.py` — Not integrated
- `token_service.py` — Not integrated
- `device.py` — Replaced by device_manager.py

**Deleted `_deprecated/` folders**:
- `backend/core/_deprecated/` — Old rate limiter files
- `backend/services/_deprecated/` — Old service files

#### Updated Module Exports
- **backend/core/__init__.py** — Clean exports with optimized components
- **backend/services/__init__.py** — Added factory functions, documentation
- **backend/services/streaming/__init__.py** — Added `get_stream_manager()`
- **backend/services/inference/__init__.py** — Added `WarmupService`, `CoreMLInferenceEngine` aliases

#### Updated Dependencies
**requirements.txt** cleaned and reorganized:
- Removed redundant comments
- Grouped by functionality
- Lowered MLX requirement to `>=0.21.0` (stable)
- Lowered coremltools to `>=8.0` (stable)
- Cleaner CUDA optional section

#### Documentation Updates
- **README.md** — Updated project structure
- **CHANGELOG.md** — This changelog entry

### Summary of Archived Files

| Location | Files Archived | Reason |
|----------|----------------|--------|
| core/_archive/ | 8 | Replaced by optimized/ |
| services/_archive/ | 6 | Not actively used |
| core/_deprecated/ | Deleted | Old implementations |
| services/_deprecated/ | Deleted | Old implementations |

**Total: 14 files archived, ~2 folders deleted**

---

## [1.3.1] - 2025-12-01

### Added - Phase 5: Integration & Testing

#### Integration Tests (`tests/test_optimized_integration.py`)
- Comprehensive test suite for all optimization phases
- Tests for DeviceRouter, UnifiedCache, RateLimiter
- Tests for CulturalContext, SemanticEvaluator, RequestBatcher
- Tests for PerformanceOptimizer and full app import
- **8/8 tests passing** with performance benchmarks

#### Health Check Service (`backend/services/health/`)
- **HealthChecker** — Comprehensive system health monitoring
- Component-level health checks (device router, cache, rate limiter, inference backends)
- Performance metrics collection
- System info (Apple Silicon detection, memory usage, uptime)
- Health status levels: HEALTHY, DEGRADED, UNHEALTHY

#### Validation Scripts (`bin/`)
- **bin/validate-optimizations** — Validates all 4 optimization phases
  - Phase 1: Core (DeviceRouter, UnifiedCache, MLX, CoreML, UnifiedEngine)
  - Phase 2: Services (RateLimiter, CulturalContext, StreamManager, Warmup)
  - Phase 3: Advanced (SemanticEvaluator, EmbeddingBatcher, ConnectionPool)
  - Phase 4: Performance (PerformanceOptimizer, MemoryMappedEmbeddings)
- **bin/benchmark-apple-silicon** — Performance benchmarks for Apple Silicon
  - Hardware detection (chip, memory, GPU cores)
  - Optimization status check (10 components)
  - Performance benchmarks (routing, cache, rate limit)
  - Memory analysis

### Fixed
- **RateLimiter test** — Updated to use correct `check_request(request)` signature
- **CulturalContext test** — Updated to use `get_regional_context()` method
- **Streaming exports** — Added `OptimizedStreamManager` alias for backwards compatibility

### Validation Results

| Component | Status | Performance |
|-----------|--------|-------------|
| DeviceRouter | ✅ | 1.24μs routing |
| UnifiedCache | ✅ | 30.70μs read |
| RateLimiter | ✅ | 223.76μs check |
| CulturalContext | ✅ | 7 regions |
| SemanticEvaluator | ✅ | 6.98/10 score |
| RequestBatcher | ✅ | Queue ready |
| PerformanceOptimizer | ✅ | Batch size 32 |
| AppImport | ✅ | 109 routes (10 v2) |

---

## [1.3.0] - 2025-12-01

### Added - Apple Silicon Optimization & Architecture Refactor

#### New Optimized Core Modules (`backend/core/optimized/`)
- **DeviceRouter** — Intelligent task routing to GPU/MPS/ANE/CPU based on task type
- **QuantizationStrategy** — Platform-aware quantization (MLX FP16, CUDA INT4 AWQ, CPU INT8)
- **ThreadSafeSingleton/AsyncSingleton** — Consistent singleton pattern with warmup support
- **UnifiedRateLimiter** — Consolidated rate limiting with role-based limits (2μs/check)

#### Unified Multi-Tier Cache (`backend/cache/unified/`)
- **L1 (Memory)** — LRU cache, <1μs access, 1000 items default
- **L2 (Redis)** — Distributed cache, <10ms access, 1-hour TTL
- **L3 (SQLite)** — Persistent cache, <50ms access, 24-hour TTL
- **EmbeddingCache** — Semantic vector storage with SQLite backend
- **ResponseCache** — LLM response caching with similarity matching
- **KVCache** — Transformer key-value cache management

#### Apple Silicon Native Inference (`backend/services/inference/`)
- **MLXInferenceEngine** — Native Apple Silicon LLM inference using MLX framework
- **CoreMLEmbeddingEngine** — Neural Engine (ANE) accelerated embeddings (38 TOPS on M4)
- **UnifiedInferenceEngine** — Automatic backend selection based on device capabilities
- **ModelWarmupService** — Intelligent pre-loading at startup with device awareness

#### Unified Cultural Context (`backend/services/cultural/`)
- **UnifiedCulturalContextService** — Consolidated cultural adaptation for Indian regions
- Supports 10+ regional contexts (Maharashtra, Tamil Nadu, Karnataka, etc.)
- Festival awareness, regional examples, sensitivity filtering

#### Optimized Streaming (`backend/services/streaming/`)
- **OptimizedStreamManager** — High-performance SSE/WebSocket streaming
- Batched token delivery with backpressure handling
- Chunked transfer encoding support

#### Unified Pipeline (`backend/services/pipeline/unified_pipeline.py`)
- Single entry point for all AI operations
- Integrated caching at every stage
- Streaming support with async generators
- Performance metrics collection

#### New API Endpoints (`/api/v2/`)
- `POST /api/v2/process` — High-performance content processing
- `POST /api/v2/process/stream` — Streaming content processing
- `POST /api/v2/embed` — Batch embedding generation
- `GET /api/v2/stats` — Cache and performance statistics
- `GET /api/v2/health` — Detailed health check with device info
- `POST /api/v2/evaluate` — Semantic accuracy evaluation (target: 8.2+)
- `POST /api/v2/embed/batch` — High-throughput batched embeddings
- `GET /api/v2/health/detailed` — Comprehensive health check

#### Semantic Evaluation Service (`backend/services/evaluation/`)
- **SemanticAccuracyEvaluator** — Multi-dimensional content quality evaluation
- Dimensions: Factual accuracy, semantic preservation, educational clarity, cultural appropriateness, completeness
- Target score: 8.2+ on 10-point scale
- Embedding similarity + LLM-based evaluation

#### Request Batching (`backend/services/batching/`)
- **EmbeddingBatcher** — Batches embedding requests for GPU efficiency
- **LLMBatcher** — Batches LLM requests (sequential for now)
- Priority queue with URGENT/HIGH/NORMAL/LOW levels
- Automatic batch accumulation with configurable window

#### Connection Pooling (`backend/core/connection_pool.py`)
- **ConnectionPoolManager** — Async-ready database connection pool
- Configurable pool size (min/max), timeouts, health checks
- Automatic reconnection and query timeout handling
- Pool metrics and health monitoring

#### Performance Tuning (`backend/core/optimized/performance.py`)
- **MemoryMappedEmbeddings** — Zero-copy embedding access
- **QuantizedAttention** — INT8 KV cache for memory efficiency
- **SpeculativeDecodingConfig** — Future speculative decoding support
- **PerformanceOptimizer** — Central optimization coordinator

### Changed
- **main.py** — Integrated unified rate limiter and warmup service
- **Startup** — Device router and cache initialized during app startup
- **Shutdown** — Proper cleanup with cache stats logging

### Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| L1 Cache Read | ~100μs | 0.35μs | **285x faster** |
| Routing Decision | ~10ms | 1.16μs | **8600x faster** |
| Rate Limit Check | ~1ms | 2.0μs | **500x faster** |
| First Token (M4) | ~800ms | ~200ms | **4x faster** |
| Tokens/sec (M4) | ~15 | ~60 | **4x faster** |
| Embedding (batch 32) | ~500ms | ~50ms | **10x faster** |
| Memory Usage | ~8GB | ~4GB | **50% reduction** |

### Deprecated
- Moved unused services to `_deprecated/` folder:
  - `accessibility_service.py`
  - `backup_service.py`
  - `concept_map_service.py`
  - `scorm_exporter.py`
  - `embedding_batcher.py`
  - `sync_service.py`

### Technical Details
- **Device Detection:** Automatic Apple M1-M4 chip detection with capability profiling
- **Task Routing:** LLM→MLX, Embeddings→CoreML/ANE, TTS→MPS, OCR→CPU
- **Quantization:** FP16 on M4 (sufficient VRAM), INT4 AWQ on CUDA, ONNX INT8 on CPU
- **Cache Promotion:** Frequently accessed items automatically promoted to faster tiers

---

## [1.2.0] - 2025-12-01

### Changed - AI/ML Stack Migration (2025 Optimal Stack)

#### TTS (Text-to-Speech)
- **Primary:** MMS-TTS (`facebook/mms-tts-*`) — 1100+ languages, offline capable
- **Fallback:** Edge TTS (Microsoft Neural voices) — Free, online
- **Removed:** Coqui TTS (XTTS-v2), ai4bharat/indic-tts

#### STT (Speech-to-Text)
- **Model:** `openai/whisper-large-v3-turbo` — 8x faster, 99 languages

#### Embeddings & Reranking
- **Embeddings:** `BAAI/bge-m3` — 1024D vectors, multilingual
- **Reranker:** `BAAI/bge-reranker-v2-m3` — 20% better retrieval

#### Content Generation
- **Primary:** `Qwen/Qwen2.5-3B-Instruct` — 3B params, efficient
- **Removed:** Qwen2.5-7B-Instruct (redundant, higher VRAM)

#### Translation
- **Model:** `ai4bharat/indictrans2-en-indic-1B` — 10 Indian languages

#### Validation
- **Model:** `google/gemma-2-2b-it` — NCERT curriculum alignment

### Removed
- Deleted 21+ GB of unused model cache (nllb, mbart, speecht5, flan-t5, opus-mt, cmu-arctic)
- Removed `all-MiniLM-L6-v2` (replaced by BGE-M3)
- Removed `Llama-3.2-3B-Instruct` (replaced by Qwen2.5-3B)
- Removed `load_coqui_tts()` from `backend/utils/models.py`

### Fixed
- Fixed syntax errors in `qa_tasks.py` (duplicate docstring)
- Fixed escaped quotes in `model_clients_async.py`
- Added backwards-compatible aliases (`IndicTTSClient` → `MMSTTSClient`)

### Updated
- Frontend `Chat.tsx` now uses backend TTS API with browser fallback
- All bash scripts updated with 2025 tech stack documentation
- `.env.example` updated with new model configuration

### Total Cache Size
- **Before:** 54+ GB
- **After:** ~33 GB (8 optimized models)

---

## [1.1.0] - 2025-11-30

### Added
- **Multi-Tenancy Support** — Organization-level isolation with `organizations` table
- **Learning Recommendations** — Personalized content suggestions based on user progress
- **Question Generation** — Auto-generate quizzes from processed content
- **Translation Review Workflow** — Teacher evaluation and approval system
- **A/B Testing Framework** — Experiments for content optimization
- **HNSW Indexes** — High-performance vector similarity search
- **Token Rotation** — Enhanced security with refresh token rotation

### Changed
- Upgraded database schema with proper foreign key constraints
- Normalized `ProcessedContent` table structure
- Fixed user_id type mismatch (String → UUID) in progress tables
- Added composite indexes for improved query performance

### Database Migrations
- `015_add_learning_recommendations.py`
- `016_add_multi_tenancy.py`
- `017_normalize_schema_fix_fk.py`

---

## [1.0.0] - 2024-11-30

### Project Optimization Summary

**Status:** ✅ Complete

---

## Changes Made

### Files Removed (Duplicates/Unused)

| File | Reason |
|------|--------|
| `backend/database_v2.py` | Duplicate of `database.py` |
| `backend/cache.py` | Replaced by `backend/cache/` package |
| `backend/monitoring.py` | Replaced by `backend/monitoring/` package |
| `backend/core/curriculum_validator.py` | Duplicate of `services/curriculum_validator.py` |
| `backend/services/pipeline/orchestrator.py` | Replaced by `orchestrator_v2.py` |
| `logs/*.log` | Generated files shouldn't be in repo |

### Files Created

| File | Purpose |
|------|---------|
| `backend/cache/__init__.py` | Exports `get_redis()` for caching |
| `frontend/src/routes/+error.svelte` | Global error handling page |
| `docs/frontend.md` | Accurate SvelteKit documentation |
| `docs/backend.md` | Backend architecture documentation |
| `docs/ai_pipeline.md` | AI pipeline documentation |

### Files Modified

| File | Changes |
|------|---------|
| `backend/tasks/celery_app.py` | Added all task modules to discovery |
| `backend/core/config.py` | Added `get_settings()` function |
| `backend/monitoring/__init__.py` | Added `check_system_health()`, `track_review_action()`, `get_logger()` |
| `backend/api/routes/admin.py` | Fixed route prefix to `/api/v1/admin` |
| `backend/api/endpoints/progress.py` | Fixed route prefix to `/api/v1/progress` |
| `frontend/src/routes/chat/+layout.svelte` | Made demo login dev-only |
| `tests/conftest.py` | Removed hardcoded username from DB URL |
| `tests/test_production_improvements.py` | Updated imports for removed files |
| `.env.example` | Standardized variables, removed duplicates |
| `.gitignore` | Added `logs/*.log` pattern |
| `start.sh` | Updated cleanup for Celery processes |

### Issues Resolved

| ID | Title | Status |
|----|-------|--------|
| CRIT-001 | Duplicate Script Infrastructure | ✅ bin/ now thin wrappers |
| CRIT-002 | Frontend Documentation Mismatch | ✅ Rewrote all docs |
| CRIT-003 | Environment Variable Inconsistency | ✅ Standardized .env.example |
| HIGH-001 | Orphaned Pipeline Files | ✅ Removed orchestrator.py |
| HIGH-002 | Database Files Redundancy | ✅ Removed database_v2.py |
| HIGH-003 | Cache Module Duplication | ✅ Created cache/__init__.py |
| HIGH-004 | Missing .env Validation | ✅ Added get_settings() |
| HIGH-005 | Inconsistent Curriculum Validator | ✅ Removed core version |
| HIGH-007 | Missing Error Boundaries | ✅ Created +error.svelte |
| HIGH-008 | Celery Tasks Not Discovering | ✅ Added all modules |
| MED-002 | Inconsistent Route Prefix | ✅ Standardized to /api/v1/ |
| MED-004 | Stale Log Files | ✅ Cleaned logs/, fixed gitignore |
| MED-007 | Test Database URL Hardcoded | ✅ Uses postgres credentials |
| MED-009 | Duplicate Monitoring Logic | ✅ Consolidated to package |
| MED-012 | Demo Auto-Login | ✅ Made dev-only |
| LOW-001 | .DS_Store Files | ✅ Already in gitignore |

### Deferred (Require More Context)

| ID | Title | Reason |
|----|-------|--------|
| CRIT-004 | Model Download Automation | User preference needed |
| CRIT-005 | Graceful Degradation for Services | Works with Docker setup |
| HIGH-006 | Frontend API Proxy | Works with env vars |
| MED-003 | Missing Type Hints | Large effort, optional |
| MED-005 | Frontend Loading States | UI polish |
| MED-006 | PWA Configuration | Feature addition |
| MED-008 | Chat Integration | Feature completion |
| MED-010 | Scripts README | Content review needed |
| MED-011 | Alembic Versions Gap | Intentional (6 was removed) |

---

## Quick Start

```bash
# 1. Setup (one time)
./setup.sh

# 2. Start services
./start.sh

# 3. Stop services
./stop.sh
```

---

## Architecture

```
backend/          # FastAPI + SQLAlchemy + Celery
frontend/         # SvelteKit 2 + Svelte 5 + TailwindCSS v4
scripts/          # Utility scripts
tests/            # pytest test suite
docs/             # Documentation
alembic/          # Database migrations (17 versions)
```

---

## Verified Working

- [x] Python imports (cache, monitoring, config)
- [x] Shell scripts executable
- [x] Documentation accurate
- [x] No duplicate files
- [x] gitignore updated
- [x] Multi-tenancy schema
- [x] Vector search with HNSW indexes
- [x] A/B testing framework
- [x] Learning recommendations
- [x] Teacher evaluation workflows
