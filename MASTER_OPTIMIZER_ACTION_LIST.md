# 🎯 MASTER-OPTIMIZER ACTION LIST
## ShikshaSetu Full-Stack Optimization Plan

**Date**: 2025-11-28  
**Branch**: `master-optimizer/full-stack-optimization`  
**Backup**: `backup/pre-master-optimizer`  
**Status**: PHASE A COMPLETE - AWAITING APPROVAL TO PROCEED

---

## 📊 ANALYSIS SUMMARY

### Repository Statistics
- **Backend**: 113 Python files, ~25,000 LOC
- **Frontend**: 63 TypeScript/TSX files, ~8,000 LOC  
- **Tests**: 11 test files, 93 passing tests (23% coverage)
- **Models**: 2.1GB in storage/data/models (Qwen, E5-Large, MuRIL)
- **Infrastructure**: Docker, K8s, monitoring stack ready

### Current State Assessment

#### ✅ **STRENGTHS**
1. **Well-structured architecture**: Clear separation (backend/, frontend/, tests/, infrastructure/)
2. **Modern stack**: FastAPI, React 19, TypeScript, Vite 7, PostgreSQL 17, Redis
3. **Production-ready features**: Auth (JWT), RBAC, monitoring (Prometheus/Grafana), CI/CD
4. **ML pipeline exists**: 4-stage orchestrator (simplify→translate→validate→TTS)
5. **Documentation**: Comprehensive (README, DEPLOYMENT.md, API docs)
6. **Device awareness**: DeviceManager with CUDA/MPS/CPU detection

#### ⚠️ **CRITICAL ISSUES**
1. **NO LOCAL-FIRST OPTIMIZATION**: Models assume API/GPU, no M4-optimized path
2. **NO QUANTIZATION**: 7B models loaded in FP32/FP16 → 14-28GB RAM
3. **NO MODEL TIERS**: No task_weight routing (SMALL/MEDIUM/LARGE)
4. **MEMORY LEAKS**: Full file reads (`await file.read()`) for 100MB+ uploads
5. **BLOCKING CALLS**: Synchronous model loading, no lazy loading implemented
6. **NO STREAMING**: Large inference responses sent as single payload
7. **DUPLICATE CODE**: 3x model client implementations (sync, async, orchestrator)
8. **MISSING TESTS**: Only 23% coverage, no E2E tests for full pipeline
9. **NO RESOURCE LIMITS**: No circuit breakers, no graceful degradation
10. **UNCLEAR DEPLOYMENT**: Docker images but no clear local vs production split

#### 🔍 **DEPENDENCY ANALYSIS**

**Backend (Python)**:
- **Heavy deps**: `torch==2.6.0` (2GB+), `transformers==4.47.1` (500MB+), `vllm==0.6.3` (1GB+)
- **Quantization deps**: `bitsandbytes`, `auto-gptq` (CUDA only, not used)
- **OCR**: `paddleocr`, `paddlepaddle` (large, CPU-bound)
- **Audio**: `TTS`, `librosa`, `pydub` (FFmpeg required)
- **Vector DB**: `pgvector` (needs PostgreSQL extension)
- **Monitoring**: OpenTelemetry, Prometheus, Sentry
- **Unused?**: `dvc`, `tritonclient` (not configured)

**Frontend (TypeScript)**:
- **Heavy deps**: `framer-motion` (52KB), `workbox` (service worker, 100KB+)
- **State**: `zustand` (good choice, 3KB)
- **UI**: Radix UI (tree-shakeable, good)
- **Query**: TanStack Query (cache, good)
- **Offline**: `idb` for IndexedDB

#### 🗺️ **API MAP** (20+ endpoints)

| Route | Method | Purpose | Issues |
|-------|--------|---------|--------|
| `/api/v1/auth/login` | POST | JWT login | ✅ Working |
| `/api/v1/auth/register` | POST | User registration | ✅ Working |
| `/api/v1/content/upload` | POST | File upload | ⚠️ Full file read |
| `/api/v1/content/process` | POST | Pipeline trigger | ⚠️ Blocking |
| `/api/v1/content/tasks/{id}` | GET | Task status | ✅ Working |
| `/api/v1/content/audio/{id}` | GET | Audio retrieval | ⚠️ No streaming |
| `/api/v1/qa/ask` | POST | RAG Q&A | ⚠️ No pagination |
| `/api/v1/streaming/translate` | WS | WebSocket translation | ⚠️ Unused? |
| `/health` | GET | Health check | ✅ Working |
| `/metrics` | GET | Prometheus metrics | ✅ Working |

#### 🤖 **ML MODEL MAP**

| Model | ID | Size | Usage | Issues |
|-------|-----|------|-------|--------|
| **Content Gen** | `Qwen/Qwen2.5-7B-Instruct` | ~14GB FP16 | Simplification | ❌ No quantization, no MPS opt |
| **Translation** | `ai4bharat/indictrans2-en-indic-1B` | ~2GB | Translate | ⚠️ API fallback only |
| **Embeddings** | `intfloat/multilingual-e5-large` | ~2GB | RAG, search | ⚠️ No ONNX |
| **Validator** | `ai4bharat/indic-bert` | ~500MB | NCERT validation | ⚠️ Loads every time |
| **TTS** | `facebook/mms-tts-hin` | ~500MB/lang | Audio | ⚠️ No caching |
| **OCR** | PaddleOCR | ~200MB | Text extraction | ⚠️ CPU-bound |

**Current Model Loading**:
- ❌ All models loaded via `AutoModel.from_pretrained()` with `device_map="auto"`
- ❌ No quantization (4-bit/8-bit disabled)
- ❌ No lazy loading (all loaded at startup or first use, never unloaded)
- ❌ No batching
- ❌ No model cache eviction (LRU exists but not used)
- ❌ No MPS-specific optimizations (PyTorch MPS backend not fully utilized)
- ❌ No llama.cpp / GGUF support
- ❌ No vLLM integration (code exists but `VLLM_ENABLED=false`)

#### 🔥 **PERFORMANCE HOTSPOTS**

1. **File Upload** (`backend/api/routes/content.py:180`):
   ```python
   content = await file.read()  # ❌ Reads entire file into RAM
   ```
   - **Impact**: 100MB PDF → 100MB+ RAM instantly
   - **Fix**: Chunk streaming, process incrementally

2. **Model Loading** (`backend/pipeline/model_clients.py:519`):
   ```python
   self.local_model = AutoModelForCausalLM.from_pretrained(...)
   ```
   - **Impact**: 14GB model load blocks API for 30-60s
   - **Fix**: Lazy load, async, LRU cache, quantize

3. **Orchestrator** (`backend/pipeline/orchestrator.py:169`):
   - Sequential blocking: simplify → translate → validate → TTS
   - **Impact**: 4-stage × 5s = 20s minimum latency
   - **Fix**: Async/parallel where possible, stream intermediate results

4. **OCR** (`backend/services/ocr.py`):
   - PaddleOCR runs on CPU, single-threaded
   - **Impact**: 10-page PDF → 30-60s
   - **Fix**: Multi-page parallel, use Tesseract fallback, cache results

5. **RAG Query** (`backend/services/rag.py`):
   - N+1 queries, no pagination, no hybrid search
   - **Impact**: Slow for large knowledge bases
   - **Fix**: Batch embeddings, use HNSW index, hybrid BM25+vector

6. **Frontend Bundle**:
   - No code splitting beyond React.lazy (not used)
   - **Impact**: 800KB+ initial load
   - **Fix**: Route-based splitting, lazy UI components

---

## 🎯 PRIORITIZED ACTION LIST

### **CRITICAL** (Must fix for local M4 functionality)

#### C1. **Implement Model Tier Routing** [EFFORT: HIGH | RISK: MEDIUM]
**Problem**: No task_weight→model mapping. All tasks use largest models.

**Changes**:
- Create `backend/core/model_tier_router.py`:
  - `ModelTier` enum: SMALL, MEDIUM, LARGE
  - `task_weight(text_length, complexity) → tier` formula
  - Threshold config: `<500 tokens → SMALL, <2000 → MEDIUM, else LARGE`
- Map tiers to models:
  - **SMALL**: `Qwen2.5-1.5B-Instruct` (3GB FP16, 800MB 4-bit) or API
  - **MEDIUM**: `Qwen2.5-7B-Instruct` (14GB FP16, 3.5GB 4-bit)
  - **LARGE**: vLLM cluster or API fallback
- Update `ContentPipelineOrchestrator` to call router
- **Files**: `backend/core/model_tier_router.py` (new), `orchestrator.py`, `config.py`
- **Tests**: Unit tests for task_weight, integration test for tier routing

**Acceptance**:
- ✅ Small tasks (grade 5, <500 tokens) use SMALL model
- ✅ Memory usage <5GB for SMALL, <8GB for MEDIUM (4-bit)
- ✅ 90%+ tasks complete successfully on M4

---

#### C2. **Add 4-bit Quantization for Qwen2.5** [EFFORT: MEDIUM | RISK: LOW]
**Problem**: 7B model → 14GB FP16, exceeds M4 unified memory budget.

**Changes**:
- Install `bitsandbytes` (CUDA) or use `llama.cpp` (cross-platform)
- Add quantization logic in `backend/core/model_loader.py`:
  ```python
  if settings.USE_QUANTIZATION and device == "cuda":
      model = AutoModelForCausalLM.from_pretrained(
          model_id,
          load_in_4bit=True,
          bnb_4bit_compute_dtype=torch.float16
      )
  elif settings.USE_QUANTIZATION and device == "mps":
      # Use llama.cpp GGUF Q4_K_M
      model = load_llama_cpp_model(gguf_path)
  ```
- Download/convert Qwen2.5-7B to GGUF Q4_K_M (~3.5GB)
- Update `settings.py`: `CONTENT_GEN_QUANTIZATION="4bit"`, `USE_QUANTIZATION=true`
- **Files**: `model_loader.py`, `settings.py`, `requirements.txt` (+llama-cpp-python)
- **Tests**: Load model, verify inference, check memory <5GB

**Acceptance**:
- ✅ Qwen2.5-7B loads in <5GB on MPS/CUDA
- ✅ Inference works (BLEU score >0.7 vs FP16 baseline)
- ✅ Load time <30s

---

#### C3. **Implement Lazy Model Loading with LRU Eviction** [EFFORT: HIGH | RISK: MEDIUM]
**Problem**: All models loaded at startup or first use, never unloaded. 20GB+ total.

**Changes**:
- Refactor `backend/core/model_loader.py` (exists but unused):
  - `LazyModelLoader` class with LRU cache
  - `load_model(model_name, tier)` returns cached or loads new
  - `evict_lru()` unloads oldest, least-used model when cache full
  - Config: `MAX_CACHE_SIZE_MB=8192` (8GB for M4)
- Update all model clients (`FlanT5Client`, `IndicTrans2Client`, etc.):
  - Remove `__init__` model loading
  - Load on first `process()` call
  - Register with `LazyModelLoader`
- Add idle timeout: unload models unused for 5 minutes
- **Files**: `model_loader.py`, `model_clients.py`, `orchestrator.py`
- **Tests**: Load 3 models, verify 1st evicted when 4th loads

**Acceptance**:
- ✅ Only used models loaded (memory <8GB)
- ✅ LRU eviction works (oldest model unloaded)
- ✅ Idle timeout unloads after 5min
- ✅ No crashes, no performance regression vs eager loading

---

#### C4. **Stream Large File Uploads** [EFFORT: MEDIUM | RISK: LOW]
**Problem**: `await file.read()` loads 100MB+ files into RAM instantly.

**Changes**:
- Refactor upload endpoints (`backend/api/routes/content.py:180`, `audio_upload.py:56`):
  ```python
  async def upload_chunked(file: UploadFile):
      async with aiofiles.open(temp_path, 'wb') as f:
          while chunk := await file.read(8192):  # 8KB chunks
              await f.write(chunk)
  ```
- Add `MAX_UPLOAD_SIZE=104857600` (100MB) check before accepting
- Process file from disk (not RAM) for OCR/parsing
- **Files**: `content.py`, `audio_upload.py`, `helpers.py`
- **Tests**: Upload 100MB file, verify memory <150MB during upload

**Acceptance**:
- ✅ 100MB file uploads without OOM
- ✅ Memory usage during upload <150MB
- ✅ No file corruption

---

#### C5. **Add MPS-Optimized PyTorch Code Paths** [EFFORT: MEDIUM | RISK: LOW]
**Problem**: MPS backend detected but not optimized (no Metal kernels, no float16).

**Changes**:
- Update `backend/utils/device_manager.py`:
  ```python
  if device == "mps":
      torch.set_default_dtype(torch.float16)  # Faster on MPS
      model = model.to("mps")
      # Use torch.mps.empty_cache() after inference
  ```
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops
- Add MPS-specific config:
  ```python
  MPS_MAX_BATCH_SIZE = 4  # Lower than CUDA
  MPS_MAX_SEQ_LENGTH = 1024  # Memory-limited
  ```
- **Files**: `device_manager.py`, `model_clients.py`, `.env.example`
- **Tests**: Run inference on M4, check throughput vs CPU

**Acceptance**:
- ✅ 3-5x speedup vs CPU on M4
- ✅ No MPS-related crashes
- ✅ Memory usage <8GB

---

### **HIGH** (Essential for production-readiness)

#### H1. **Consolidate Model Client Code** [EFFORT: HIGH | RISK: MEDIUM]
**Problem**: 3 implementations (sync, async, orchestrator) with duplicate logic.

**Changes**:
- Create single `backend/services/unified_model_client.py`:
  - Async-first design
  - Lazy loading + LRU
  - Quantization support
  - Bhashini API fallback
  - Circuit breaker pattern
- Deprecate `model_clients.py`, `model_clients_async.py`
- Update orchestrator to use unified client
- **Files**: `unified_model_client.py` (new), `orchestrator.py`, delete old clients
- **Tests**: Full pipeline test with new client

**Acceptance**:
- ✅ All 4 pipeline stages work with unified client
- ✅ Code reduced by 500+ LOC
- ✅ No performance regression

---

#### H2. **Implement Circuit Breaker + Graceful Degradation** [EFFORT: MEDIUM | RISK: LOW]
**Problem**: No fallback when models fail. Service crashes or hangs.

**Changes**:
- Add `backend/utils/circuit_breaker.py` (exists, unused):
  - `@circuit_breaker(failures=3, timeout=10)` decorator
  - Auto-opens after 3 failures, retries after timeout
- Add fallback chain:
  1. Local quantized model (MPS/CUDA)
  2. Bhashini API
  3. Rule-based fallback (for simplification/translation)
- Update all model clients to use circuit breaker
- **Files**: `circuit_breaker.py`, `model_clients.py`, `orchestrator.py`
- **Tests**: Simulate model failure, verify fallback

**Acceptance**:
- ✅ Circuit opens after 3 failures
- ✅ API fallback works
- ✅ No user-facing errors, degraded results returned

---

#### H3. **Add Batch Processing Queue** [EFFORT: HIGH | RISK: MEDIUM]
**Problem**: Sequential 1-by-1 inference. No batching for similar tasks.

**Changes**:
- Create `backend/core/batch_queue.py`:
  - Queue similar tasks (same model, tier)
  - Batch up to `MAX_BATCH_SIZE` (4 for MPS, 16 for CUDA)
  - Dynamic batching: max wait time 200ms
- Update orchestrator to submit to queue instead of direct inference
- **Files**: `batch_queue.py` (new), `orchestrator.py`, `celery_app.py`
- **Tests**: Submit 10 tasks, verify batching

**Acceptance**:
- ✅ Tasks batched (2-10x throughput improvement)
- ✅ Latency <500ms added
- ✅ No deadlocks

---

#### H4. **Implement Incremental OCR Processing** [EFFORT: MEDIUM | RISK: LOW]
**Problem**: OCR blocks on large PDFs (10+ pages, 30-60s).

**Changes**:
- Refactor `backend/services/ocr.py`:
  - Process pages in parallel (ThreadPoolExecutor)
  - Yield results incrementally (async generator)
  - Cache OCR results (Redis, TTL 1 hour)
- Update content upload to stream OCR results
- **Files**: `ocr.py`, `content.py`
- **Tests**: 10-page PDF, verify <15s, memory <500MB

**Acceptance**:
- ✅ 2-3x speedup on multi-page PDFs
- ✅ Incremental results (page-by-page)
- ✅ Cache hit rate >50% for repeated uploads

---

#### H5. **Optimize Frontend Bundle** [EFFORT: MEDIUM | RISK: LOW]
**Problem**: 800KB+ initial bundle, no code splitting.

**Changes**:
- Add route-based splitting in `frontend/src/main.tsx`:
  ```tsx
  const UploadPage = lazy(() => import('./pages/UploadPage'))
  const TaskPage = lazy(() => import('./pages/TaskPage'))
  ```
- Lazy load heavy components (AudioPlayer, FileDropzone, ResultsPanel)
- Analyze bundle with `vite-plugin-visualizer`, split vendor chunks
- Enable compression in Vite (Brotli for production)
- **Files**: `main.tsx`, `vite.config.ts`, component files
- **Tests**: Build, check bundle sizes

**Acceptance**:
- ✅ Initial bundle <300KB
- ✅ Route chunks <100KB each
- ✅ LCP <2s on 3G

---

#### H6. **Add Comprehensive E2E Tests** [EFFORT: HIGH | RISK: LOW]
**Problem**: Only 23% coverage, no full pipeline E2E tests.

**Changes**:
- Create `tests/e2e/test_full_pipeline.py`:
  - Upload PDF → OCR → Simplify → Translate → TTS → Audio
  - Verify all stages complete successfully
  - Check output quality (BLEU, NCERT score)
  - Measure latency and memory
- Mock heavy external calls (Bhashini, large models)
- Run in CI (GitHub Actions)
- Target: 60% coverage
- **Files**: `test_full_pipeline.py` (new), `.github/workflows/test.yml`
- **Tests**: E2E test suite (5-10 scenarios)

**Acceptance**:
- ✅ E2E test passes on CI
- ✅ Coverage >60%
- ✅ Memory profiling shows peak <8GB

---

### **MEDIUM** (Important for scalability)

#### M1. **Implement vLLM Production Path** [EFFORT: HIGH | RISK: HIGH]
**Problem**: vLLM code exists but unused. No production GPU serving.

**Changes**:
- Create `backend/services/vllm_client.py`:
  - Connect to vLLM server (HTTP API)
  - Health checks, retries, fallback
- Add vLLM deployment scripts:
  - `scripts/deployment/deploy_vllm.sh`
  - Docker image with vLLM + Qwen2.5-7B
  - K8s manifest with GPU node selector
- Update tier router: LARGE tier → vLLM
- **Files**: `vllm_client.py`, deploy scripts, k8s manifests
- **Tests**: Integration test with local vLLM instance

**Acceptance**:
- ✅ vLLM deploys on GPU node
- ✅ 5-10x throughput vs local
- ✅ Auto-fallback to API if vLLM down

---

#### M2. **Add ONNX Export for Embeddings** [EFFORT: MEDIUM | RISK: MEDIUM]
**Problem**: Embeddings (E5-Large) slow on CPU (2-5s per query).

**Changes**:
- Export E5-Large to ONNX:
  ```bash
  optimum-cli export onnx --model intfloat/multilingual-e5-large \
    --optimize O3 --fp16 e5_large_onnx/
  ```
- Update `backend/services/rag.py`:
  ```python
  from optimum.onnxruntime import ORTModelForFeatureExtraction
  model = ORTModelForFeatureExtraction.from_pretrained("e5_large_onnx")
  ```
- **Files**: `rag.py`, export script, ONNX model storage
- **Tests**: Benchmark embeddings (ONNX vs PyTorch)

**Acceptance**:
- ✅ 2-3x speedup on CPU
- ✅ No quality degradation (cosine sim >0.99)

---

#### M3. **Implement Hybrid Search (BM25 + Vector)** [EFFORT: MEDIUM | RISK: LOW]
**Problem**: Pure vector search, no keyword matching, slow for exact queries.

**Changes**:
- Add BM25 index (Elasticsearch or pgvector FTS)
- Combine BM25 + vector with reciprocal rank fusion:
  ```python
  hybrid_score = 0.3 * bm25_score + 0.7 * vector_score
  ```
- Update `backend/services/rag.py`
- **Files**: `rag.py`, `models/rag.py` (add FTS index)
- **Tests**: Query precision/recall benchmarks

**Acceptance**:
- ✅ Exact keyword matches rank higher
- ✅ Precision +10-20%

---

#### M4. **Add Prometheus Alerts for Resource Usage** [EFFORT: LOW | RISK: LOW]
**Problem**: Monitoring exists but no alerting.

**Changes**:
- Add Prometheus alert rules (`infrastructure/monitoring/prometheus-alerts.yml`):
  - Memory usage >80%
  - Model load time >60s
  - Request latency P95 >10s
  - Error rate >5%
- Configure Alertmanager (Slack/email)
- **Files**: `prometheus-alerts.yml`, `alertmanager.yml`
- **Tests**: Trigger alert (simulate high memory), verify notification

**Acceptance**:
- ✅ Alerts fire on threshold breach
- ✅ Notifications sent to Slack/email

---

#### M5. **Optimize Docker Images** [EFFORT: MEDIUM | RISK: LOW]
**Problem**: Large Docker images (3GB+ backend), slow builds.

**Changes**:
- Multi-stage Dockerfile:
  - Build stage: compile deps
  - Runtime stage: copy only runtime deps, no build tools
- Use slim base image (`python:3.11-slim` instead of full)
- Add `.dockerignore`: exclude tests, docs, .git
- Pre-download models in build (bake into image)
- **Files**: `Dockerfile.backend`, `.dockerignore`
- **Tests**: Build, check size <1GB

**Acceptance**:
- ✅ Backend image <1GB (from 3GB+)
- ✅ Build time <5min

---

### **LOW** (Nice-to-have, polish)

#### L1. **Add Admin Dashboard** [EFFORT: MEDIUM | RISK: LOW]
- React admin UI for user management, system stats, model monitoring
- **Files**: `frontend/src/pages/AdminDashboard.tsx` (new)

#### L2. **Implement Content Versioning** [EFFORT: MEDIUM | RISK: LOW]
- Track edits, rollback, diff view
- **Files**: `backend/models/content.py`, new API routes

#### L3. **Add Multi-Tenancy** [EFFORT: HIGH | RISK: MEDIUM]
- Organization/school isolation, separate DB schemas
- **Files**: `backend/middleware/tenant.py`, migrations

#### L4. **Optimize Database Indexes** [EFFORT: LOW | RISK: LOW]
- Add HNSW index for pgvector (faster ANN search)
- Add composite indexes for common queries
- **Files**: Alembic migrations

#### L5. **Add Health Check for Models** [EFFORT: LOW | RISK: LOW]
- Extend `/health/detailed` to check if models loaded
- **Files**: `backend/api/routes/health.py`

---

## 📈 EXPECTED IMPACT

| Metric | Before | After C1-C5 | After H1-H6 | After M1-M5 |
|--------|--------|-------------|-------------|-------------|
| **Peak Memory (M4)** | 20GB+ ❌ | 5-8GB ✅ | 5-8GB ✅ | 5-8GB ✅ |
| **Load Time (Model)** | 60s | 15-30s | 10-20s | 5-10s (vLLM) |
| **Inference Latency** | 5-10s | 3-5s | 2-4s (batch) | 1-2s (vLLM) |
| **Upload (100MB file)** | 100MB RAM | 50MB RAM | 50MB RAM | 50MB RAM |
| **Test Coverage** | 23% | 30% | 60% | 65% |
| **Bundle Size** | 800KB | 800KB | 300KB | 300KB |
| **Local Runnable** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Production Ready** | ⚠️ Partial | ⚠️ Partial | ✅ Yes | ✅ Yes |

---

## 🛠️ EFFORT & RISK SUMMARY

| Priority | Tasks | Total Effort | Total Risk | Estimated Time |
|----------|-------|--------------|------------|----------------|
| **CRITICAL** | 5 | 19 person-days | Medium | 3-4 weeks |
| **HIGH** | 6 | 20 person-days | Medium | 3-4 weeks |
| **MEDIUM** | 5 | 15 person-days | Medium | 2-3 weeks |
| **LOW** | 5 | 10 person-days | Low | 1-2 weeks |
| **TOTAL** | 21 | 64 person-days | - | **8-12 weeks** |

---

## 🚦 ROLLBACK PLAN

All changes are on `master-optimizer/full-stack-optimization` branch.

**Rollback commands**:
```bash
# Option 1: Revert to backup branch
git checkout backup/pre-master-optimizer
git branch -D master-optimizer/full-stack-optimization

# Option 2: Reset to pre-optimizer commit
git checkout main
git reset --hard <commit-id-before-optimizer>

# Option 3: Revert specific commit
git revert <commit-sha>
```

**Backup files**: All replaced files stored in `.backup/` with timestamps.

---

## 📋 NEXT STEPS (PHASE B)

**AWAITING USER APPROVAL TO PROCEED WITH PHASE C (CRITICAL CHANGES)**

1. **User Approval**: Confirm ACTION LIST priorities
2. **Select Tasks**: Which priority level to implement (CRITICAL? CRITICAL + HIGH?)
3. **Begin PHASE C**: Implement changes in priority order
4. **Iterative Testing**: Run tests after each task
5. **Commit Strategy**: Small, reversible commits with `MO:` prefix
6. **Report Progress**: Update this document with completion status

---

## 🎯 ACCEPTANCE CRITERIA (FINAL)

Before merging to `main`, all must be ✅:

- [ ] All CRITICAL tasks (C1-C5) completed and tested
- [ ] All unit tests pass (`pytest -q`)
- [ ] E2E pipeline test passes (upload→OCR→simplify→translate→TTS)
- [ ] Memory usage <8GB on M4 for heavy task
- [ ] Qwen2.5-7B loads in 4-bit (<5GB)
- [ ] Test coverage >60%
- [ ] All API endpoints return HTTP 2xx for valid inputs
- [ ] Docker images build successfully
- [ ] Documentation updated (README, DEPLOYMENT, API docs)
- [ ] No secrets in logs or code
- [ ] Rollback plan tested

---

**Generated by**: MASTER-OPTIMIZER  
**Contact**: Ready to proceed with PHASE C on approval
