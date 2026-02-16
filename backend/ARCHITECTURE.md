# Oryon Backend — Six-Layer Research-Lab Architecture

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — API           (FastAPI routes, middleware, guards)   │
│  backend/api/                                                   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — Services      (business logic, chat, content, users)│
│  backend/services/                                              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 — ML            (inference, pipeline, speech, translate)│
│  backend/ml/                                                    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4 — Infrastructure (cache, hardware, telemetry, runtime)│
│  backend/infra/                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5 — Core          (config, security, exceptions, HAL)   │
│  backend/core/                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6 — Database      (SQLAlchemy engine, sessions, models) │
│  backend/db/ + backend/models/                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Invariants

1. **Downward-only dependencies**: Layer N may only import from layer N+1 or higher.
   API → Services → ML → Infrastructure → Core → Database.
2. **Singleton pattern**: Every layer exposes `get_*()` factory functions returning
   thread-safe singletons. No global mutable state.
3. **Async-first**: All I/O-bound operations are `async`. CPU-bound work uses
   `asyncio.to_thread()` or the worker pool.
4. **Graceful degradation**: Each layer initializes inside a `try/except` in the
   lifespan manager. If a layer fails, higher layers continue with reduced capability.

## Request Flow

```
HTTP Request
  → UnifiedMiddleware (rate limit, request ID, timing)
  → FastAPI Route Handler (Layer 1)
    → RequestDispatcher.dispatch() (Layer 2)
      → PipelineContext created
      → AdaptiveRouter.route() → complexity analysis → model selection
      → PriorityQueueManager.enqueue() (if needed)
      → InferenceRuntime.execute() (Layer 3)
        → WorkerPool.submit() → concurrency control
        → DynamicBatcher.submit() → batch accumulation (if eligible)
        → InferenceBackend.infer() → actual model call
          → UnifiedBridgeBackend → existing services (ai_core, rag, tts, etc.)
      → Result returned through pipeline stages
  → HTTP Response
```

## Startup Sequence

```
1. Core Infrastructure
   ├── Memory coordinator
   ├── OpenTelemetry tracing
   ├── Circuit breakers (legacy)
   ├── Sentry error tracking
   ├── Database (PostgreSQL + pgvector)
   ├── Device router + Unified cache
   └── GPU Pipeline Scheduler

2. Layer 4 — Hardware
   ├── DeviceManager.detect() → CPU/CUDA/MPS/MLX/ANE
   ├── PrecisionManager → per-model dtype selection
   └── MemoryManager → budget allocation

3. Layer 3 — Execution
   ├── InferenceRuntime.initialize()
   ├── WorkerPool.start()
   └── DynamicBatcher.start()

4. Layer 5 — Scalability
   ├── HealthChecker + probes (database, memory, runtime)
   ├── CircuitBreaker instances
   └── RateLimiter + CacheLayer

5. Layer 2 — Orchestration
   ├── RequestDispatcher.initialize(runtime)
   ├── AdaptiveRouter (complexity → model routing)
   └── PriorityQueueManager

6. Background Warmup
   └── AIEngine + LLM pre-loading (non-blocking)

7. Agent System (6 agents)
8. Middleware Orchestrator
```

## Shutdown Sequence (reverse order)

```
1. Stop agent system
2. Drain execution runtime (worker pool, batcher)
3. Cancel warmup tasks
4. Stop GPU scheduler
5. Unload all models
6. Close cache + storage connections
```

## Module Index

| Module | File | Purpose |
|--------|------|---------|
| **Telemetry (infra/telemetry/)** | | |
| StructuredLogger | `infra/telemetry/logger.py` | JSON logging with context vars |
| Tracer | `infra/telemetry/tracer.py` | OpenTelemetry spans + decorators |
| MetricsCollector | `infra/telemetry/metrics.py` | Prometheus counters/histograms |
| PerformanceProfiler | `infra/telemetry/profiler.py` | Hierarchical span-tree profiling |
| Dashboard | `infra/telemetry/dashboard.py` | Aggregated observability endpoint |
| **Infrastructure (infra/)** | | |
| HealthChecker | `infra/health.py` | Readiness/liveness probes |
| DeviceManager | `infra/hardware/device.py` | Hardware detection + placement |
| PrecisionManager | `infra/hardware/precision.py` | Per-model dtype management |
| MemoryManager | `infra/hardware/memory.py` | Budget allocation + pressure |
| GPUWarmPool | `infra/hardware/gpu_pool.py` | Pre-warmed GPU contexts |
| UnifiedCache | `infra/cache/multi_tier_cache.py` | L1 dict + L2 Redis + L3 SQLite |
| EmbeddingCache | `infra/cache/embedding_cache.py` | Embedding-specific caching |
| **Runtime (infra/runtime/)** | | |
| RequestPipeline | `infra/runtime/pipeline.py` | Stage-based request processing |
| AdaptiveRouter | `infra/runtime/router.py` | Complexity → model routing |
| PriorityQueueManager | `infra/runtime/queue.py` | Priority queue with load shedding |
| RequestDispatcher | `infra/runtime/dispatcher.py` | Single API→execution entry point |
| InferenceRuntime | `infra/runtime/runtime.py` | Central execution coordinator |
| DynamicBatcher | `infra/runtime/batcher.py` | Time/count-triggered batching |
| TokenStreamer | `infra/runtime/streamer.py` | Backpressure-aware streaming |
| WorkerPool | `infra/runtime/worker_pool.py` | Async concurrency control |
| **ML (ml/)** | | |
| UnifiedInferenceEngine | `ml/inference/unified_engine.py` | Unified model inference |
| MLXBackend | `ml/inference/mlx_backend.py` | Apple MLX backend |
| UnifiedPipelineService | `ml/pipeline/unified_pipeline.py` | Content processing pipeline |
| PipelineOrchestrator | `ml/pipeline/orchestrator_v2.py` | Async orchestration |
| TranslationEngine | `ml/translate/engine.py` | IndicTrans2 integration |
| TTSService | `ml/speech/tts/` | Text-to-speech services |
| **Services (services/)** | | |
| AIEngine | `services/chat/engine.py` | AI chat engine with intent routing |
| RAGService | `services/chat/rag.py` | RAG Q&A with BGE-M3 |
| UserProfileService | `services/users/user_profile.py` | User personalization |
| ReviewQueueService | `services/users/review_queue.py` | Content review workflow |
| **Database (db/)** | | |
| Database | `db/database.py` | SQLAlchemy engine & sessions |

## Consolidation Notes

| Old Module | New Module | Status |
|-----------|------------|--------|
| `backend/database.py` | `backend/db/database.py` | Moved |
| `backend/cache/` | `backend/infra/cache/` | Moved |
| `backend/hardware/` | `backend/infra/hardware/` | Moved |
| `backend/telemetry/` | `backend/infra/telemetry/` | Moved |
| `backend/execution/` | `backend/infra/runtime/` | Merged |
| `backend/orchestration/` | `backend/infra/runtime/` | Merged |
| `backend/scalability/` | `backend/infra/` | Merged (health.py) |
| `backend/services/ai_core/` | `backend/services/chat/` | Renamed |
| `backend/services/inference/` | `backend/ml/inference/` | Moved |
| `backend/services/pipeline/` | `backend/ml/pipeline/` | Moved |
| `backend/services/tts/` | `backend/ml/speech/tts/` | Moved |
| `backend/services/translate/` | `backend/ml/translate/` | Moved |
| `core/tracing.py` | `infra/telemetry/tracer.py` | Both initialized; tracer reuses core's OTel provider |
| `core/optimized/` | `infra/hardware/` | Wrapped — hardware layer delegates to core/optimized |

## API Endpoints

| Path | Description |
|------|-------------|
| `/` | Root with architecture info |
| `/health` | Structured health (per-probe results) |
| `/metrics` | Prometheus metrics |
| `/observability` | Full observability dashboard |
| `/api/v2/*` | All domain API endpoints |
| `/docs` | OpenAPI documentation |
