# Backend Architecture

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## Framework Selection: FastAPI

FastAPI was selected for specific technical requirements:

1. **Native async support**: Handles concurrent AI inference requests without blocking
2. **Automatic OpenAPI documentation**: Self-documenting API at `/docs`
3. **Pydantic integration**: Request/response validation with minimal boilerplate
4. **High performance**: Comparable to Node.js/Go for I/O-bound operations
5. **Python ecosystem access**: Seamless integration with PyTorch, Transformers, MLX

Built on Starlette with `uvloop` for async I/O, FastAPI delivers performance that rivals Node.js while maintaining access to the rich Python ML ecosystem.

---

## Directory Structure

```
backend/
├── __init__.py
│
├── api/                     # HTTP layer
│   ├── __init__.py
│   ├── main.py              # FastAPI app initialization & lifespan events
│   ├── deps.py              # Shared lazy-loaded singletons (get_ai_engine, get_pipeline, json_dumps)
│   ├── documentation.py     # OpenAPI customization
│   ├── metrics.py           # Prometheus instrumentation
│   ├── exception_handlers.py # Exception handlers (was middleware.py)
│   ├── unified_middleware.py    # Consolidated middleware chain
│   ├── validation_middleware.py # Request validation
│   ├── version_middleware.py    # API versioning headers
│   ├── middleware/          # Middleware orchestration layer
│   │   ├── classifier.py    #   Request classification
│   │   ├── evaluator.py     #   Self-evaluation loop
│   │   ├── latency.py       #   Latency control
│   │   ├── memory.py        #   Memory management
│   │   ├── orchestrator.py  #   Unified facade
│   │   └── pipeline.py      #   Agent pipeline
│   └── routes/
│       ├── __init__.py      # Router consolidation
│       ├── auth.py          # Authentication endpoints
│       ├── batch.py         # Batch processing endpoints
│       ├── chat.py          # Chat & streaming endpoints
│       ├── content.py       # Content processing endpoints
│       ├── health.py        # Health, status & admin endpoints
│       └── agents.py        # Multi-agent system endpoints
│
├── core/                    # Core infrastructure & system-level operations
│   ├── config.py            # Application settings (env-backed)
│   ├── constants.py         # Application-wide constants
│   ├── types.py             # Canonical enums: ModelTier, TaskType, ModelType
│   ├── exceptions.py        # Custom exceptions with retry decorators
│   ├── circuit_breaker.py   # Fault tolerance patterns
│   ├── security.py          # JWT encoding/decoding, password hashing
│   ├── storage.py           # Redis/Memory storage backends
│   ├── policy.py            # Content policy engine
│   ├── model_config.py      # Hot-reloadable model configuration
│   ├── correlation.py       # Request correlation ID logging
│   ├── tracing.py           # OpenTelemetry distributed tracing
│   ├── hal.py               # Hardware abstraction layer
│   └── optimized/           # M4-specific optimizations (lazy-loaded via __getattr__)
│       ├── device_router.py        # MPS/CUDA/CPU routing
│       ├── model_manager.py        # High-performance model lifecycle
│       ├── memory_coordinator.py   # Global memory management
│       ├── memory_pool.py          # Buffer pool management
│       ├── async_optimizer.py      # Async-first patterns
│       ├── gpu_pipeline.py         # GPU queue pipelining
│       ├── core_affinity.py        # P/E core routing
│       ├── apple_silicon.py        # M4-specific optimizations
│       ├── rate_limiter.py         # Unified rate limiting
│       ├── quantization.py         # INT4/INT8 quantization
│       ├── benchmark.py            # Performance benchmarking
│       └── ...                     # (22 submodules total)
│
├── db/                      # Database layer
│   ├── __init__.py
│   └── database.py          # SQLAlchemy engine & session management
│
├── infra/                   # Infrastructure layer
│   ├── __init__.py
│   ├── health.py            # Health checker & probes
│   ├── cache/               # Multi-tier caching infrastructure
│   │   ├── __init__.py      #   Redis client singleton
│   │   ├── multi_tier_cache.py  # L1/L2/L3 cache implementation
│   │   ├── redis_cache.py   #   Redis cache backend
│   │   ├── embedding_cache.py   # Embedding-specific cache (SQLite)
│   │   ├── response_cache.py    # Response cache
│   │   ├── kv_cache.py      #   Key-value cache
│   │   └── fast_serializer.py   # msgpack serialization
│   ├── hardware/            # Hardware management
│   │   ├── device.py        #   Device detection & placement
│   │   ├── precision.py     #   Per-model dtype management
│   │   ├── memory.py        #   Memory budget allocation
│   │   └── gpu_pool.py      #   Pre-warmed GPU contexts
│   ├── telemetry/           # Observability
│   │   ├── __init__.py      #   Logger, metrics, profiler exports
│   │   ├── logger.py        #   JSON logging with context vars
│   │   ├── tracer.py        #   OpenTelemetry spans + decorators
│   │   ├── metrics.py       #   Prometheus counters/histograms
│   │   ├── profiler.py      #   Hierarchical span-tree profiling
│   │   └── dashboard.py     #   Aggregated observability endpoint
│   └── runtime/             # Execution & orchestration
│       ├── backends/        #   Inference backends
│       ├── runtime.py       #   Central execution coordinator
│       ├── batcher.py       #   Time/count-triggered batching
│       ├── streamer.py      #   Backpressure-aware streaming
│       ├── worker_pool.py   #   Async concurrency control
│       ├── pipeline.py      #   Stage-based request processing
│       ├── router.py        #   Complexity → model routing
│       ├── queue.py         #   Priority queue with load shedding
│       └── dispatcher.py    #   Single API→execution entry point
│
├── ml/                      # Machine Learning layer
│   ├── __init__.py
│   ├── inference/           # ML backends
│   │   ├── mlx_backend.py   #   Apple MLX backend
│   │   ├── coreml_backend.py #  CoreML backend
│   │   └── unified_engine.py #  Unified inference engine
│   ├── pipeline/            # Content processing pipeline
│   │   ├── unified_pipeline.py  # Main pipeline service
│   │   ├── orchestrator.py  #   Pipeline orchestration
│   │   ├── orchestrator_v2.py # Async orchestrator
│   │   └── model_collaboration.py # Model collaboration
│   ├── speech/              # Speech services
│   │   ├── tts/             #   Text-to-speech (Edge TTS, MMS-TTS)
│   │   ├── speech_generator.py  # Speech generation
│   │   └── speech_processor.py  # Speech processing
│   ├── translate/           # Translation services
│   │   ├── engine.py        #   IndicTrans2 engine
│   │   └── service.py       #   Translation service
│   ├── ocr/                 # OCR services
│   │   └── ocr.py           #   GOT-OCR2 integration
│   └── evaluation/          # Evaluation & refinement
│       └── semantic_accuracy/   # Semantic accuracy evaluation
│
├── services/                # Business logic services
│   ├── __init__.py
│   ├── error_tracking.py    # Sentry integration
│   ├── chat/                # Chat & AI engine
│   │   ├── engine.py        #   AI engine (intent, routing, safety)
│   │   ├── rag.py           #   RAG Q&A with BGE-M3 embeddings
│   │   ├── prompts.py       #   Prompt templates
│   │   ├── citations.py     #   Citation extraction
│   │   └── context.py       #   Context management
│   ├── content/             # Content services
│   │   ├── simplify/        #   Content simplification
│   │   ├── validate/        #   Curriculum validation
│   │   ├── safety_pipeline.py   # 3-pass safety verification
│   │   ├── cultural_context.py  # Indian cultural context adaptation
│   │   └── complexity_adaptation.py # Grade-level content adaptation
│   └── users/               # User services
│       ├── user_profile.py  #   User personalization
│       └── review_queue.py  #   Content review workflow
│
├── models/                  # SQLAlchemy ORM definitions
│   ├── auth.py              # User, APIKey, Token models
│   ├── chat.py              # Conversation, Message models
│   ├── content.py           # ProcessedContent, Translation, Audio
│   ├── progress.py          # UserProgress, Quiz, Achievement
│   ├── rag.py               # DocumentChunk, Embedding, ChatHistory
│   └── user_profile.py      # UserProfile, UserPreference
│
├── schemas/                 # Pydantic request/response models
│   ├── auth.py              # Auth DTOs (UserCreate, Token, etc.)
│   ├── content.py           # Content DTOs (ProcessRequest, etc.)
│   └── qa.py                # Q&A DTOs (QAQueryRequest, etc.)
│
├── agents/                  # Multi-agent system
│   ├── base.py              # BaseAgent protocol & registry
│   ├── orchestrator.py      # Request routing & coordination
│   ├── model_execution.py   # ML model lifecycle & inference
│   ├── hardware_optimizer.py # Dynamic hardware tuning
│   ├── evaluation.py        # Quality measurement
│   ├── resource_monitor.py  # Memory, GPU, latency tracking
│   └── self_improvement.py  # Closed-loop optimization
│
├── tasks/                   # Celery background tasks
│   ├── celery_app.py        # Celery application instance
│   ├── celery_config.py     # Worker configuration
│   └── *_tasks.py           # Domain-specific async tasks
│
└── utils/                   # Shared utilities
    ├── auth.py              # Auth helpers (get_current_user)
    ├── logging.py           # Structured logging setup
    ├── hashing.py           # Hashing utilities
    ├── lock_factory.py      # Injectable lock factory for DI
    ├── cancellation.py      # Task cancellation
    └── memory_guard.py      # Memory guard utilities
```

---

## API Layer Implementation

### Application Initialization (`main.py`)

The FastAPI application uses the modern `lifespan` context manager pattern:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize connections and load initial models
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Initialize Memory Coordinator first
    _init_memory_coordinator()

    # Initialize device router and cache
    _init_device_router_and_cache()

    # Preload essential models
    await _preload_models()

    yield  # Application runs here

    # Shutdown: Graceful cleanup
    cleanup_rag_models()
    await database.disconnect()
```

This ensures all resources are properly initialized before accepting requests and cleanly released during shutdown.

### Route Organization

Endpoints are organized by domain in the `routes/` directory:

| File | Prefix | Responsibility |
|------|--------|----------------|
| `auth.py` | `/api/v2/auth` | Registration, login, token refresh |
| `chat.py` | `/api/v2/chat` | Streaming and non-streaming conversations |
| `content.py` | `/api/v2/content` | Simplification, translation, TTS |
| `batch.py` | `/api/v2/batch` | Bulk processing operations |
| `health.py` | `/api/v2/health` | Liveness, readiness probes, admin |
| `agents.py` | `/api/v2/agents` | Multi-agent system endpoints |

### Middleware Pipeline

Every request passes through a consolidated middleware chain:

```python
# Order matters - executed in sequence
app.add_middleware(UnifiedMiddleware)  # Combines multiple middleware
app.add_middleware(CORSMiddleware, ...)  # Cross-origin requests
```

The `UnifiedMiddleware` handles:

1. **Request ID Assignment**: UUID for tracing
2. **Timing Headers**: `X-Process-Time` in responses
3. **Rate Limiting**: Redis-backed request counters
4. **Circuit Breaking**: Fail-fast when AI models are overloaded
5. **Request Logging**: Structured logging for debugging

The circuit breaker prevents cascade failures. If the model throws 5 consecutive errors, the breaker trips and returns 503 for 30 seconds, allowing the system time to recover.

---

## Core Layer: System-Level Components

### Configuration (`core/config.py`)

Strongly-typed configuration using a Settings class:

```python
class Settings:
    """Application settings with optimal 2025 model stack."""

    # Application
    APP_NAME: str = "Oryon AI Platform API"
    APP_VERSION: str = "4.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")

    # Device & Compute
    DEVICE: str = os.getenv("DEVICE", "auto")  # auto | cuda | mps | cpu
    USE_QUANTIZATION: bool = True
    QUANTIZATION_TYPE: str = "int4"  # int4 | int8 | fp16
    MAX_GPU_MEMORY_GB: float = 16.0

    # Model Stack (2025 Optimal)
    SIMPLIFICATION_MODEL_ID: str = "mlx-community/Qwen3-8B-4bit"
    TRANSLATION_MODEL_ID: str = "ai4bharat/indictrans2-en-indic-1B"
    EMBEDDING_MODEL_ID: str = "BAAI/bge-m3"
    RERANKER_MODEL_ID: str = "BAAI/bge-reranker-v2-m3"

    # Supported Languages
    SUPPORTED_LANGUAGES: list[str] = [
        "Hindi", "Tamil", "Telugu", "Bengali",
        "Marathi", "Gujarati", "Kannada", "Malayalam",
        "Punjabi", "Odia"
    ]
```

### Memory Coordinator (`core/optimized/memory_coordinator.py`)

The Global Memory Coordinator orchestrates multiple AI models in unified memory:

```python
class MemoryCoordinator:
    """Coordinates memory across all loaded AI models."""

    def __init__(self, budget_gb: float = 12.0):
        self.budget_bytes = budget_gb * 1024**3
        self.models: Dict[str, RegisteredModel] = {}
        self.lock = threading.Lock()

    def get_memory_pressure(self) -> MemoryPressure:
        """Returns current memory pressure level."""
        usage = self._get_current_usage()
        ratio = usage / self.budget_bytes

        if ratio < 0.7:
            return MemoryPressure.NORMAL
        elif ratio < 0.85:
            return MemoryPressure.HIGH
        elif ratio < 0.95:
            return MemoryPressure.CRITICAL
        else:
            return MemoryPressure.EMERGENCY

    @contextmanager
    def acquire_memory_sync(self, model_name: str, size_gb: float, priority: int):
        """Synchronously acquire memory, evicting if necessary."""
        required = size_gb * 1024**3

        while self._get_current_usage() + required > self.budget_bytes:
            if not self._evict_lowest_priority():
                raise MemoryError(f"Cannot allocate {size_gb}GB for {model_name}")

        yield

    def register_model(self, name: str, model: Any, unload_fn: Callable):
        """Register a model for lifecycle management."""
        self.models[name] = RegisteredModel(
            model=model,
            unload_fn=unload_fn,
            last_access=time.time(),
            priority=1
        )

    def touch_model(self, name: str):
        """Update last access time for LRU tracking."""
        if name in self.models:
            self.models[name].last_access = time.time()
```

### Device Router (`core/optimized/device_router.py`)

Routes operations to optimal compute units:

```python
class DeviceRouter:
    """Routes AI operations to optimal hardware."""

    def __init__(self):
        self.capabilities = self._detect_capabilities()

    def route(self, task_type: TaskType) -> RoutingDecision:
        """Determine optimal device for task."""

        if task_type == TaskType.EMBEDDING:
            # Neural Engine preferred for embeddings
            if self.capabilities.has_ane:
                return RoutingDecision(device="ane", estimated_speedup=2.0)
            return RoutingDecision(device="mps", estimated_speedup=1.5)

        elif task_type == TaskType.LLM_INFERENCE:
            # GPU preferred for attention operations
            if self.capabilities.device_type == "mps":
                return RoutingDecision(device="mps", estimated_speedup=3.0)
            elif self.capabilities.device_type == "cuda":
                return RoutingDecision(device="cuda", estimated_speedup=4.0)
            return RoutingDecision(device="cpu", estimated_speedup=1.0)
```

---

## Service Layer: Business Logic

### RAG Service (`services/rag.py`)

The core retrieval-augmented generation implementation:

```python
class RAGService:
    """Retrieval-Augmented Generation with multilingual support."""

    def __init__(self):
        self.embedder = get_embedder()  # BGE-M3 singleton
        self.reranker = get_reranker()  # BGE-Reranker singleton

    async def query(
        self,
        question: str,
        top_k: int = 5,
        language: str = "English"
    ) -> RAGResponse:
        """Execute RAG query with reranking."""

        # 1. Embed the query
        query_embedding = self.embedder.encode_query(question)

        # 2. Vector search with pgvector
        candidates = await self._vector_search(
            query_embedding,
            limit=top_k * 3  # Over-retrieve for reranking
        )

        # 3. Rerank candidates
        if candidates:
            reranked = self.reranker.rerank(
                question,
                [c.text for c in candidates],
                top_k=top_k
            )
            candidates = [candidates[idx] for idx, _ in reranked]

        # 4. Build context
        context = self._build_context(candidates)

        return RAGResponse(
            query=question,
            context=context,
            sources=candidates
        )
```

### BGE-M3 Embedder

Optimized for Apple Silicon with memory coordination:

```python
class BGEM3Embedder:
    """BGE-M3 embedding model - best multilingual retrieval."""

    def __init__(self):
        self.model_id = settings.EMBEDDING_MODEL_ID
        self.dimension = 1024  # BGE-M3 fixed dimension
        self._model = None
        self._memory_registered = False

        # Use hardware optimizer for device routing
        router = get_device_router()
        routing = router.route(TaskType.EMBEDDING)
        self.device = routing.device_str

    def _load_model(self):
        """Lazy load with memory coordination."""
        if self._model is not None:
            return

        coordinator = get_memory_coordinator()

        with coordinator.acquire_memory_sync("bgem3_embedder", 1.5, priority=2):
            # On Apple Silicon, prefer sentence-transformers
            # FlagEmbedding has MPS watermark ratio bugs
            if self.device in ("mps", "cpu"):
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_id,
                    device=self.device,
                    trust_remote_code=True
                )
            else:
                from FlagEmbedding import BGEM3FlagModel
                self._model = BGEM3FlagModel(
                    self.model_id,
                    use_fp16=True,
                    device=self.device
                )

            coordinator.register_model("bgem3_embedder", self._model, self.unload)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts with GPU coordination and circuit breaker."""
        self._load_model()

        # Touch model in coordinator for LRU
        get_memory_coordinator().touch_model("bgem3_embedder")

        # Use GPU lock to prevent Metal conflicts with MLX
        from .inference import run_on_gpu_sync

        def _do_encode():
            return self._model.encode(texts, batch_size=batch_size)

        return get_ml_breaker().call_sync(lambda: run_on_gpu_sync(_do_encode))
```

### Translation Service (`services/translate/`)

IndicTrans2 integration for Indian language translation:

```python
class TranslationService:
    """Translation service using IndicTrans2."""

    async def translate_async(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate text between supported languages."""

        # Get or create translator
        translator = await self._get_translator()

        # Execute translation
        result = await translator.translate_async(
            text,
            src_lang=source_lang,
            tgt_lang=target_lang
        )

        return result
```

### TTS Services (`services/tts/`)

Dual TTS implementation with fallback:

```python
class EdgeTTSService:
    """Microsoft Edge TTS service - high quality, network required."""

    async def synthesize(
        self,
        text: str,
        language: str = "en",
        voice: str = None
    ) -> bytes:
        """Generate speech audio."""
        voice = voice or self._get_voice_for_language(language)

        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        return audio_data


class MMSTTSService:
    """Meta MMS TTS - local inference, no network required."""

    def synthesize(self, text: str, language: str = "eng") -> bytes:
        """Generate speech audio locally."""
        self._load_model()

        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model(**inputs)

        return self._to_audio_bytes(output.waveform)
```

---

## Database Layer

### PostgreSQL with pgvector

Vector similarity search using HNSW indexes:

```python
# Alembic migration for vector support
def upgrade():
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create embeddings table with vector column
    op.create_table(
        "document_embeddings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.id")),
        sa.Column("chunk_text", sa.Text),
        sa.Column("embedding", Vector(1024)),  # BGE-M3 dimension
        sa.Column("created_at", sa.DateTime, default=func.now())
    )

    # Create HNSW index for fast similarity search
    op.execute("""
        CREATE INDEX embedding_hnsw_idx
        ON document_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
```

### Multi-Tier Cache

Three-tier caching with msgpack serialization:

```python
class MultiTierCache:
    """L1 (memory) → L2 (Redis) → L3 (disk) cache."""

    async def get(self, key: str) -> Optional[Any]:
        # L1: In-memory cache
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2: Redis
        cached = await self.redis.get(key)
        if cached:
            value = msgpack.unpackb(cached)
            self.l1_cache[key] = value
            return value

        # L3: Disk
        disk_path = self.disk_dir / f"{key}.msgpack"
        if disk_path.exists():
            value = msgpack.unpackb(disk_path.read_bytes())
            await self.redis.set(key, msgpack.packb(value))
            self.l1_cache[key] = value
            return value

        return None
```

---

## Error Handling

Consistent error handling with custom exceptions:

```python
class OryonException(Exception):
    """Base exception for application errors."""

    def __init__(self, message: str, code: str, status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code


@app.exception_handler(OryonException)
async def oryon_exception_handler(request: Request, exc: OryonException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.code,
            "message": exc.message,
            "request_id": request.state.request_id
        }
    )
```

---

*For frontend implementation details, see Section 4: Frontend Architecture.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com
