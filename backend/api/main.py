"""
ShikshaSetu Main Application - V2 Optimized API Only
=====================================================

Streamlined application using ONLY the optimized V2 API.
All legacy V1 routes have been consolidated into the v2_api module.

Features:
- Single unified API at /api/v2/*
- Native Apple Silicon (M4) optimization
- Multi-tier caching (L1 memory, L2 Redis)
- Concurrent processing with batching
- No middleware patching - native optimizations
- OpenTelemetry distributed tracing
- Circuit breakers for resilience
- Consistent validation error handling
- API versioning headers
- Configurable policy engine for content filtering

Policy Modes (controlled via ALLOW_UNRESTRICTED_MODE env var):
- RESTRICTED (default): Full curriculum/safety enforcement
- UNRESTRICTED: Educational filters bypassed, system safety active
- EXTERNAL_ALLOWED: Unrestricted + external API calls enabled

FIXES APPLIED:
- M6: Uses lifespan context manager instead of deprecated @app.on_event
- C3: Sequential model loading with memory coordination
- C4: Proper async sleep in startup tasks
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from ..core.exceptions import ShikshaSetuException
from ..services.error_tracking import init_sentry
from ..utils.logging import setup_logging
from .middleware import (
    exception_handler,
    generic_exception_handler,
)
from .unified_middleware import UnifiedMiddleware
from .validation_middleware import register_validation_handlers

# Policy module for configurable content filtering
try:
    from ..policy import get_policy_engine, print_startup_banner

    _POLICY_AVAILABLE = True
except ImportError:
    _POLICY_AVAILABLE = False

    def print_startup_banner():
        pass


# Modular router structure (flattened from v2/)
from .metrics import metrics_endpoint
from .routes import router as v2_router

# Initialize logging
logger = setup_logging()


# ==================== LIFESPAN CONTEXT MANAGER ====================
# FIX M6: Use lifespan instead of deprecated @app.on_event handlers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager - handles startup and shutdown.

    FIXES:
    - M6: Uses modern lifespan context manager
    - C3: Sequential model loading with proper ordering
    - C4: No blocking sleeps in async context
    """
    import asyncio

    # ==================== STARTUP ====================
    logger.info(f"Starting {settings.APP_NAME} v2.0.0")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # Print Policy Engine startup banner
    if _POLICY_AVAILABLE:
        try:
            print_startup_banner()
        except Exception as e:
            logger.warning(f"Policy banner failed: {e}")

    # Initialize Memory Coordinator FIRST
    try:
        _init_memory_coordinator()
    except Exception as e:
        logger.warning(f"Memory coordinator initialization failed: {e}")

    # Initialize OpenTelemetry tracing
    try:
        from ..core.tracing import init_tracing

        init_tracing()
        logger.info("OpenTelemetry tracing initialized")
    except Exception as e:
        logger.warning(
            f"Tracing initialization failed (continuing without tracing): {e}"
        )

    # Initialize circuit breakers
    try:
        from ..core.circuit_breaker import (
            get_database_breaker,
            get_ml_breaker,
            get_redis_breaker,
        )

        get_database_breaker()
        get_redis_breaker()
        get_ml_breaker()
        logger.info("Circuit breakers initialized (database, redis, ml_model)")
    except Exception as e:
        logger.warning(f"Circuit breaker initialization failed: {e}")

    # Initialize Sentry and validate JWT
    try:
        init_sentry()
        logger.info("Sentry error tracking initialized")
    except Exception as e:
        logger.warning(f"Sentry initialization failed: {e}")

    _validate_jwt_secret()

    # Initialize database
    from ..database import init_db

    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # Initialize device router and cache
    try:
        _init_device_router_and_cache()
    except Exception as e:
        logger.warning(f"Optimized components initialization failed: {e}")

    # Initialize GPU Pipeline Scheduler
    try:
        from ..core.optimized.gpu_pipeline import get_gpu_scheduler

        scheduler = get_gpu_scheduler()
        app.state.gpu_scheduler = scheduler
        logger.info("GPU Pipeline Scheduler initialized (will start with warmup)")
    except Exception as e:
        logger.warning(f"GPU Scheduler initialization failed: {e}")

    # Start memory coordinator background monitor
    if hasattr(app.state, "memory_coordinator"):
        app.state.memory_monitor_task = asyncio.create_task(
            app.state.memory_coordinator.start_monitor(
                interval=30.0
            )  # Reduced frequency
        )
        logger.info("Memory coordinator monitor started (30s interval)")

    # Background model warm-up - runs in background thread to avoid blocking
    # This pre-initializes the AIEngine and loads the LLM model
    if settings.ENVIRONMENT != "test":

        async def _background_warmup():
            """Warmup in background - doesn't block server startup."""
            await asyncio.sleep(2)  # Let server fully start first
            try:
                logger.info("Starting background model warmup...")
                # Initialize AIEngine (loads all optimized components)
                from ..services.ai_core.engine import get_ai_engine

                engine = get_ai_engine()
                engine._ensure_initialized()

                # Pre-load LLM model
                llm = engine._get_llm_client()
                if llm:
                    logger.info("✓ AIEngine and LLM pre-warmed in background")
            except Exception as e:
                logger.warning(f"Background warmup failed (will lazy-load): {e}")

        app.state.warmup_task = asyncio.create_task(_background_warmup())
        logger.info("Background warmup scheduled (non-blocking)")

    logger.info("V2 API startup complete - all systems operational")

    yield  # Application runs here

    # ==================== SHUTDOWN ====================
    logger.info(f"Shutting down {settings.APP_NAME}")

    _cancel_warmup_task()

    try:
        await _stop_gpu_scheduler()
    except Exception as e:
        logger.warning(f"GPU Scheduler shutdown failed: {e}")

    _cleanup_memory_coordinator()
    _log_cache_stats()

    logger.info("V2 API shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade multilingual education content processing with AI/ML pipeline - V2 Optimized API",
    version="2.0.0",
    debug=settings.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # FIX M6: Use lifespan context manager
)

# ==================== MIDDLEWARE CONFIGURATION ====================
# OPTIMIZED: Using UnifiedMiddleware for CPU/GPU/ANE optimization
# This single middleware handles: request ID, security headers, timing, rate limiting

# GZip compression for responses > 500 bytes
from starlette.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=6)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
    expose_headers=[
        "X-Process-Time",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-API-Version",
        "X-Request-ID",
    ],
)

# UNIFIED MIDDLEWARE: Optimized for Apple Silicon (CPU/GPU/ANE)
# Replaces multiple stacked middlewares with single efficient dispatch
app.add_middleware(
    UnifiedMiddleware,
    rate_limit_enabled=settings.RATE_LIMIT_ENABLED,
    rate_limit_per_minute=getattr(settings, "RATE_LIMIT_CALLS", 100),
)

logger.info("Optimized middleware configured (GZip + CORS + UnifiedMiddleware)")

# ==================== EXCEPTION HANDLERS ====================
app.add_exception_handler(ShikshaSetuException, exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Register validation error handlers for consistent error format
register_validation_handlers(app)

# ==================== ROUTE REGISTRATION ====================
# V2 Modular API - all endpoints at /api/v2/*
app.include_router(v2_router, prefix="/api/v2")

logger.info("V2 Modular API registered - all endpoints at /api/v2/*")

# ==================== ROOT & METRICS ENDPOINTS ====================


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API info."""
    # Include policy mode in root response
    policy_mode = "unknown"
    if _POLICY_AVAILABLE:
        try:
            engine = get_policy_engine()
            policy_mode = engine.mode.value
        except Exception:
            pass

    return {
        "name": settings.APP_NAME,
        "version": "2.0.0",
        "api_version": "v2",
        "docs": "/docs",
        "health": "/health",
        "status": "operational",
        "policy_mode": policy_mode,
    }


@app.get("/health", include_in_schema=False)
async def root_health_check():
    """Root-level health check for load balancers and orchestrators."""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return metrics_endpoint()


# ==================== LIFECYCLE HELPERS ====================


def _init_memory_coordinator():
    """Initialize global memory coordinator."""
    from ..core.optimized.memory_coordinator import (
        MemoryPressure,
        get_memory_coordinator,
    )

    memory_coordinator = get_memory_coordinator()
    app.state.memory_coordinator = memory_coordinator

    def on_memory_pressure(pressure: MemoryPressure):
        """Handle memory pressure changes."""
        if pressure in (MemoryPressure.CRITICAL, MemoryPressure.EMERGENCY):
            logger.warning(
                f"Memory pressure detected: {pressure.value} - "
                "triggering model eviction"
            )

    memory_coordinator.register_pressure_callback(on_memory_pressure)
    logger.info(
        f"Memory coordinator initialized: "
        f"{memory_coordinator.available_memory_gb:.1f}GB available for models"
    )
    return memory_coordinator


def _init_device_router_and_cache():
    """Initialize device router and unified cache."""
    from ..cache import UnifiedCache
    from ..core.optimized import get_device_router

    # Use singleton - avoids duplicate initialization
    device_router = get_device_router()
    caps = device_router.capabilities
    logger.info(
        f"Device router initialized: {caps.chip_name} ({caps.device_type}) with {caps.memory_gb:.1f}GB memory"
    )

    app.state.device_router = device_router
    app.state.cache = UnifiedCache()
    logger.info("Unified multi-tier cache initialized")


async def _warmup_llm_model(coordinator):
    """Pre-load MLX LLM model.

    FIX: Uses asyncio.to_thread() with ProcessPoolExecutor to avoid blocking the event loop.
    MLX/PyTorch can block the GIL extensively, so we run in subprocess.
    """
    import asyncio
    import time

    start = time.perf_counter()

    if coordinator:
        acquired = await coordinator.acquire("llm")
        if not acquired:
            logger.warning("Could not acquire memory for LLM - skipping warmup")
            return None

    def _load_llm_sync():
        from ..services.inference import get_inference_engine

        return get_inference_engine(auto_load=True)

    # Run blocking model load in thread pool (asyncio.to_thread for Python 3.11+)
    engine = await asyncio.to_thread(_load_llm_sync)
    app.state.inference_engine = engine

    elapsed = time.perf_counter() - start
    logger.info(f"✓ MLX model pre-loaded in {elapsed:.2f}s")
    return engine


async def _warmup_embedder(coordinator):
    """Pre-load RAG embedding model with progressive warmup.

    FIX: Uses asyncio run_in_executor to avoid blocking the event loop.
    """
    import asyncio
    import time

    start = time.perf_counter()

    if coordinator:
        acquired = await coordinator.acquire("embedder")
        if not acquired:
            logger.warning("Could not acquire memory for embedder - skipping warmup")
            return None

    def _load_embedder_sync():
        from ..services.rag import get_embedder

        embedder = get_embedder()
        embedder._load_model()

        # Progressive batch warmup
        warmup_texts = ["warmup test"] * 64
        for batch_size in [1, 8, 32, 64]:
            batch = warmup_texts[:batch_size]
            _ = embedder.encode(batch, batch_size=batch_size)
        return embedder

    # Run blocking model load in thread pool (asyncio.to_thread for Python 3.11+)
    embedder = await asyncio.to_thread(_load_embedder_sync)
    logger.debug("  Embedder warmup completed")

    elapsed = time.perf_counter() - start
    logger.info(f"✓ RAG embedder pre-loaded in {elapsed:.2f}s - ready for fast search")
    return embedder


async def _setup_gpu_scheduler(embedder):
    """Initialize and start GPU Pipeline Scheduler with queues."""
    from ..core.optimized.gpu_pipeline import get_gpu_scheduler

    scheduler = get_gpu_scheduler()

    if hasattr(app.state, "inference_engine") and app.state.inference_engine:
        engine = app.state.inference_engine

        def llm_executor(batch):
            results = []
            for prompt in batch:
                result = engine.generate_sync(prompt, max_tokens=512)
                results.append(result)
            return results

        scheduler.register_queue(
            name="llm",
            executor=llm_executor,
            max_batch_size=4,
            max_wait_ms=50.0,
        )

    if embedder is not None:

        def embed_executor(batch):
            return embedder.encode(batch)

        scheduler.register_queue(
            name="embedding",
            executor=embed_executor,
            max_batch_size=64,
            max_wait_ms=10.0,
        )

    await scheduler.start_all()
    logger.info("✓ GPU Pipeline Scheduler started with queues")


def _validate_jwt_secret():
    """Validate JWT secret key configuration."""
    if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
        msg = (
            "JWT secret key is too short or missing. Use a strong secret (>=32 chars)."
        )
        if settings.ENVIRONMENT == "production":
            logger.error(msg)
            raise RuntimeError(msg)
        logger.warning(msg)


def _unload_models():
    """Gracefully unload all loaded models."""
    if hasattr(app.state, "inference_engine") and app.state.inference_engine:
        app.state.inference_engine.unload()
        logger.info("Inference engine unloaded")

    from ..services.rag import get_embedder, get_reranker

    embedder = get_embedder()
    if embedder.is_loaded:
        embedder.unload()
        logger.info("RAG embedder unloaded")

    reranker = get_reranker()
    if reranker.is_loaded:
        reranker.unload()
        logger.info("RAG reranker unloaded")

    from ..services.tts.mms_tts_service import unload_mms_tts_service

    unload_mms_tts_service()
    logger.info("TTS models unloaded")

    # Shutdown translation executor
    from ..services.translate.engine import shutdown_translation_executor

    shutdown_translation_executor(wait=True)


def _cancel_warmup_task():
    """Cancel warmup task if running."""
    if hasattr(app.state, "warmup_task") and app.state.warmup_task:
        if not app.state.warmup_task.done():
            app.state.warmup_task.cancel()
            logger.info("Warmup task cancelled")


async def _stop_gpu_scheduler():
    """Stop GPU Pipeline Scheduler."""
    if hasattr(app.state, "gpu_scheduler") and app.state.gpu_scheduler:
        await app.state.gpu_scheduler.stop_all()
        logger.info("GPU Pipeline Scheduler stopped")


def _cleanup_memory_coordinator():
    """Cleanup memory coordinator and unload models."""
    if not (hasattr(app.state, "memory_coordinator") and app.state.memory_coordinator):
        return

    coordinator = app.state.memory_coordinator
    loaded_models = list(coordinator._loaded_models.keys())
    if loaded_models:
        logger.info(f"Unloading {len(loaded_models)} models: {loaded_models}")

    try:
        _unload_models()
    except Exception as e:
        logger.warning(f"Model unload failed: {e}")

    coordinator.force_cleanup()
    logger.info(f"Final memory stats: {coordinator.get_memory_stats()}")


def _log_cache_stats():
    """Log cache statistics."""
    if hasattr(app.state, "cache") and app.state.cache:
        try:
            stats = app.state.cache.get_stats()
            logger.info(f"Cache stats at shutdown: {stats}")
        except Exception:
            pass


# Export
__all__ = ["app"]
