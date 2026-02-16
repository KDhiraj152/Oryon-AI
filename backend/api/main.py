"""
Oryon Main Application — Research-Lab-Grade Architecture
================================================================

Six-layer architecture:
  Layer 6 — Telemetry     (structured logging, tracing, metrics, profiling)
  Layer 5 — Scalability   (circuit breakers, rate limiting, health, cache)
  Layer 4 — Hardware       (device detection, precision, memory, GPU pool)
  Layer 3 — Execution     (runtime, batcher, streamer, worker pool, backends)
  Layer 2 — Orchestration  (pipeline, router, queue, dispatcher)
  Layer 1 — API            (FastAPI routes, middleware, exception handling)

All layers are initialized in dependency order during lifespan startup
and torn down in reverse order during shutdown.

Features:
- Async-first, hardware-aware, horizontally scalable, fully observable
- Native Apple Silicon (M4) optimization via HAL
- Multi-tier caching (L1 memory, L2 Redis)
- Dynamic batching with backpressure-aware streaming
- OpenTelemetry distributed tracing + Prometheus metrics
- Circuit breakers, rate limiting, health probes
- Configurable policy engine for content filtering
- Self-optimizing 6-agent system

FIXES APPLIED:
- M6: Uses lifespan context manager instead of deprecated @app.on_event
- C3: Sequential model loading with memory coordination
- C4: Proper async sleep in startup tasks
"""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from ..core.exceptions import OryonException
from ..services.error_tracking import init_sentry
from ..utils.logging import setup_logging
from .exception_handlers import (
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

# ==================== NEW ARCHITECTURE LAYER IMPORTS ====================
# Lazy-loaded singletons — imported at use site to avoid circular deps
# These factory functions are called during lifespan startup.
def _get_layer_singletons():
    """Import all layer singletons. Called once during startup."""
    from ..infra.runtime.runtime import get_runtime
    from ..infra.hardware.device import get_device_manager
    from ..infra.hardware.memory import get_memory_manager
    from ..infra.hardware.precision import get_precision_manager
    from ..infra.runtime.dispatcher import get_dispatcher
    from ..core.circuit_breaker import get_circuit_breaker
    from ..infra.health import check_database, check_memory, get_health_checker
    from ..infra.telemetry.dashboard import get_observability_dashboard

    return {
        "device_manager": get_device_manager,
        "precision_manager": get_precision_manager,
        "memory_manager": get_memory_manager,
        "runtime": get_runtime,
        "dispatcher": get_dispatcher,
        "health_checker": get_health_checker,
        "check_database": check_database,
        "check_memory": check_memory,
        "get_circuit_breaker": get_circuit_breaker,
        "get_dashboard": get_observability_dashboard,
    }

# ==================== LIFESPAN CONTEXT MANAGER ====================
# FIX M6: Use lifespan instead of deprecated @app.on_event handlers

async def _shutdown_cleanup() -> None:
    """Shutdown cleanup helper — extracted to reduce cognitive complexity."""
    logger.info("Shutting down %s", settings.APP_NAME)

    _cancel_warmup_task()

    try:
        await _stop_gpu_scheduler()
    except Exception as e:
        logger.warning("GPU Scheduler shutdown failed: %s", e)

    _cleanup_memory_coordinator()
    _log_cache_stats()

    # Close cache Redis connections (L2 + L3)
    if hasattr(app.state, "cache") and app.state.cache:
        try:
            await app.state.cache.close()
            logger.info("Cache connections closed")
        except Exception as e:
            logger.warning("Cache close failed: %s", e)

    # Close storage Redis connections
    try:
        from ..core.storage import get_storage

        storage = get_storage()
        await storage.close()
        logger.info("Storage connections closed")
    except Exception as e:
        logger.warning("Storage close failed: %s", e)

    logger.info("Graceful shutdown complete — all layers torn down")

def _safe_init(label: str, initializer: Callable[[], Any], level: str = "warning") -> None:
    """Run an initializer with try/except and log on failure."""
    try:
        initializer()
    except Exception as e:
        msg = f"{label} failed: {e}"
        if level == "error":
            logger.error(msg)
        else:
            logger.warning(msg)

def _init_tracing() -> None:
    from ..core.tracing import init_tracing
    init_tracing()
    # Also initialize the new telemetry layer tracer so all new-layer
    # modules (orchestration, execution, etc.) produce real spans.
    try:
        from ..infra.telemetry.tracer import init_tracing as init_layer_tracing
        init_layer_tracing(service_name="oryon-ai", enabled=True)
    except Exception as e:
        logger.warning("Telemetry tracer init failed (new-layer tracing disabled): %s", e)
    logger.info("OpenTelemetry tracing initialized (core + telemetry layer)")

def _init_circuit_breakers() -> None:
    from ..core.circuit_breaker import (
        get_database_breaker,
        get_ml_breaker,
        get_redis_breaker,
    )
    get_database_breaker()
    get_redis_breaker()
    get_ml_breaker()
    logger.info("Circuit breakers initialized (database, redis, ml_model)")

def _init_sentry_tracking() -> None:
    init_sentry()
    logger.info("Sentry error tracking initialized")

def _init_database() -> None:
    from backend.db.database import init_db
    init_db()
    logger.info("Database initialized successfully")

def _init_gpu_scheduler(app: FastAPI) -> None:
    from ..core.optimized.gpu_pipeline import get_gpu_scheduler
    scheduler = get_gpu_scheduler()
    app.state.gpu_scheduler = scheduler
    logger.info("GPU Pipeline Scheduler initialized (will start with warmup)")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager — initializes all six architectural layers
    in dependency order and tears them down in reverse.

    Startup order:
      1. Core infra    (logging, tracing, sentry, database, memory coordinator)
      2. Layer 6        (telemetry — already active via module-level setup)
      3. Layer 4        (hardware — device detection, precision, memory)
      4. Layer 3        (execution — runtime, worker pool, batcher)
      5. Layer 5        (scalability — circuit breakers, health, cache)
      6. Layer 2        (orchestration — dispatcher wired to runtime)
      7. Legacy systems (agents, middleware orchestrator, GPU scheduler)

    Shutdown order is the reverse.
    """
    import asyncio

    # ==================== STARTUP ====================
    logger.info("Starting %s v3.0.0 — research-lab architecture", settings.APP_NAME)
    logger.info("Environment: %s", settings.ENVIRONMENT)
    logger.info("Debug mode: %s", settings.DEBUG)

    if _POLICY_AVAILABLE:
        _safe_init("Policy banner", print_startup_banner)

    # ── 1. Core Infrastructure ──
    _safe_init("Memory coordinator", _init_memory_coordinator)
    _safe_init("Tracing", _init_tracing)
    _safe_init("Circuit breakers (legacy)", _init_circuit_breakers)
    _safe_init("Sentry", _init_sentry_tracking)
    _validate_jwt_secret()
    _safe_init("Database", _init_database, level="error")
    _safe_init("Optimized components", _init_device_router_and_cache)
    _safe_init("GPU Scheduler", lambda: _init_gpu_scheduler(app))

    # Start memory coordinator background monitor
    if hasattr(app.state, "memory_coordinator"):
        app.state.memory_monitor_task = asyncio.create_task(
            app.state.memory_coordinator.start_monitor(interval=30.0)
        )
        logger.info("Memory coordinator monitor started (30s interval)")

    # ── 2. Layer 4: Hardware Detection ──
    try:
        layers = _get_layer_singletons()

        device_mgr = layers["device_manager"]()
        device_mgr.detect()
        app.state.device_manager = device_mgr
        primary = device_mgr.primary_info
        logger.info(
            f"Layer 4 (Hardware): {primary.device_type.value} — "
            f"{primary.total_memory_bytes / (1024**3):.1f}GB, "
            f"fp16={'✓' if primary.supports_fp16 else '✗'}"
        )

        precision_mgr = layers["precision_manager"]()
        app.state.precision_manager = precision_mgr

        memory_mgr = layers["memory_manager"]()
        app.state.memory_manager_v2 = memory_mgr
        logger.info("Layer 4 (Hardware): precision + memory managers ready")
    except Exception as e:
        logger.warning("Layer 4 (Hardware) init degraded: %s", e)

    # ── 3. Layer 3: Execution Runtime ──
    try:
        runtime = layers["runtime"]()
        await runtime.initialize()
        app.state.runtime = runtime
        logger.info("Layer 3 (Execution): runtime + worker pool + batcher started")
    except Exception as e:
        logger.warning("Layer 3 (Execution) init degraded: %s", e)

    # ── 4. Layer 5: Scalability ──
    try:
        health_checker = layers["health_checker"]()
        health_checker.register("database", layers["check_database"])
        health_checker.register("memory", layers["check_memory"])

        # Register execution health check
        async def _check_runtime():
            from ..infra.health import HealthCheck, HealthStatus
            rt = getattr(app.state, "runtime", None)
            if rt and rt._initialized:
                return HealthCheck(name="runtime", status=HealthStatus.HEALTHY)
            return HealthCheck(
                name="runtime",
                status=HealthStatus.DEGRADED,
                message="Runtime not initialized",
            )

        health_checker.register("runtime", _check_runtime)
        app.state.health_checker = health_checker
        logger.info("Layer 5 (Scalability): health checker with 3 probes registered")
    except Exception as e:
        logger.warning("Layer 5 (Scalability) init degraded: %s", e)

    # ── 5. Layer 2: Orchestration Dispatcher ──
    try:
        dispatcher = layers["dispatcher"]()
        execution_handler = getattr(app.state, "runtime", None)
        await dispatcher.initialize(execution_handler=execution_handler)
        app.state.dispatcher = dispatcher
        logger.info("Layer 2 (Orchestration): dispatcher wired to execution runtime")
    except Exception as e:
        logger.warning("Layer 2 (Orchestration) init degraded: %s", e)

    # Store dashboard factory
    with suppress(Exception):
        app.state.get_dashboard = layers["get_dashboard"]

    # ── 6. Background Model Warmup ──
    if settings.ENVIRONMENT != "test":

        async def _background_warmup():
            """Warmup in background — doesn't block server startup."""
            await asyncio.sleep(2)
            try:
                logger.info("Starting background model warmup...")
                from ..services.chat.engine import get_ai_engine

                engine = get_ai_engine()
                engine._ensure_initialized()
                llm = engine._get_llm_client()
                if llm:
                    logger.info("✓ AIEngine and LLM pre-warmed in background")
            except Exception as e:
                logger.warning("Background warmup failed (will lazy-load): %s", e)

        app.state.warmup_task = asyncio.create_task(_background_warmup())
        logger.info("Background warmup scheduled (non-blocking)")

    # ── 7. Legacy Agent System ──
    _safe_init("Agent system", lambda: _init_agent_system(app))
    if hasattr(app.state, "agent_registry"):
        try:
            await app.state.agent_registry.start_all()
            logger.info("Multi-agent system started (6 agents)")
        except Exception as e:
            logger.warning("Agent system start failed: %s", e)

    # ── 8. Middleware Orchestrator ──
    try:
        from .middleware.orchestrator import initialize_middleware

        registry = getattr(app.state, "agent_registry", None)
        mw = await initialize_middleware(registry)
        app.state.middleware_orchestrator = mw
        logger.info("Middleware orchestrator initialized")
    except Exception as e:
        logger.warning("Middleware orchestrator init failed: %s", e)

    logger.info(
        "All layers operational — "
        "API(L1) ← Orchestration(L2) ← Execution(L3) ← Hardware(L4) "
        "← Scalability(L5) ← Telemetry(L6)"
    )

    yield  # ═══════════ Application runs here ═══════════

    # ==================== SHUTDOWN (reverse order) ====================
    logger.info("Beginning graceful shutdown...")

    # Middleware orchestrator cleanup (flush evaluator, log stats)
    if hasattr(app.state, "middleware_orchestrator") and app.state.middleware_orchestrator:
        try:
            mw = app.state.middleware_orchestrator
            stats = await mw.get_stats()
            logger.info(
                f"Middleware shutdown — "
                f"requests={stats.get('total_requests', 0)}, "
                f"errors={stats.get('total_errors', 0)}, "
                f"cache_hits={stats.get('total_cache_hits', 0)}"
            )
            # Reset singleton so next startup gets a fresh instance
            import backend.api.middleware.orchestrator as _mw_mod

            from .middleware.orchestrator import get_middleware as _get_mw_ref
            _mw_mod._instance = None
            app.state.middleware_orchestrator = None
            logger.info("Middleware orchestrator shut down")
        except Exception as e:
            logger.warning("Middleware shutdown error: %s", e)

    # Stop agents first
    if hasattr(app.state, "agent_registry"):
        try:
            await app.state.agent_registry.stop_all()
            logger.info("Multi-agent system stopped")
        except Exception as e:
            logger.warning("Agent shutdown error: %s", e)

    # Drain execution layer
    if hasattr(app.state, "runtime") and app.state.runtime:
        try:
            await app.state.runtime.shutdown()
            logger.info("Layer 3 (Execution): runtime shut down")
        except Exception as e:
            logger.warning("Runtime shutdown error: %s", e)

    await _shutdown_cleanup()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Research-lab-grade multilingual education AI — "
        "six-layer architecture with hardware-aware execution"
    ),
    version="3.0.0",
    debug=settings.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
app.add_exception_handler(OryonException, exception_handler)  # type: ignore[arg-type]
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
    policy_mode = "unknown"
    if _POLICY_AVAILABLE:
        try:
            engine = get_policy_engine()
            policy_mode = engine.mode.value
        except (RuntimeError, ValueError, OSError):  # service call
            pass

    return {
        "name": settings.APP_NAME,
        "version": "3.0.0",
        "architecture": "six-layer-research-grade",
        "api_version": "v2",
        "docs": "/docs",
        "health": "/health",
        "observability": "/observability",
        "status": "operational",
        "policy_mode": policy_mode,
    }

@app.get("/health", include_in_schema=False)
async def root_health_check():
    """
    Structured health check using Layer 5 HealthChecker.

    Returns per-probe status for database, memory, runtime.
    Falls back to simple OK if HealthChecker not available.
    """
    checker = getattr(app.state, "health_checker", None)
    if checker:
        health = await checker.check()
        return health.to_dict()
    return {"status": "healthy", "version": "3.0.0"}

@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return metrics_endpoint()

@app.get("/observability", include_in_schema=False)
async def observability_dashboard():
    """
    Full observability dashboard — aggregates telemetry from all layers.

    Returns real-time snapshot of: metrics, hardware, orchestration,
    execution, cache, workers, circuit breakers, and profiling data.
    """
    get_dashboard = getattr(app.state, "get_dashboard", None)
    if get_dashboard:
        return await get_dashboard()
    return {"error": "Observability dashboard not available", "status": "degraded"}

# ==================== LIFECYCLE HELPERS ====================

def _init_agent_system(app: FastAPI) -> None:
    """Initialize the multi-agent self-optimization system."""
    from ..agents import (
        AgentRegistry,
        EvaluationAgent,
        HardwareOptimizerAgent,
        ModelExecutionAgent,
        OrchestratorAgent,
        ResourceMonitorAgent,
        SelfImprovementAgent,
    )

    registry = AgentRegistry()

    # Register all agents
    registry.register(ResourceMonitorAgent())
    registry.register(HardwareOptimizerAgent())
    registry.register(ModelExecutionAgent())
    registry.register(EvaluationAgent())
    registry.register(OrchestratorAgent())
    registry.register(SelfImprovementAgent())

    app.state.agent_registry = registry
    app.state.orchestrator_agent = registry._agents.get("orchestrator")
    logger.info("Agent system initialized: %s", list(registry._agents.keys()))

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
    from backend.infra.cache import UnifiedCache

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

    from ..services.chat.rag import get_embedder, get_reranker

    embedder = get_embedder()
    if embedder.is_loaded:
        embedder.unload()
        logger.info("RAG embedder unloaded")

    reranker = get_reranker()
    if reranker.is_loaded:
        reranker.unload()
        logger.info("RAG reranker unloaded")

    from ..ml.speech.tts import unload_mms_tts_service

    unload_mms_tts_service()
    logger.info("TTS models unloaded")

    # Shutdown translation executor
    from ..ml.translate.engine import shutdown_translation_executor

    shutdown_translation_executor(wait=True)

def _cancel_warmup_task():
    """Cancel warmup task if running."""
    if (hasattr(app.state, "warmup_task") and app.state.warmup_task
            and not app.state.warmup_task.done()):
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
        logger.info("Unloading %s models: %s", len(loaded_models), loaded_models)

    try:
        _unload_models()
    except Exception as e:
        logger.warning("Model unload failed: %s", e)

    coordinator.force_cleanup()
    logger.info("Final memory stats: %s", coordinator.get_memory_stats())

def _log_cache_stats():
    """Log cache statistics."""
    if hasattr(app.state, "cache") and app.state.cache:
        try:
            stats = app.state.cache.get_stats()
            logger.info("Cache stats at shutdown: %s", stats)
        except Exception:
            pass

# Export
__all__ = ["app"]
