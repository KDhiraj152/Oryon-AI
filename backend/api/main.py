"""
ShikshaSetu Main Application - Streamlined and modular.

This is the new main application file that uses the refactored structure:
- Route modules in backend/api/routes/
- Core configuration in backend/core/
- Centralized logging in backend/utils/logging_config.py
- Middleware in backend/api/middleware.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from ..core.exceptions import ShikshaSetuException
from ..utils.logging import setup_logging
from ..services.error_tracking import init_sentry
from .middleware import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    RequestTimingMiddleware,
    RequestLoggingMiddleware,
    exception_handler,
    generic_exception_handler
)
from .sentry_middleware import SentryContextMiddleware
from ..core.rate_limiter import RateLimitMiddleware
from ..cache import get_redis
from .routes import health_router, auth_router, content_router, qa_router, streaming_router
from .routes.experiments import router as experiments_router
from .routes.admin import router as admin_router
from .routes.quantization import router as quantization_router
from .metrics import metrics_endpoint

# Initialize logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade multilingual education content processing with AI/ML pipeline",
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Add custom middleware (order matters - they execute in reverse order of addition)
# Request ID middleware (added first to generate ID before other middleware needs it)
app.add_middleware(RequestIDMiddleware)

# Sentry context middleware (after Request ID)
app.add_middleware(SentryContextMiddleware)

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Timing middleware
app.add_middleware(RequestTimingMiddleware)

# Logging middleware (only in development)
if settings.ENVIRONMENT != "production":
    app.add_middleware(RequestLoggingMiddleware)

# Add CORS middleware (added last to apply first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
    expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
)

# Add rate limiting middleware (production ready)
if settings.RATE_LIMIT_ENABLED:
    try:
        redis_client = get_redis()
        app.add_middleware(RateLimitMiddleware, redis_client=redis_client)
        logger.info("Rate limiting enabled with Redis backend")
    except Exception as e:
        logger.warning(f"Failed to initialize Redis for rate limiting, using in-memory: {e}")
        app.add_middleware(RateLimitMiddleware, redis_client=None)
        logger.info("Rate limiting enabled with in-memory backend")

# Exception handlers
app.add_exception_handler(ShikshaSetuException, exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include route modules
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(content_router)
app.include_router(qa_router)
app.include_router(streaming_router)
app.include_router(experiments_router)
app.include_router(admin_router)
app.include_router(quantization_router)

# Add Prometheus metrics endpoint
@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi import Request
    return await metrics_endpoint(Request)


@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize device manager and configure MPS if available
    try:
        from ..utils.device_manager import get_device_manager
        device_manager = get_device_manager()
        
        if device_manager.device == "mps":
            device_manager.configure_mps_environment()
            logger.info("Apple Silicon MPS environment configured")
        
        logger.info(f"Device: {device_manager.device_name} ({device_manager.device})")
    except Exception as e:
        logger.warning(f"Device manager initialization: {e}")
    
    # Initialize model tier router
    try:
        from ..core.model_tier_router import init_router
        router = init_router(
            max_memory_gb=settings.MAX_MODEL_MEMORY_GB,
            device_type=device_manager.device if 'device_manager' in locals() else "cpu"
        )
        logger.info(f"Model tier router initialized (max memory: {settings.MAX_MODEL_MEMORY_GB}GB)")
    except Exception as e:
        logger.warning(f"Model router initialization: {e}")
    
    # Initialize Sentry error tracking
    try:
        init_sentry()
        logger.info("Sentry error tracking initialized")
    except Exception as e:
        logger.warning(f"Sentry initialization failed: {e}")
    
    # Validate JWT secret strength in non-dev environments
    try:
        if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
            msg = "JWT secret key is too short or missing. Use a strong secret (>=32 chars)."
            if settings.ENVIRONMENT == "production":
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                logger.warning(msg)
    except Exception as e:
        logger.exception(f"Startup validation error: {e}")
        raise
    
    # Initialize database
    from ..database import init_db
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info(f"Shutting down {settings.APP_NAME}")


# For backward compatibility, expose the app as the default export
__all__ = ["app"]
