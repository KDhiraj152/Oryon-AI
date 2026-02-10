#!/usr/bin/env python3
"""
Test server with lifespan to isolate blocking.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Test lifespan - import from main module."""
    import asyncio

    logger.info("Starting lifespan...")

    # Test database init
    try:
        from backend.database import init_db

        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    # Test device router
    try:
        from backend.core.optimized import get_device_router

        dr = get_device_router()
        logger.info(f"Device router: {dr.capabilities.chip_name}")
    except Exception as e:
        logger.error(f"Device router failed: {e}")

    # Test rate limiter initialization
    try:
        from backend.cache import get_redis
        from backend.core.optimized.rate_limiter import UnifiedRateLimitMiddleware

        redis_client = get_redis()
        logger.info(f"Redis client: {redis_client}")
    except Exception as e:
        logger.error(f"Redis/Rate limiter failed: {e}")

    # Test background tasks (model warmup)
    async def test_warmup():
        await asyncio.sleep(0.5)
        logger.info("Test warmup task completed")

    warmup_task = asyncio.create_task(test_warmup())
    logger.info("Background task created")

    logger.info("Lifespan startup complete")
    yield
    logger.info("Lifespan shutdown")


# Create app with lifespan
app = FastAPI(title="ShikshaSetu Test", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Import V2 router
logger.info("Testing V2 router import...")
try:
    from backend.api.routes.v2 import router as v2_router

    app.include_router(v2_router, prefix="/api/v2")
    logger.info("V2 router imported and mounted successfully")
except Exception as e:
    logger.error(f"V2 router import failed: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
