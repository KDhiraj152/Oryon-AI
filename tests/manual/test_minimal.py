#!/usr/bin/env python3
"""
Minimal server to isolate blocking in ShikshaSetu.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create minimal app
app = FastAPI(title="ShikshaSetu Test", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware from ShikshaSetu
logger.info("Adding ShikshaSetu middleware...")
try:
    from backend.api.middleware import (
        RequestIDMiddleware,
        RequestLoggingMiddleware,
        RequestTimingMiddleware,
        SecurityHeadersMiddleware,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestTimingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    logger.info("Middleware added successfully")
except Exception as e:
    logger.error(f"Middleware import failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


# Test import each component incrementally
logger.info("Testing V2 router import...")
try:
    from backend.api.routes.v2 import router as v2_router

    app.include_router(v2_router, prefix="/api/v2")
    logger.info("V2 router imported and mounted successfully")
except Exception as e:
    logger.error(f"V2 router import failed: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
