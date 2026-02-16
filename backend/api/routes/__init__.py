"""API route modules initialization - Modular API.

All endpoints organized by domain:
- auth.py: Authentication endpoints
- chat.py: Chat and conversation endpoints
- content.py: Content processing, TTS, OCR, embeddings
- batch.py: Hardware-optimized batch processing
- health.py: Health, monitoring, admin, profile

Optimized for:
- Native Apple Silicon (M4) with MPS/ANE acceleration
- Multi-tier caching (L1 memory, L2 Redis, L3 SQLite)
- Concurrent processing with batching
"""

from fastapi import APIRouter

from .agents import router as agents_router
from .auth import router as auth_router
from .batch import router as batch_router
from .chat import router as chat_router
from .content import router as content_router
from .health import router as health_router
from .middleware import router as middleware_router

# Consolidated router - replaces v2_router
router = APIRouter()
router.include_router(auth_router, prefix="/auth")
router.include_router(chat_router, prefix="/chat")
router.include_router(content_router, prefix="/content")
router.include_router(batch_router, prefix="/batch")
router.include_router(health_router, prefix="/health")
router.include_router(agents_router, prefix="/agents")
router.include_router(middleware_router, prefix="/middleware")

# Backwards compatibility alias
v2_router = router

__all__ = [
    "auth_router",
    "batch_router",
    "chat_router",
    "content_router",
    "health_router",
    "router",
    "v2_router",
]
