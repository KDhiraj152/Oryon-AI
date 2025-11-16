"""API route modules initialization."""
from .health import router as health_router
from .auth import router as auth_router
from .content import router as content_router
from .qa import router as qa_router

__all__ = ["health_router", "auth_router", "content_router", "qa_router"]
