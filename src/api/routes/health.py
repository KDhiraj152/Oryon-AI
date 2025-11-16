"""Health check endpoints."""
from datetime import datetime, timezone
from fastapi import APIRouter

from ...monitoring import check_system_health

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check."""
    return check_system_health()
