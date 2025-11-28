"""
API Routes for Dynamic Quantization Monitoring and Control.

Provides endpoints to:
- Get current quantization status
- Monitor memory usage
- Adjust quantization settings
- View system metrics
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from backend.core.dynamic_quantization import get_quantization_manager, QuantizationLevel
from backend.core.model_loader import get_model_loader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/quantization", tags=["quantization"])


class QuantizationStatus(BaseModel):
    """Quantization status response."""
    device: str
    active_requests: int
    memory: Dict[str, float]
    capabilities: Dict[str, bool]
    recommended_level: str
    current_models: list


class QuantizationOverride(BaseModel):
    """Quantization override request."""
    force_level: Optional[str] = None  # "fp16", "int8", "int4", "int2", or None for auto
    task_priority: str = "balanced"  # "quality", "balanced", "speed"


@router.get("/status", response_model=QuantizationStatus)
async def get_quantization_status():
    """
    Get current dynamic quantization status.
    
    Returns system metrics, active quantization level, and recommendations.
    """
    try:
        manager = get_quantization_manager()
        status = manager.get_status()
        
        # Get model cache status
        loader = get_model_loader()
        cache_stats = loader.cache.get_stats()
        
        return QuantizationStatus(
            device=status["device"],
            active_requests=status["active_requests"],
            memory=status["memory"],
            capabilities=status["capabilities"],
            recommended_level=status["recommended_level"],
            current_models=cache_stats.get("models", [])
        )
    except Exception as e:
        logger.error(f"Failed to get quantization status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate")
async def calculate_optimal_quantization(
    model_size_params: float = 7.0,
    task_priority: str = "balanced"
):
    """
    Calculate optimal quantization for given parameters.
    
    Args:
        model_size_params: Model size in billions of parameters
        task_priority: "quality", "balanced", or "speed"
    
    Returns:
        Recommended quantization configuration
    """
    try:
        manager = get_quantization_manager()
        config = manager.calculate_optimal_quantization(
            model_size_params=model_size_params,
            task_priority=task_priority
        )
        
        return {
            "level": config.level.value,
            "precision": config.precision,
            "compression_ratio": f"{config.compression_ratio:.1f}x",
            "estimated_memory_gb": config.estimated_memory_gb,
            "config": config.config
        }
    except Exception as e:
        logger.error(f"Failed to calculate quantization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory")
async def get_memory_metrics():
    """
    Get detailed memory metrics.
    
    Returns current system memory usage, process memory, and recommendations.
    """
    try:
        manager = get_quantization_manager()
        metrics = manager.get_memory_metrics()
        
        return {
            "total_gb": metrics.total_gb,
            "available_gb": metrics.available_gb,
            "used_gb": metrics.used_gb,
            "percent_used": f"{metrics.percent_used:.1%}",
            "process_memory_gb": metrics.process_memory_gb,
            "status": _get_memory_status(metrics.percent_used)
        }
    except Exception as e:
        logger.error(f"Failed to get memory metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache")
async def get_cache_stats():
    """
    Get model cache statistics.
    
    Returns information about currently cached models and memory usage.
    """
    try:
        loader = get_model_loader()
        stats = loader.cache.get_stats()
        
        return {
            "current_size_mb": stats["current_size_mb"],
            "max_size_mb": stats["max_size_mb"],
            "utilization": f"{(stats['current_size_mb'] / stats['max_size_mb']) * 100:.1f}%",
            "models_cached": stats["models_cached"],
            "models": stats["models"]
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_model_cache():
    """
    Clear all cached models to free memory.
    
    Use this when system is under memory pressure or for testing.
    """
    try:
        loader = get_model_loader()
        loader.cache.clear()
        
        # Also clear device cache
        from backend.utils.device_manager import get_device_manager
        device_manager = get_device_manager()
        device_manager.empty_cache()
        
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_memory_status(percent_used: float) -> str:
    """Get memory status description."""
    if percent_used < 0.4:
        return "comfortable"
    elif percent_used < 0.6:
        return "moderate"
    elif percent_used < 0.75:
        return "tight"
    elif percent_used < 0.85:
        return "critical"
    else:
        return "emergency"


# Lazy loader singleton import
def get_model_loader():
    """Get model loader instance (lazy import to avoid circular dependency)."""
    from backend.core.model_loader import _global_loader
    if _global_loader is None:
        raise RuntimeError("Model loader not initialized")
    return _global_loader
