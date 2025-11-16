"""
Simplified monitoring system for pipeline operations.

Replaces: logger.py, metrics_collector.py, alert_manager.py, monitoring_service.py, health_checks.py
Reduction: ~1,200 lines → ~300 lines (75% reduction)
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# =============================================================================
# SIMPLE METRICS TRACKING
# =============================================================================

class MetricType(Enum):
    """Metric types for tracking"""
    PIPELINE_EXECUTION = "pipeline_execution"
    STAGE_EXECUTION = "stage_execution"
    API_REQUEST = "api_request"
    ERROR = "error"


# Simple in-memory metrics storage
_metrics = defaultdict(lambda: {
    'count': 0,
    'total_time_ms': 0,
    'errors': 0,
    'success': 0,
    'last_execution': None
})

_error_log = []  # Simple error tracking


def track_metric(
    metric_type: str,
    name: str,
    duration_ms: float = 0,
    success: bool = True,
    metadata: Optional[Dict] = None
):
    """
    Simple metric tracking function.
    
    Args:
        metric_type: Type of metric (pipeline, stage, api, error)
        name: Metric identifier
        duration_ms: Execution duration in milliseconds
        success: Whether operation succeeded
        metadata: Additional context
    """
    key = f"{metric_type}:{name}"
    _metrics[key]['count'] += 1
    _metrics[key]['total_time_ms'] += duration_ms
    _metrics[key]['last_execution'] = datetime.now().isoformat()
    
    if success:
        _metrics[key]['success'] += 1
    else:
        _metrics[key]['errors'] += 1
        if metadata:
            _error_log.append({
                'type': metric_type,
                'name': name,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata
            })


def get_metrics(metric_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get collected metrics.
    
    Args:
        metric_type: Filter by metric type (optional)
    
    Returns:
        Dictionary of metrics
    """
    if metric_type:
        return {
            k: dict(v) for k, v in _metrics.items()
            if k.startswith(f"{metric_type}:")
        }
    return {k: dict(v) for k, v in _metrics.items()}


def get_metric_summary() -> Dict[str, Any]:
    """Get summary of all metrics"""
    total_executions = sum(m['count'] for m in _metrics.values())
    total_errors = sum(m['errors'] for m in _metrics.values())
    
    return {
        'total_executions': total_executions,
        'total_errors': total_errors,
        'error_rate': total_errors / total_executions if total_executions > 0 else 0,
        'metrics_tracked': len(_metrics),
        'recent_errors': _error_log[-10:] if _error_log else []
    }


def clear_metrics():
    """Clear all metrics (useful for testing)"""
    _metrics.clear()
    _error_log.clear()


# =============================================================================
# HEALTH CHECKS
# =============================================================================

def check_system_health() -> Dict[str, Any]:
    """
    Simple system health check.
    
    Returns:
        Health status dictionary
    """
    health = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {}
    }
    
    # Check database
    try:
        from sqlalchemy import text
        from .database import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health['components']['database'] = {'status': 'up', 'message': 'Connected'}
    except Exception as e:
        health['status'] = 'degraded'
        health['components']['database'] = {'status': 'down', 'error': str(e)}
    
    # Check Redis
    try:
        import redis
        from os import getenv
        client = redis.Redis(
            host=getenv('REDIS_HOST', 'localhost'),
            port=int(getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        client.ping()
        health['components']['redis'] = {'status': 'up', 'message': 'Connected'}
    except Exception as e:
        health['status'] = 'degraded'
        health['components']['redis'] = {'status': 'down', 'error': str(e)}
    
    # Add metrics summary
    health['metrics'] = get_metric_summary()
    
    return health


def check_detailed_health() -> Dict[str, Any]:
    """
    Detailed health check with metrics.
    
    Returns:
        Detailed health status
    """
    health = check_system_health()
    
    # Add detailed metrics
    health['detailed_metrics'] = {
        'pipeline': get_metrics('pipeline_execution'),
        'stages': get_metrics('stage_execution'),
        'api': get_metrics('api_request')
    }
    
    return health


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_pipeline_execution(
    content_id: str,
    language: str,
    grade_level: int,
    success: bool,
    duration_ms: float,
    error: Optional[str] = None
):
    """Log pipeline execution with metrics tracking"""
    logger = get_logger('pipeline')
    
    if success:
        logger.info(f"Pipeline completed: {content_id} ({language}, grade {grade_level}) in {duration_ms}ms")
    else:
        logger.error(f"Pipeline failed: {content_id} - {error}")
    
    track_metric(
        metric_type='pipeline_execution',
        name=f"{language}_grade{grade_level}",
        duration_ms=duration_ms,
        success=success,
        metadata={'content_id': content_id, 'error': error} if error else None
    )


def log_stage_execution(
    stage: str,
    content_id: str,
    success: bool,
    duration_ms: float,
    retry_count: int = 0,
    error: Optional[str] = None
):
    """Log stage execution with metrics tracking"""
    logger = get_logger('pipeline.stage')
    
    if success:
        retry_info = f" (retries: {retry_count})" if retry_count > 0 else ""
        logger.info(f"Stage {stage} completed: {content_id} in {duration_ms}ms{retry_info}")
    else:
        logger.error(f"Stage {stage} failed: {content_id} - {error}")
    
    track_metric(
        metric_type='stage_execution',
        name=stage,
        duration_ms=duration_ms,
        success=success,
        metadata={
            'content_id': content_id,
            'retry_count': retry_count,
            'error': error
        } if error else None
    )


def log_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None
):
    """Log API request with metrics tracking"""
    logger = get_logger('api')
    
    success = 200 <= status_code < 400
    logger.info(f"{method} {endpoint} - {status_code} ({duration_ms}ms)")
    
    track_metric(
        metric_type='api_request',
        name=f"{method}_{endpoint}",
        duration_ms=duration_ms,
        success=success,
        metadata={'status_code': status_code, 'user_id': user_id}
    )


# =============================================================================
# LEGACY COMPATIBILITY (for gradual migration)
# =============================================================================

class MetricsCollector:
    """Legacy compatibility wrapper"""
    
    def record_pipeline_execution(self, **kwargs):
        """Legacy method - use log_pipeline_execution instead"""
        track_metric('pipeline_execution', 'legacy', **kwargs)
    
    def get_summary(self):
        """Legacy method - use get_metric_summary instead"""
        return get_metric_summary()


def get_metrics_collector():
    """Legacy compatibility - use direct functions instead"""
    return MetricsCollector()


# Export commonly used items
__all__ = [
    'track_metric',
    'get_metrics',
    'get_metric_summary',
    'check_system_health',
    'check_detailed_health',
    'get_logger',
    'log_pipeline_execution',
    'log_stage_execution',
    'log_api_request',
    'MetricsCollector',
    'get_metrics_collector'
]
