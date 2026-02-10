"""
API Version Middleware - Version Management and Compatibility
==============================================================

Provides:
- API version information
- Version compatibility checking
- Deprecation notices
- Migration guidance
"""

from datetime import datetime, timedelta
from typing import Any

# Current API version
CURRENT_VERSION = "v2"

# Version configuration
# NOTE: v1 sunset date is fixed, not computed dynamically
API_VERSIONS = {
    "v1": {
        "status": "deprecated",
        "deprecated": True,
        "sunset": "2026-08-10T00:00:00",
        "description": "Legacy API - deprecated in favor of v2",
        "migration_guide": "/docs/migration-v1-to-v2",
    },
    "v2": {
        "status": "current",
        "deprecated": False,
        "sunset": None,
        "description": "Current production API with optimized endpoints",
        "features": [
            "streaming_sse",
            "batch_processing",
            "multi_language",
            "rag_enabled",
        ],
    },
}


def get_api_version_info() -> dict[str, Any]:
    """
    Get API version information.

    Returns:
        Dict with version details including current version and all versions
    """
    return {
        "current_version": CURRENT_VERSION,
        "versions": API_VERSIONS,
        "recommended": CURRENT_VERSION,
    }


def check_api_version(version: str) -> dict[str, Any]:
    """
    Check if an API version is supported.

    Args:
        version: Version string to check (e.g., "v1", "v2")

    Returns:
        Dict with support status and details
    """
    version_lower = version.lower()

    if version_lower in API_VERSIONS:
        info = API_VERSIONS[version_lower]
        return {
            "supported": True,
            "status": info["status"],
            "deprecated": info.get("deprecated", False),
            "sunset": info.get("sunset"),
            "description": info.get("description", ""),
            "migration_guide": info.get("migration_guide"),
        }
    else:
        return {
            "supported": False,
            "status": "unsupported",
            "recommended": CURRENT_VERSION,
            "message": f"Version {version} is not supported. Use {CURRENT_VERSION}.",
        }


def get_version_headers(version: str = CURRENT_VERSION) -> dict[str, str]:
    """
    Get headers to include in API responses for version info.

    Args:
        version: Current API version being used

    Returns:
        Dict of headers to add to response
    """
    headers = {
        "X-API-Version": version,
    }

    if version in API_VERSIONS:
        info = API_VERSIONS[version]
        if info.get("deprecated"):
            headers["X-API-Deprecated"] = "true"
            if info.get("sunset"):
                headers["X-API-Sunset"] = info["sunset"]
            headers["X-API-Recommended-Version"] = CURRENT_VERSION

    return headers


__all__ = [
    "API_VERSIONS",
    "CURRENT_VERSION",
    "check_api_version",
    "get_api_version_info",
    "get_version_headers",
]
