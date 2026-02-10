"""
Text-to-Speech Module
=====================

TTS services for Indian languages and multilingual content.

Services:
- edge_tts: Microsoft Edge TTS (400+ neural voices, free)
- mms_tts: Facebook MMS-TTS (1100+ languages, local inference)

Usage:
    from backend.services.tts import EdgeTTSService, MMSTTSService
"""

from .edge_tts import EdgeTTSService
from .mms_tts import MMSTTSService


def get_edge_tts_service():
    """Get Edge TTS service instance."""
    return EdgeTTSService()


def get_mms_tts_service():
    """Get MMS-TTS service instance."""
    return MMSTTSService()


__all__ = [
    "EdgeTTSService",
    "MMSTTSService",
    "get_edge_tts_service",
    "get_mms_tts_service",
]
