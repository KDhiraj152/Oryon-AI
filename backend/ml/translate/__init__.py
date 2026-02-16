"""Translation using IndicTrans2-1B - Best-in-class for Indian languages."""

from .engine import TranslatedText, TranslationEngine
from .model import IndicTrans2, TranslationResult, get_translator
from .service import TranslationService

__all__ = [
    "IndicTrans2",
    "TranslatedText",
    "TranslationEngine",
    "TranslationResult",
    "TranslationService",
    "get_translator",
]
