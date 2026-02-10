"""
ShikshaSetu Services Module
============================

Business logic services for the ShikshaSetu platform.

Active Services:
- inference/: LLM and embedding generation (MLX, CoreML, Unified)
- pipeline/: Content processing pipeline (unified, optimized)
- evaluation/: Semantic accuracy evaluation
- cultural_context.py: Cultural context adaptation for Indian regions
- ocr.py: Document text extraction (GOT-OCR2)
- rag.py: Retrieval augmented generation (BGE-M3)
- speech_generator.py, speech_processor.py: Speech processing
- translate/: Translation services (IndicTrans2)
- tts/: Text-to-speech (MMS-TTS, Edge-TTS)
- validate/: Curriculum validation
- simplifier.py: Content simplification
- ai_core/: AI service core
- review_queue.py: Teacher review queue
- student_profile.py: Student personalization

Usage:
    from backend.services import get_inference_engine
    engine = get_inference_engine()
"""


# OCR imports are lazy to avoid fitz dependency issues
def get_ocr_service():
    """Get the OCR service (lazy import to avoid fitz dependency)."""
    from .ocr import OCRService

    return OCRService()


# Lazy imports to avoid circular dependencies
def get_inference_engine():
    """Get the unified inference engine."""
    from .inference import get_inference_engine as _get

    return _get()


def get_pipeline_service():
    """Get the unified pipeline service."""
    from .pipeline import get_pipeline_service as _get

    return _get()


def get_semantic_evaluator():
    """Get the semantic accuracy evaluator."""
    from .evaluation import get_semantic_evaluator as _get

    return _get()


def get_cultural_context():
    """Get the cultural context service."""
    from .cultural_context import UnifiedCulturalContextService

    return UnifiedCulturalContextService()


def get_simplifier():
    """Get the text simplifier service."""
    from .simplifier import TextSimplifier

    return TextSimplifier()


def get_review_queue():
    """Get the response review queue."""
    from .review_queue import get_review_queue as _get

    return _get()


# Alias for backwards compatibility
ReviewQueue = None


def _get_review_queue_class():
    """Get the ReviewQueue class (ResponseReviewQueue)."""
    global ReviewQueue
    if ReviewQueue is None:
        from .review_queue import ResponseReviewQueue

        ReviewQueue = ResponseReviewQueue
    return ReviewQueue


__all__ = [
    "GOTOCR2",
    "ExtractionResult",
    # OCR
    "OCRService",
    "PDFProcessor",
    # Review Queue
    "ReviewQueue",
    "get_cultural_context",
    # Factory functions
    "get_inference_engine",
    "get_ocr_service",
    "get_pipeline_service",
    "get_review_queue",
    "get_semantic_evaluator",
    "get_simplifier",
]
