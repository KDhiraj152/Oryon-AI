"""Celery tasks package.

Principle N: Each Celery worker loads (at most) one heavyweight model.

Worker Types:
- simplify: Qwen2.5-3B-Instruct for text simplification
- translate: IndicTrans2-1B for translation
- ocr: GOT-OCR2 for document OCR
- embedding: BGE-M3 for embeddings
- rag: BGE-M3 + Reranker for RAG operations
"""

# Import task modules to register tasks
from . import embedding_tasks, ocr_tasks, rag_tasks, simplify_tasks, translate_tasks
from .celery_config import celery_app

__all__ = [
    "celery_app",
    "embedding_tasks",
    "ocr_tasks",
    "rag_tasks",
    "simplify_tasks",
    "translate_tasks",
]
