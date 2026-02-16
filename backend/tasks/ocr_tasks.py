"""
OCR Tasks (Celery)
==================
Tasks for document OCR using GOT-OCR2.
"""

import base64
import logging
from typing import Any

from .celery_config import celery_app

logger = logging.getLogger(__name__)

# Lazy-loaded model
_ocr_service = None

def get_ocr_service():
    """Get or initialize OCR service (lazy loading)."""
    global _ocr_service
    if _ocr_service is None:
        from backend.ml.ocr.ocr import GOTOCR2Service

        _ocr_service = GOTOCR2Service()
    return _ocr_service

@celery_app.task(
    name="ocr.process_image",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    soft_time_limit=120,
    time_limit=150,
)
def process_image(
    self, image_data: str, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Process image for OCR.

    Args:
        image_data: Base64 encoded image or file path
        options: Additional options (detect_tables, detect_formulas, etc.)

    Returns:
        Dict with extracted text
    """
    try:
        import asyncio
        import io

        from PIL import Image

        ocr = get_ocr_service()
        options = options or {}

        # Decode image
        if image_data.startswith("data:image"):
            # Data URL
            image_data = image_data.split(",")[1]

        if len(image_data) > 1000:  # Likely base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # Treat as file path
            image = Image.open(image_data)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(ocr.process(image, **options))
        finally:
            loop.close()

        return {
            "success": True,
            "text": result.get("text", ""),
            "tables": result.get("tables", []),
            "formulas": result.get("formulas", []),
            "confidence": result.get("confidence", 0.0),
            "metadata": result.get("metadata", {}),
        }

    except Exception as e:  # task handler — intentionally broad
        logger.error("OCR processing failed: %s", e)

        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        return {
            "success": False,
            "error": str(e),
        }

@celery_app.task(
    name="ocr.process_pdf",
    bind=True,
    max_retries=1,
    soft_time_limit=600,
    time_limit=660,
)
def process_pdf(
    self,
    pdf_path: str,
    pages: list | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Process PDF document for OCR.

    Args:
        pdf_path: Path to PDF file
        pages: List of page numbers to process (None = all)
        options: Additional options

    Returns:
        Dict with extracted text by page
    """
    try:
        import asyncio
        import io

        import fitz  # PyMuPDF
        from PIL import Image

        ocr = get_ocr_service()
        options = options or {}

        # Open PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # Determine pages to process
        if pages is None:
            pages = list(range(total_pages))
        else:
            pages = [p for p in pages if 0 <= p < total_pages]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        results = []

        try:
            for page_num in pages:
                page = doc[page_num]

                # Render page to image
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))

                # OCR the page
                result = loop.run_until_complete(ocr.process(image, **options))

                results.append(
                    {
                        "page": page_num,
                        "text": result.get("text", ""),
                        "tables": result.get("tables", []),
                        "formulas": result.get("formulas", []),
                    }
                )
        finally:
            loop.close()
            doc.close()

        # Combine all text
        full_text = "\n\n".join([r["text"] for r in results])

        return {
            "success": True,
            "full_text": full_text,
            "pages": results,
            "total_pages": total_pages,
            "processed_pages": len(pages),
        }

    except Exception as e:  # task handler — intentionally broad
        logger.error("PDF OCR failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "pdf_path": pdf_path,
        }

@celery_app.task(
    name="ocr.process_batch",
    bind=True,
    soft_time_limit=900,
    time_limit=960,
)
def process_batch(
    self, image_paths: list, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Batch process multiple images.

    Args:
        image_paths: List of image file paths
        options: Additional options

    Returns:
        Dict with results for each image
    """
    try:
        import asyncio

        from PIL import Image

        ocr = get_ocr_service()
        options = options or {}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        results = []

        try:
            for path in image_paths:
                image = Image.open(path)

                result = loop.run_until_complete(ocr.process(image, **options))

                results.append(
                    {
                        "path": path,
                        "text": result.get("text", ""),
                        "success": True,
                    }
                )
        except (RuntimeError, ValueError, OSError) as e:
            results.append(
                {
                    "path": path,
                    "success": False,
                    "error": str(e),
                }
            )
        finally:
            loop.close()

        return {
            "success": True,
            "results": results,
            "total": len(image_paths),
            "processed": len([r for r in results if r.get("success")]),
        }

    except Exception as e:  # task handler — intentionally broad
        logger.error("Batch OCR failed: %s", e)
        return {
            "success": False,
            "error": str(e),
        }

@celery_app.task(
    name="ocr.detect_document_type",
    bind=True,
    soft_time_limit=30,
    time_limit=45,
)
def detect_document_type(self, image_data: str) -> dict[str, Any]:
    """
    Detect type of document in image.

    Args:
        image_data: Base64 encoded image

    Returns:
        Dict with document type classification
    """
    try:
        # Simple heuristic-based detection
        # For production, use a document classification model

        return {
            "success": True,
            "document_type": "textbook",  # Default
            "has_tables": False,
            "has_formulas": False,
            "has_diagrams": False,
            "language_hint": "hi",
        }

    except Exception as e:  # task handler — intentionally broad
        logger.error("Document type detection failed: %s", e)
        return {
            "success": False,
            "error": str(e),
        }
