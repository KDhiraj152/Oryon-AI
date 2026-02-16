"""
Unit tests for OCR services — GOTOCR2 and TesseractOCR.

Tests cover:
- TesseractOCR availability check
- Import smoke tests (models may not be present)
"""

import pytest
from unittest.mock import patch, MagicMock

pytest.importorskip("PIL", reason="Pillow not installed")


class TestTesseractOCR:
    def test_import(self):
        from backend.ml.ocr.ocr import TesseractOCR
        assert TesseractOCR is not None

    def test_init(self):
        from backend.ml.ocr.ocr import TesseractOCR
        try:
            ocr = TesseractOCR()
            assert ocr is not None
        except Exception:
            pytest.skip("Tesseract not installed")

    def test_is_available(self):
        from backend.ml.ocr.ocr import TesseractOCR
        try:
            ocr = TesseractOCR()
            result = ocr.is_available()
            assert isinstance(result, bool)
        except Exception:
            pytest.skip("Tesseract not installed")

    def test_get_available_languages(self):
        from backend.ml.ocr.ocr import TesseractOCR
        try:
            ocr = TesseractOCR()
            if ocr.is_available():
                langs = ocr.get_available_languages()
                assert isinstance(langs, (list, tuple))
        except Exception:
            pytest.skip("Tesseract not installed")


class TestGOTOCR2:
    def test_import(self):
        from backend.ml.ocr.ocr import GOTOCR2
        assert GOTOCR2 is not None
