"""OCR service for PDF and image text extraction with formula preservation."""
import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of text extraction."""
    text: str
    num_pages: int
    has_images: bool
    has_formulas: bool
    formula_blocks: List[Dict]
    metadata: Dict
    confidence: float


class MathFormulaDetector:
    """Detect and preserve mathematical formulas."""
    
    # Math symbol patterns
    MATH_PATTERNS = [
        r'[∫∑∏√∂∇∆]',  # Calculus and operators
        r'[≤≥≠≈±∓×÷]',  # Comparison and arithmetic
        r'[αβγδεζηθλμπρσφψω]',  # Greek letters
        r'[⊕⊗⊥∥∠∡∢]',  # Geometry
        r'\^[0-9]+|\_{0-9}+',  # Superscript/subscript
        r'\\[a-zA-Z]+\{[^}]*\}',  # LaTeX commands
        r'\$[^\$]+\$',  # Inline LaTeX
        r'\\\[[^\]]+\\\]',  # Display LaTeX
    ]
    
    @classmethod
    def contains_math(cls, text: str) -> bool:
        """Check if text contains mathematical notation."""
        for pattern in cls.MATH_PATTERNS:
            if re.search(pattern, text):
                return True
        return False
    
    @classmethod
    def extract_formulas(cls, text: str) -> List[Dict]:
        """Extract formula blocks with positions."""
        formulas = []
        
        # LaTeX display equations
        for match in re.finditer(r'\\\[([^\]]+)\\\]', text):
            formulas.append({
                'type': 'latex_display',
                'content': match.group(1),
                'start': match.start(),
                'end': match.end(),
                'original': match.group(0)
            })
        
        # LaTeX inline equations
        for match in re.finditer(r'\$([^\$]+)\$', text):
            formulas.append({
                'type': 'latex_inline',
                'content': match.group(1),
                'start': match.start(),
                'end': match.end(),
                'original': match.group(0)
            })
        
        # Unicode math symbols
        for pattern in cls.MATH_PATTERNS[:6]:  # Symbol patterns only
            for match in re.finditer(pattern, text):
                formulas.append({
                    'type': 'unicode_math',
                    'content': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'original': match.group(0)
                })
        
        return formulas
    
    @classmethod
    def preserve_formulas(cls, text: str) -> Tuple[str, List[Dict]]:
        """Replace formulas with placeholders and return mapping."""
        formulas = cls.extract_formulas(text)
        processed_text = text
        
        # Replace formulas with placeholders (reverse order to maintain positions)
        for i, formula in enumerate(reversed(formulas)):
            placeholder = f"__FORMULA_{len(formulas) - i - 1}__"
            processed_text = (
                processed_text[:formula['start']] + 
                placeholder + 
                processed_text[formula['end']:]
            )
            formula['placeholder'] = placeholder
        
        return processed_text, list(reversed(formulas))
    
    @classmethod
    def restore_formulas(cls, text: str, formulas: List[Dict]) -> str:
        """Restore formulas from placeholders."""
        restored_text = text
        for formula in formulas:
            restored_text = restored_text.replace(
                formula['placeholder'], 
                formula['original']
            )
        return restored_text


class PDFExtractor:
    """Extract text from PDF files using PyMuPDF."""
    
    def __init__(self):
        self.math_detector = MathFormulaDetector()
    
    def extract_text(self, pdf_path: str) -> ExtractionResult:
        """Extract text from PDF with formula preservation."""
        try:
            doc = fitz.open(pdf_path)
            
            full_text = []
            has_images = False
            formula_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                page_text = page.get_text("text")
                
                # Check for images
                if page.get_images():
                    has_images = True
                
                # Detect and preserve formulas
                if self.math_detector.contains_math(page_text):
                    processed_text, page_formulas = self.math_detector.preserve_formulas(page_text)
                    full_text.append(processed_text)
                    
                    # Add page number to formulas
                    for formula in page_formulas:
                        formula['page'] = page_num + 1
                    formula_blocks.extend(page_formulas)
                else:
                    full_text.append(page_text)
            
            doc.close()
            
            combined_text = '\n\n'.join(full_text)
            
            # Clean text
            cleaned_text = self._clean_text(combined_text)
            
            return ExtractionResult(
                text=cleaned_text,
                num_pages=len(doc),
                has_images=has_images,
                has_formulas=len(formula_blocks) > 0,
                formula_blocks=formula_blocks,
                metadata={
                    'filename': Path(pdf_path).name,
                    'file_size': os.path.getsize(pdf_path),
                    'pages': len(doc)
                },
                confidence=0.95  # High confidence for text-based PDFs
            )
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff')
        
        return text.strip()


class TesseractOCR:
    """Fallback OCR using Tesseract for image-based PDFs."""
    
    # Supported Indian languages
    LANGUAGES = {
        'Hindi': 'hin',
        'English': 'eng',
        'Tamil': 'tam',
        'Telugu': 'tel',
        'Bengali': 'ben',
        'Marathi': 'mar',
        'Gujarati': 'guj',
        'Kannada': 'kan',
        'Malayalam': 'mal',
        'Punjabi': 'pan'
    }
    
    def __init__(self, languages: List[str] = None):
        """Initialize with language support."""
        if languages is None:
            languages = ['English', 'Hindi']
        
        # Build language string for Tesseract
        self.lang_codes = [self.LANGUAGES.get(lang, 'eng') for lang in languages]
        self.lang_string = '+'.join(self.lang_codes)
        
        # Verify Tesseract installation
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise RuntimeError("Tesseract OCR not installed. Run: brew install tesseract")
    
    def extract_from_pdf(self, pdf_path: str) -> ExtractionResult:
        """Extract text from image-based PDF."""
        try:
            doc = fitz.open(pdf_path)
            
            full_text = []
            formula_blocks = []
            total_confidence = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # OCR with confidence
                ocr_data = pytesseract.image_to_data(
                    img,
                    lang=self.lang_string,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'  # Assume uniform block of text
                )
                
                # Extract text and confidence
                page_text = pytesseract.image_to_string(img, lang=self.lang_string)
                
                # Calculate average confidence for this page
                confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
                page_confidence = sum(confidences) / len(confidences) if confidences else 0
                total_confidence += page_confidence
                
                full_text.append(page_text)
            
            doc.close()
            
            combined_text = '\n\n'.join(full_text)
            avg_confidence = total_confidence / len(doc) if len(doc) > 0 else 0
            
            # Check for math symbols
            math_detector = MathFormulaDetector()
            has_formulas = math_detector.contains_math(combined_text)
            
            if has_formulas:
                combined_text, formula_blocks = math_detector.preserve_formulas(combined_text)
            
            return ExtractionResult(
                text=combined_text,
                num_pages=len(doc),
                has_images=True,
                has_formulas=has_formulas,
                formula_blocks=formula_blocks,
                metadata={
                    'filename': Path(pdf_path).name,
                    'file_size': os.path.getsize(pdf_path),
                    'pages': len(doc),
                    'ocr_method': 'tesseract',
                    'languages': self.lang_codes
                },
                confidence=avg_confidence / 100  # Normalize to 0-1
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise
    
    def extract_from_image(self, image_path: str) -> str:
        """Extract text from a single image."""
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=self.lang_string)
            return text.strip()
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            raise


class OCRService:
    """Unified OCR service with automatic fallback."""
    
    def __init__(self, languages: List[str] = None):
        """Initialize OCR service."""
        self.pdf_extractor = PDFExtractor()
        self.tesseract_ocr = TesseractOCR(languages)
        self.math_detector = MathFormulaDetector()
    
    def extract_text(
        self, 
        file_path: str, 
        force_ocr: bool = False
    ) -> ExtractionResult:
        """
        Extract text from PDF or image with automatic method selection.
        
        Args:
            file_path: Path to PDF or image file
            force_ocr: Force Tesseract OCR even for text-based PDFs
            
        Returns:
            ExtractionResult with text and metadata
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            if force_ocr:
                logger.info("Using Tesseract OCR (forced)")
                return self.tesseract_ocr.extract_from_pdf(file_path)
            
            # Try PyMuPDF first
            try:
                result = self.pdf_extractor.extract_text(file_path)
                
                # Check if text extraction was successful
                if len(result.text.strip()) < 100:
                    logger.warning("Low text yield from PyMuPDF, falling back to OCR")
                    return self.tesseract_ocr.extract_from_pdf(file_path)
                
                return result
                
            except Exception as e:
                logger.warning(f"PyMuPDF failed, using Tesseract: {e}")
                return self.tesseract_ocr.extract_from_pdf(file_path)
        
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            # Image file - use Tesseract
            text = self.tesseract_ocr.extract_from_image(file_path)
            
            has_formulas = self.math_detector.contains_math(text)
            formula_blocks = []
            
            if has_formulas:
                text, formula_blocks = self.math_detector.preserve_formulas(text)
            
            return ExtractionResult(
                text=text,
                num_pages=1,
                has_images=True,
                has_formulas=has_formulas,
                formula_blocks=formula_blocks,
                metadata={
                    'filename': Path(file_path).name,
                    'file_size': os.path.getsize(file_path),
                    'ocr_method': 'tesseract'
                },
                confidence=0.85  # Estimated confidence for images
            )
        
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def validate_extraction(self, result: ExtractionResult) -> bool:
        """Validate extraction quality."""
        # Check minimum text length
        if len(result.text.strip()) < 50:
            logger.warning("Extracted text too short")
            return False
        
        # Check confidence threshold
        if result.confidence < 0.60:
            logger.warning(f"Low OCR confidence: {result.confidence:.2f}")
            return False
        
        return True


# Export
__all__ = ['OCRService', 'ExtractionResult', 'MathFormulaDetector']
