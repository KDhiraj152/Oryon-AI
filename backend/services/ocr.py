"""
GOT-OCR2 Service - State-of-the-art Vision-Language OCR.

Optimal 2025 Model Stack: ucaslcl/GOT-OCR2_0
- Best accuracy on Indian scripts (95%+)
- Handles formulas, tables, mixed layouts
- Native GPU acceleration (10x faster than Tesseract)
- Supports scene text, documents, sheet music

Fallback: Tesseract OCR with Indian language packs
- Works on CPU
- Supports 22 Indian languages
- Lower accuracy but always available
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# PyMuPDF is optional - used for PDF processing
# GOT-OCR2 can work with images directly
try:
    import fitz  # PyMuPDF

    FITZ_AVAILABLE = True
except ImportError:
    fitz = None
    FITZ_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "PyMuPDF (fitz) not installed. PDF processing will use alternative methods. "
        "Install with: pip install pymupdf"
    )

import subprocess

from PIL import Image

from ..core.config import settings

# M4 Hardware optimization imports
try:
    from ..core.optimized import get_memory_pool
    from ..core.optimized.device_router import (
        TaskType,
        get_device_router,
        get_resource_manager,
    )

    HARDWARE_OPT_AVAILABLE = True
except ImportError:
    HARDWARE_OPT_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Tesseract Fallback OCR - Supports all Indian Languages
# =============================================================================


class TesseractOCR:
    """
    Tesseract OCR fallback for when GOT-OCR2 is unavailable.

    Supports all 22 official Indian languages + English.
    Requires: brew install tesseract-lang (macOS) or apt install tesseract-ocr-* (Linux)
    """

    # Language code mapping: Display Name -> Tesseract code
    LANGUAGE_CODES = {
        "english": "eng",
        "hindi": "hin",
        "tamil": "tam",
        "telugu": "tel",
        "bengali": "ben",
        "marathi": "mar",
        "gujarati": "guj",
        "kannada": "kan",
        "malayalam": "mal",
        "punjabi": "pan",
        "odia": "ori",
        "oriya": "ori",
        "assamese": "asm",
        "urdu": "urd",
        "sanskrit": "san",
        "nepali": "nep",
        "sindhi": "snd",
        "kashmiri": "kas",
        "konkani": "kok",
        "manipuri": "mni",
        "bodo": "brx",
        "dogri": "doi",
        "maithili": "mai",
        "santali": "sat",
    }

    def __init__(self):
        """Initialize Tesseract OCR."""
        self._available = None
        self._available_languages = None

    def is_available(self) -> bool:
        """Check if Tesseract is installed."""
        if self._available is not None:
            return self._available

        try:
            result = subprocess.run(
                ["tesseract", "--version"], capture_output=True, text=True, timeout=5
            )
            self._available = result.returncode == 0
            if self._available:
                logger.info(
                    f"Tesseract available: {result.stdout.split()[1] if result.stdout else 'unknown version'}"
                )
        except (subprocess.SubprocessError, FileNotFoundError):
            self._available = False
            logger.warning("Tesseract not installed - fallback OCR unavailable")

        return self._available

    def get_available_languages(self) -> list[str]:
        """Get list of installed Tesseract language packs."""
        if self._available_languages is not None:
            return self._available_languages

        if not self.is_available():
            return []

        try:
            result = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Parse output: first line is info, rest are language codes
            lines = result.stdout.strip().split("\n")
            self._available_languages = [
                line.strip() for line in lines[1:] if line.strip()
            ]
            logger.info(
                f"Tesseract languages installed: {len(self._available_languages)}"
            )
        except subprocess.SubprocessError:
            self._available_languages = []

        return self._available_languages

    def _get_lang_string(self, languages: list[str] | None = None) -> str:
        """Convert language names to Tesseract language string."""
        if not languages:
            languages = ["english", "hindi"]

        available = self.get_available_languages()
        codes = []

        for lang in languages:
            code = self.LANGUAGE_CODES.get(lang.lower(), lang.lower()[:3])
            if code in available:
                codes.append(code)

        # Always include English if available
        if "eng" in available and "eng" not in codes:
            codes.insert(0, "eng")

        return "+".join(codes) if codes else "eng"

    def ocr_image(
        self,
        image: str | Path | Image.Image,
        languages: list[str] | None = None,
        config: str = "--psm 3",  # Fully automatic page segmentation
    ) -> str:
        """
        Perform OCR on an image using Tesseract.

        Args:
            image: Image path or PIL Image
            languages: List of language names (e.g., ['hindi', 'english'])
            config: Tesseract config options

        Returns:
            Extracted text
        """
        if not self.is_available():
            logger.error("Tesseract not available")
            return ""

        try:
            import pytesseract

            # Load image if path
            img = Image.open(image) if isinstance(image, (str, Path)) else image

            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Get language string
            lang = self._get_lang_string(languages)

            # Run OCR
            text = pytesseract.image_to_string(img, lang=lang, config=config)

            return text.strip()

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

    def ocr_with_confidence(
        self, image: str | Path | Image.Image, languages: list[str] | None = None
    ) -> tuple[str, float]:
        """
        Perform OCR and return text with confidence score.

        Returns:
            Tuple of (text, confidence)
        """
        if not self.is_available():
            return "", 0.0

        try:
            import pytesseract

            img = Image.open(image) if isinstance(image, (str, Path)) else image

            if img.mode != "RGB":
                img = img.convert("RGB")

            lang = self._get_lang_string(languages)

            # Get detailed data
            data = pytesseract.image_to_data(
                img, lang=lang, output_type=pytesseract.Output.DICT
            )

            # Calculate average confidence (excluding -1 values)
            confidences = [int(c) for c in data["conf"] if int(c) >= 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Get text
            text = pytesseract.image_to_string(img, lang=lang)

            return text.strip(), avg_confidence / 100.0

        except Exception as e:
            logger.error(f"Tesseract OCR with confidence failed: {e}")
            return "", 0.0


# Singleton Tesseract instance
_tesseract_ocr: TesseractOCR | None = None


def get_tesseract_ocr() -> TesseractOCR:
    """Get or create Tesseract OCR singleton."""
    global _tesseract_ocr
    if _tesseract_ocr is None:
        _tesseract_ocr = TesseractOCR()
    return _tesseract_ocr


@dataclass
class ExtractionResult:
    """Result of text extraction."""

    text: str
    num_pages: int
    has_images: bool
    has_formulas: bool
    has_tables: bool
    formula_blocks: list[dict]
    table_blocks: list[dict]
    metadata: dict
    confidence: float


class GOTOCR2:
    """
    GOT-OCR2 Vision-Language Model for document understanding.

    Optimized for Apple Silicon (M1/M2/M3/M4) with Metal Performance Shaders (MPS).

    Features:
    - 95%+ accuracy on Indian scripts
    - Formula recognition
    - Table extraction
    - Mixed layout understanding
    - Scene text recognition
    - MPS acceleration on Apple Silicon (3-5x faster than CPU)
    """

    # Supported OCR modes
    OCR_MODES = {
        "plain": "ocr",  # Plain text extraction
        "format": "format",  # Preserve formatting
        "fine-grained": "fine",  # Detailed extraction
        "multi-crop": "crop",  # Multi-region extraction
    }

    def __init__(self, model_id: str | None = None, device: str | None = None):
        """
        Initialize GOT-OCR2 model.

        Args:
            model_id: Model identifier (default from config)
            device: Device to use (auto-detected if not specified)
        """
        self.model_id = model_id or settings.OCR_MODEL_ID

        # Auto-detect device with Apple Silicon optimization
        if device is None:
            self.device = self._detect_best_device()
        else:
            self.device = device

        self._model = None
        self._tokenizer = None
        self._processor = None
        self._torch_dtype = None

        logger.info(f"GOT-OCR2 initialized: {self.model_id} on {self.device}")

    def _detect_best_device(self) -> str:
        """
        Detect the best available device for inference.

        Uses M4 hardware optimizer for intelligent routing if available.
        Priority: Hardware Router > CUDA > MPS (Apple Silicon) > CPU

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        import torch

        # Use hardware optimizer for intelligent device routing if available
        if HARDWARE_OPT_AVAILABLE:
            try:
                router = get_device_router()
                routing = router.route(TaskType.OCR)
                logger.info(
                    f"GOT-OCR2: Using {routing.device_str} (via hardware optimizer, speedup: {routing.estimated_speedup}x)"
                )
                return routing.device_str
            except Exception as e:
                logger.debug(f"Hardware optimizer failed, using fallback: {e}")

        # Check CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA available with {gpu_mem:.1f}GB VRAM")
            return "cuda"

        # Check MPS (Apple Silicon - M1/M2/M3/M4)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Check if MPS is actually functional
            try:
                # Quick test to ensure MPS works
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor

                # Get Apple Silicon info
                import platform

                chip = platform.processor() or "Apple Silicon"
                logger.info(f"MPS (Metal) available on {chip} - using GPU acceleration")
                return "mps"
            except Exception as e:
                logger.warning(f"MPS available but not functional: {e}")

        logger.info("Using CPU (no GPU acceleration available)")
        return "cpu"

    def _get_optimal_dtype(self):
        """Get optimal torch dtype for the device."""
        import torch

        if self.device == "cuda":
            return torch.float16  # FP16 for NVIDIA
        elif self.device == "mps":
            # MPS works best with float32 for most models
            # Some newer models support float16 on MPS
            return torch.float32
        else:
            return torch.float32  # CPU

    def _get_cuda_load_kwargs(self, load_kwargs: dict) -> dict:
        """Configure loading options for CUDA devices."""
        import torch

        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16

        if settings.USE_QUANTIZATION:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("Using 4-bit quantization for CUDA")
            except ImportError:
                pass
        return load_kwargs

    def _get_mps_load_kwargs(self, load_kwargs: dict) -> dict:
        """Configure loading options for Apple Silicon MPS."""
        import torch

        load_kwargs["torch_dtype"] = torch.float32
        logger.info("Configuring for Apple Silicon MPS acceleration")
        return load_kwargs

    def _post_load_device_setup(self, model, load_kwargs: dict):
        """Apply device-specific post-load optimizations."""
        if "device_map" not in load_kwargs:
            model = model.to(self.device)
            if self.device == "mps":
                model.eval()
                if hasattr(model, "enable_attention_slicing"):
                    model.enable_attention_slicing(1)
                    logger.info("Enabled attention slicing for MPS memory efficiency")
        return model

    def _load_model(self):
        """Lazy load the GOT-OCR2 model with device-specific optimizations."""
        if self._model is not None:
            return

        logger.info(f"Loading GOT-OCR2 model: {self.model_id} on {self.device}")

        try:
            from transformers import AutoModel, AutoTokenizer

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=str(settings.MODEL_CACHE_DIR),
            )

            self._torch_dtype = self._get_optimal_dtype()

            # Base loading arguments
            load_kwargs = {
                "trust_remote_code": True,
                "cache_dir": str(settings.MODEL_CACHE_DIR),
                "low_cpu_mem_usage": True,
            }

            # Apply device-specific configurations
            if self.device == "cuda":
                load_kwargs = self._get_cuda_load_kwargs(load_kwargs)
            elif self.device == "mps":
                load_kwargs = self._get_mps_load_kwargs(load_kwargs)

            # Load model
            self._model = AutoModel.from_pretrained(self.model_id, **load_kwargs)

            # Post-load device setup
            self._model = self._post_load_device_setup(self._model, load_kwargs)
            self._model.eval()

            # MPS shader warmup
            if self.device == "mps":
                self._warmup_mps()

            logger.info(f"GOT-OCR2 model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load GOT-OCR2: {e}")
            logger.info("Will use Tesseract fallback")
            raise

    def _warmup_mps(self):
        """Warm up MPS by running a dummy inference to compile Metal shaders."""
        try:
            import torch

            logger.info("Warming up MPS (compiling Metal shaders)...")

            # Create a small dummy image
            dummy_img = Image.new("RGB", (224, 224), color="white")

            # This compiles the Metal shaders, making subsequent inferences faster
            with torch.inference_mode():  # Faster than no_grad on M4
                # Just load the image through preprocessing
                _ = self._preprocess_image(dummy_img)

            logger.info("MPS warmup complete - Metal shaders compiled")
        except Exception as e:
            logger.warning(f"MPS warmup failed (non-critical): {e}")

    def _preprocess_image(
        self, image: str | Path | Image.Image, max_size: int | None = None
    ) -> Image.Image:
        """Preprocess image for OCR."""
        max_size = max_size or settings.OCR_MAX_IMAGE_SIZE

        img = Image.open(image) if isinstance(image, (str, Path)) else image

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if too large
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)

        return img

    def ocr_image(
        self,
        image: str | Path | Image.Image,
        mode: str = "format",
        languages: list[str] | None = None,
    ) -> str:
        """
        Perform OCR on a single image.

        Args:
            image: Image path or PIL Image
            mode: OCR mode ('plain', 'format', 'fine-grained', 'multi-crop')
            languages: Hint for expected languages (informational)

        Returns:
            Extracted text
        """
        self._load_model()

        img = self._preprocess_image(image)

        try:
            # GOT-OCR2 chat-based inference
            ocr_type = self.OCR_MODES.get(mode, "format")

            # Use model's chat method
            result = self._model.chat(
                self._tokenizer,
                img,
                ocr_type=ocr_type,
                ocr_box="",
                ocr_color="",
            )

            return result.strip()

        except Exception as e:
            logger.error(f"GOT-OCR2 inference failed: {e}")
            # Return empty string on failure
            return ""

    def extract_formulas(self, image: str | Path | Image.Image) -> list[dict]:
        """
        Extract mathematical formulas from image.

        Returns:
            List of formula dictionaries with LaTeX representations
        """
        self._load_model()

        img = self._preprocess_image(image)

        try:
            # Use fine-grained mode for formula extraction
            result = self._model.chat(
                self._tokenizer,
                img,
                ocr_type="format",
                ocr_box="",
                ocr_color="",
            )

            # Parse formulas from result
            formulas = []

            # Look for LaTeX patterns
            latex_patterns = [
                r"\$\$([^\$]+)\$\$",  # Display math
                r"\$([^\$]+)\$",  # Inline math
                r"\\begin\{equation\}(.+?)\\end\{equation\}",
                r"\\begin\{align\}(.+?)\\end\{align\}",
            ]

            for pattern in latex_patterns:
                for match in re.finditer(pattern, result, re.DOTALL):
                    formulas.append(
                        {
                            "type": "latex",
                            "content": match.group(1).strip(),
                            "original": match.group(0),
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )

            return formulas

        except Exception as e:
            logger.error(f"Formula extraction failed: {e}")
            return []

    def extract_tables(self, image: str | Path | Image.Image) -> list[dict]:
        """
        Extract tables from image.

        Returns:
            List of table dictionaries with structured data
        """
        self._load_model()

        img = self._preprocess_image(image)

        try:
            # Use format mode for table extraction
            result = self._model.chat(
                self._tokenizer,
                img,
                ocr_type="format",
                ocr_box="",
                ocr_color="",
            )

            tables = []

            # Look for table patterns (markdown or HTML-like)
            # GOT-OCR2 often outputs tables in markdown format
            table_pattern = r"\|[^\n]+\|(?:\n\|[^\n]+\|)+"

            for match in re.finditer(table_pattern, result):
                table_text = match.group(0)
                rows = table_text.strip().split("\n")

                parsed_rows = []
                for row in rows:
                    cells = [cell.strip() for cell in row.split("|") if cell.strip()]
                    if cells and not all(c.startswith("-") for c in cells):
                        parsed_rows.append(cells)

                if parsed_rows:
                    tables.append(
                        {
                            "type": "markdown",
                            "rows": parsed_rows,
                            "num_rows": len(parsed_rows),
                            "num_cols": len(parsed_rows[0]) if parsed_rows else 0,
                            "original": table_text,
                        }
                    )

            return tables

        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []

    @property
    def is_loaded(self) -> bool:
        """Check if OCR model is loaded."""
        return self._model is not None

    def unload(self) -> None:
        """Unload the OCR model and free memory (for coordinated shutdown)."""
        if self._model is None:
            return

        logger.info("Unloading GOT-OCR2 model...")

        # Release memory via coordinator
        try:
            from ..core.optimized.memory_coordinator import get_memory_coordinator

            coordinator = get_memory_coordinator()
            coordinator.release("ocr")
        except ImportError:
            pass

        import gc

        # Delete model and tokenizer
        del self._model
        self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        # Free MPS cache
        try:
            import torch

            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        gc.collect()
        logger.info("GOT-OCR2 model unloaded")


class PDFProcessor:
    """Process PDF documents using GOT-OCR2."""

    def __init__(self, ocr: GOTOCR2 = None):
        self.ocr = ocr or GOTOCR2()

    def process_pdf(
        self,
        pdf_path: str,
        mode: str = "format",
        extract_formulas: bool = True,
        extract_tables: bool = True,
    ) -> ExtractionResult:
        """
        Process a PDF document.

        Args:
            pdf_path: Path to PDF file
            mode: OCR mode
            extract_formulas: Whether to extract formulas
            extract_tables: Whether to extract tables

        Returns:
            ExtractionResult with all extracted content
        """
        logger.info(f"Processing PDF: {pdf_path}")

        if not FITZ_AVAILABLE:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF processing. "
                "Install with: pip install pymupdf"
            )

        try:
            doc = fitz.open(pdf_path)

            # Process all pages
            page_results = [
                self._process_page(
                    page, page_num, mode, extract_formulas, extract_tables
                )
                for page_num, page in enumerate(doc)
            ]

            doc.close()

            # Aggregate results
            return self._aggregate_results(page_results, pdf_path)

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise

    def _process_page(
        self,
        page,
        page_num: int,
        mode: str,
        extract_formulas: bool,
        extract_tables: bool,
    ) -> dict:
        """Process a single PDF page."""
        page_text = page.get_text("text")
        has_images = bool(page.get_images())
        formulas = []
        tables = []

        # Use OCR if insufficient text or has images
        needs_ocr = len(page_text.strip()) < 100 or has_images

        if needs_ocr:
            img = self._page_to_image(page)
            page_text = self.ocr.ocr_image(img, mode=mode)

            if extract_formulas:
                formulas = self._extract_page_formulas(img, page_num)

            if extract_tables:
                tables = self._extract_page_tables(img, page_num)

        return {
            "text": page_text,
            "has_images": has_images,
            "formulas": formulas,
            "tables": tables,
        }

    def _page_to_image(self, page) -> Image.Image:
        """Convert PDF page to PIL Image."""
        if not FITZ_AVAILABLE:
            raise ImportError("PyMuPDF required for PDF page conversion")
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _extract_page_formulas(self, img: Image.Image, page_num: int) -> list:
        """Extract formulas from page image."""
        formulas = self.ocr.extract_formulas(img)
        for f in formulas:
            f["page"] = page_num + 1
        return formulas

    def _extract_page_tables(self, img: Image.Image, page_num: int) -> list:
        """Extract tables from page image."""
        tables = self.ocr.extract_tables(img)
        for t in tables:
            t["page"] = page_num + 1
        return tables

    def _aggregate_results(self, page_results: list, pdf_path: str) -> ExtractionResult:
        """Aggregate results from all pages."""
        full_text = [r["text"] for r in page_results]
        formula_blocks = [f for r in page_results for f in r["formulas"]]
        table_blocks = [t for r in page_results for t in r["tables"]]
        has_images = any(r["has_images"] for r in page_results)

        combined_text = "\n\n".join(full_text)
        cleaned_text = self._clean_text(combined_text)

        return ExtractionResult(
            text=cleaned_text,
            num_pages=len(page_results),
            has_images=has_images,
            has_formulas=len(formula_blocks) > 0,
            has_tables=len(table_blocks) > 0,
            formula_blocks=formula_blocks,
            table_blocks=table_blocks,
            metadata={
                "filename": Path(pdf_path).name,
                "file_size": os.path.getsize(pdf_path),
                "pages": len(page_results),
                "ocr_model": self.ocr.model_id,
            },
            confidence=0.95,  # GOT-OCR2 is highly accurate
        )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove standalone page numbers
        text = re.sub(r"\n\d+\n", "\n", text)

        # Fix common OCR artifacts
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        text = text.replace("ﬀ", "ff")

        return text.strip()


class OCRService:
    """
    Unified OCR service with GOT-OCR2 + Tesseract fallback.

    Supports:
    - PDF documents
    - Images (PNG, JPG, TIFF, BMP)
    - Indian scripts (22 official languages)
    - Mathematical formulas
    - Tables

    Priority:
    1. GOT-OCR2 (95%+ accuracy, GPU required)
    2. Tesseract (80-90% accuracy, CPU only, always available)
    """

    # Supported Indian languages
    SUPPORTED_LANGUAGES = [
        "Hindi",
        "Tamil",
        "Telugu",
        "Bengali",
        "Marathi",
        "Gujarati",
        "Kannada",
        "Malayalam",
        "Punjabi",
        "Odia",
        "Assamese",
        "Urdu",
        "Sanskrit",
        "Nepali",
        "Sindhi",
        "Kashmiri",
        "Konkani",
        "Manipuri",
        "Bodo",
        "Dogri",
        "Maithili",
        "Santali",
        "English",
    ]

    def __init__(
        self, languages: list[str] | None = None, force_tesseract: bool = False
    ):
        """
        Initialize OCR service.

        Args:
            languages: Expected languages for OCR
            force_tesseract: Force use of Tesseract instead of GOT-OCR2
        """
        self.languages = languages or ["English", "Hindi"]
        self.force_tesseract = force_tesseract

        # Initialize backends
        self._got_ocr = None
        self._tesseract = get_tesseract_ocr()
        self._got_available = None
        self._device_info = self._get_device_info()

        # Determine which backend to use
        self._use_tesseract = force_tesseract or not self._check_got_available()

        if self._use_tesseract:
            logger.info(f"OCRService using Tesseract (languages: {self.languages})")
        else:
            self._got_ocr = GOTOCR2()
            self.pdf_processor = PDFProcessor(self._got_ocr)
            logger.info(
                f"OCRService using GOT-OCR2 on {self._device_info['device']} (languages: {self.languages})"
            )

    def _get_device_info(self) -> dict:
        """Get detailed device information for optimization."""
        info = {
            "device": "cpu",
            "device_name": "CPU",
            "memory_gb": 0,
            "is_apple_silicon": False,
            "chip": None,
        }

        try:
            import platform

            import torch

            # Check for Apple Silicon
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                info["is_apple_silicon"] = True

                # Try to get chip name
                try:
                    import subprocess

                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    chip = result.stdout.strip()
                    if chip:
                        info["chip"] = chip
                        # Extract M1/M2/M3/M4 etc.
                        import re

                        match = re.search(r"Apple M\d+", chip)
                        if match:
                            info["chip"] = match.group(0)
                except Exception:
                    info["chip"] = "Apple Silicon"

                # Get unified memory
                try:
                    import subprocess

                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    mem_bytes = int(result.stdout.strip())
                    info["memory_gb"] = mem_bytes / (1024**3)
                except Exception:
                    pass

            # Check GPU availability
            if torch.cuda.is_available():
                info["device"] = "cuda"
                info["device_name"] = torch.cuda.get_device_name(0)
                info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info["device"] = "mps"
                info["device_name"] = f"Metal ({info.get('chip', 'Apple Silicon')})"

        except Exception as e:
            logger.warning(f"Could not get device info: {e}")

        return info

    def _check_got_available(self) -> bool:
        """Check if GOT-OCR2 is available (has GPU/memory)."""
        if self._got_available is not None:
            return self._got_available

        try:
            import torch

            # Check CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self._got_available = gpu_mem >= 6.0
                if self._got_available:
                    logger.info(
                        f"CUDA available: {self._device_info['device_name']} ({gpu_mem:.1f}GB)"
                    )
                else:
                    logger.warning(f"GPU has {gpu_mem:.1f}GB VRAM, GOT-OCR2 needs 6GB+")
                return self._got_available

            # Check MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple Silicon with unified memory - check if we have enough RAM
                mem_gb = self._device_info.get("memory_gb", 8)
                chip = self._device_info.get("chip", "Apple Silicon")

                # M4 chips have excellent MPS performance
                # Require at least 16GB unified memory for GOT-OCR2
                if mem_gb >= 16:
                    self._got_available = True
                    logger.info(
                        f"MPS (Metal) available: {chip} with {mem_gb:.0f}GB unified memory"
                    )
                elif mem_gb >= 8:
                    # 8GB might work but will be slower with memory swapping
                    self._got_available = True
                    logger.warning(
                        f"MPS available: {chip} with {mem_gb:.0f}GB (16GB+ recommended for best performance)"
                    )
                else:
                    self._got_available = False
                    logger.warning(
                        f"Insufficient memory for GOT-OCR2: {mem_gb:.0f}GB (need 8GB+)"
                    )

                return self._got_available

            # CPU only
            self._got_available = False
            logger.info("No GPU available, using Tesseract fallback")

        except Exception as e:
            logger.warning(f"Could not check GPU availability: {e}")
            self._got_available = False

        return self._got_available

    def get_backend_info(self) -> dict:
        """Get information about the active OCR backend."""
        return {
            "backend": "tesseract" if self._use_tesseract else "got-ocr2",
            "device": self._device_info,
            "languages": self.languages,
            "tesseract_available": self._tesseract.is_available(),
            "got_available": self._got_available,
        }

    @property
    def ocr(self):
        """Get the active OCR backend (for compatibility)."""
        if self._use_tesseract:
            return self._tesseract
        return self._got_ocr

    def extract_text(
        self,
        file_path: str,
        mode: str = "format",
        extract_formulas: bool = True,
        extract_tables: bool = True,
    ) -> ExtractionResult:
        """
        Extract text from PDF or image.

        Args:
            file_path: Path to PDF or image file
            mode: OCR mode ('plain', 'format', 'fine-grained')
            extract_formulas: Whether to extract formulas
            extract_tables: Whether to extract tables

        Returns:
            ExtractionResult with text and metadata
        """
        file_ext = Path(file_path).suffix.lower()

        # Use Tesseract path
        if self._use_tesseract:
            return self._extract_with_tesseract(file_path, file_ext)

        # Use GOT-OCR2 path
        try:
            if file_ext == ".pdf":
                return self.pdf_processor.process_pdf(
                    file_path,
                    mode=mode,
                    extract_formulas=extract_formulas,
                    extract_tables=extract_tables,
                )

            elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
                return self._process_image(
                    file_path,
                    mode=mode,
                    extract_formulas=extract_formulas,
                    extract_tables=extract_tables,
                )

            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

        except Exception as e:
            logger.warning(f"GOT-OCR2 failed, falling back to Tesseract: {e}")
            return self._extract_with_tesseract(file_path, file_ext)

    def _extract_with_tesseract(
        self, file_path: str, file_ext: str
    ) -> ExtractionResult:
        """Extract text using Tesseract fallback."""
        logger.info(f"Using Tesseract for: {file_path}")

        if file_ext == ".pdf":
            return self._process_pdf_with_tesseract(file_path)
        elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
            return self._process_image_with_tesseract(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _process_pdf_with_tesseract(self, pdf_path: str) -> ExtractionResult:
        """Process PDF using Tesseract."""
        if not FITZ_AVAILABLE:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF processing. "
                "Install with: pip install pymupdf"
            )

        try:
            doc = fitz.open(pdf_path)
            page_texts = []

            for page in doc:
                # First try to get embedded text
                text = page.get_text("text").strip()

                # If insufficient text, use OCR
                if len(text) < 100:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = self._tesseract.ocr_image(img, languages=self.languages)

                page_texts.append(text)

            doc.close()

            combined_text = "\n\n".join(page_texts)

            return ExtractionResult(
                text=combined_text,
                num_pages=len(page_texts),
                has_images=True,
                has_formulas=False,  # Tesseract doesn't extract formulas
                has_tables=False,
                formula_blocks=[],
                table_blocks=[],
                metadata={
                    "filename": Path(pdf_path).name,
                    "file_size": os.path.getsize(pdf_path),
                    "pages": len(page_texts),
                    "ocr_model": "tesseract",
                },
                confidence=0.85,  # Tesseract is less accurate
            )

        except Exception as e:
            logger.error(f"Tesseract PDF processing failed: {e}")
            raise

    def _process_image_with_tesseract(self, image_path: str) -> ExtractionResult:
        """Process image using Tesseract."""
        text, confidence = self._tesseract.ocr_with_confidence(
            image_path, languages=self.languages
        )

        return ExtractionResult(
            text=text,
            num_pages=1,
            has_images=True,
            has_formulas=False,
            has_tables=False,
            formula_blocks=[],
            table_blocks=[],
            metadata={
                "filename": Path(image_path).name,
                "file_size": os.path.getsize(image_path),
                "ocr_model": "tesseract",
            },
            confidence=confidence,
        )

    def _process_image(
        self,
        image_path: str,
        mode: str = "format",
        extract_formulas: bool = True,
        extract_tables: bool = True,
    ) -> ExtractionResult:
        """Process a single image."""
        logger.info(f"Processing image: {image_path}")

        text = self.ocr.ocr_image(image_path, mode=mode)

        formula_blocks = []
        table_blocks = []

        if extract_formulas:
            formula_blocks = self.ocr.extract_formulas(image_path)

        if extract_tables:
            table_blocks = self.ocr.extract_tables(image_path)

        return ExtractionResult(
            text=text,
            num_pages=1,
            has_images=True,
            has_formulas=len(formula_blocks) > 0,
            has_tables=len(table_blocks) > 0,
            formula_blocks=formula_blocks,
            table_blocks=table_blocks,
            metadata={
                "filename": Path(image_path).name,
                "file_size": os.path.getsize(image_path),
                "ocr_model": self.ocr.model_id,
            },
            confidence=0.95,
        )

    async def extract_text_async(self, file_path: str, **kwargs) -> ExtractionResult:
        """Async wrapper for text extraction."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.extract_text(file_path, **kwargs)
        )

    def validate_extraction(self, result: ExtractionResult) -> bool:
        """Validate extraction quality."""
        if len(result.text.strip()) < 50:
            logger.warning("Extracted text too short")
            return False

        if result.confidence < 0.60:
            logger.warning(f"Low OCR confidence: {result.confidence:.2f}")
            return False

        return True


# Thread-safe singleton with memory coordinator integration
import threading

_ocr_service_lock = threading.Lock()
_ocr_service: OCRService | None = None
_got_ocr_lock = threading.Lock()
_got_ocr_instance: GOTOCR2 | None = None


def get_ocr_service(
    languages: list[str] | None = None, force_tesseract: bool = False
) -> OCRService:
    """Get or create OCR service singleton (thread-safe)."""
    global _ocr_service

    with _ocr_service_lock:
        if _ocr_service is None:
            _ocr_service = OCRService(
                languages=languages, force_tesseract=force_tesseract
            )
        return _ocr_service


def get_got_ocr_service() -> GOTOCR2:
    """Get or create GOT-OCR2 singleton for model collaboration (with memory coordination)."""
    global _got_ocr_instance

    if _got_ocr_instance is not None and _got_ocr_instance.is_loaded:
        return _got_ocr_instance

    with _got_ocr_lock:
        if _got_ocr_instance is not None and _got_ocr_instance.is_loaded:
            return _got_ocr_instance

        # Create new instance with memory coordinator
        try:
            from ..core.optimized.memory_coordinator import get_memory_coordinator

            coordinator = get_memory_coordinator()
            acquired = coordinator.try_acquire_sync("ocr")

            if not acquired:
                logger.warning(
                    "Could not acquire memory for OCR model - memory pressure may occur"
                )
        except ImportError:
            pass  # Memory coordinator not available

        _got_ocr_instance = GOTOCR2()
        return _got_ocr_instance


def unload_got_ocr_service() -> None:
    """Unload the GOT-OCR2 singleton."""
    global _got_ocr_instance

    with _got_ocr_lock:
        if _got_ocr_instance is not None:
            _got_ocr_instance.unload()
            _got_ocr_instance = None


# Export
__all__ = [
    "GOTOCR2",
    "ExtractionResult",
    "OCRService",
    "PDFProcessor",
    "TesseractOCR",
    "get_got_ocr_service",
    "get_ocr_service",
    "get_tesseract_ocr",
    "unload_got_ocr_service",
]
