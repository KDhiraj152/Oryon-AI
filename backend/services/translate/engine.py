"""Translation Engine component using IndicTrans2 for multi-language translation.

v1.3.4: Added semantic accuracy refinement (target 8.2+)
v1.3.5: Added translation caching + dedicated ML executor (M4 optimization)
v1.3.6: Switched to fast xxhash for cache keys (10x faster)
"""

import asyncio
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.utils.hashing import fast_hash

logger = logging.getLogger(__name__)

# M4 Optimization: Use unified ThreadPoolManager for ML translation operations
# Prevents blocking the async event loop with model inference
_translation_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_translation_executor() -> ThreadPoolExecutor:
    """Get translation executor from unified ThreadPoolManager (thread-safe, lazy init)."""
    global _translation_executor
    if _translation_executor is None:
        with _executor_lock:
            if _translation_executor is None:
                from backend.core.optimized.thread_pool_manager import get_ml_executor
                _translation_executor = get_ml_executor()
    return _translation_executor


def shutdown_translation_executor(wait: bool = True) -> None:
    """Shutdown the translation executor gracefully.

    Args:
        wait: If True, wait for pending tasks to complete.
              If False, cancel pending tasks immediately.

    Note: No-op since executor is now shared via ThreadPoolManager.
    """
    global _translation_executor
    with _executor_lock:
        # Don't shutdown — it's owned by ThreadPoolManager
        _translation_executor = None
        logger.info("Translation executor reference cleared (pool managed globally)")


# Translation cache: key = hash(text + source + target), value = TranslatedText
# Uses OrderedDict for proper LRU eviction (FIXED: was naive deletion)
from collections import OrderedDict

_translation_cache: OrderedDict[str, Any] = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX_SIZE = 10000  # Max cached translations


def _get_cache_key(text: str, source_lang: str, target_lang: str, subject: str) -> str:
    """Generate cache key for translation. Uses fast xxhash if available."""
    content = f"{text}|{source_lang}|{target_lang}|{subject}"
    return fast_hash(content, length=32)


def _cache_get(key: str) -> Any | None:
    """Get cached translation and move to end (LRU)."""
    with _cache_lock:
        if key in _translation_cache:
            # Move to end (most recently used)
            _translation_cache.move_to_end(key)
            return _translation_cache[key]
        return None


def _cache_set(key: str, value: Any) -> None:
    """Cache translation result with proper LRU eviction."""
    with _cache_lock:
        # If key exists, update and move to end
        if key in _translation_cache:
            _translation_cache[key] = value
            _translation_cache.move_to_end(key)
            return

        # Evict oldest entries if at capacity
        while len(_translation_cache) >= _CACHE_MAX_SIZE:
            _translation_cache.popitem(last=False)  # Remove oldest (LRU)

        _translation_cache[key] = value


# Refinement pipeline for semantic accuracy
try:
    from ..evaluation.refinement_pipeline import (
        RefinementConfig,
        RefinementTask,
        SemanticRefinementPipeline,
    )

    REFINEMENT_AVAILABLE = True
except ImportError:
    REFINEMENT_AVAILABLE = False
    logger.warning("Refinement pipeline not available for translation")

# Singleton adapter instance (thread-safe)
_indictrans2_adapter = None
_indictrans2_lock = None  # Lazy init


class IndicTrans2Adapter:
    """
    Adapter to make IndicTrans2 model compatible with TranslationEngine interface.

    The engine expects model_client.process(text, target_language)
    but IndicTrans2 has translate(text, source_lang, target_lang).
    """

    def __init__(self, model):
        self._model = model

    def process(
        self, text: str, target_language: str, source_language: str = "English"
    ) -> str:
        """
        Process translation request using IndicTrans2.

        Args:
            text: Text to translate
            target_language: Target language name (e.g., 'Hindi', 'Tamil')
            source_language: Source language name (default: 'English')

        Returns:
            Translated text string
        """
        result = self._model.translate(
            text=text, source_lang=source_language, target_lang=target_language
        )
        return result.translated_text


def get_indictrans2_model():
    """Get or create singleton IndicTrans2 adapter using centralized singleton (thread-safe)."""
    global _indictrans2_adapter, _indictrans2_lock
    import threading

    if _indictrans2_lock is None:
        _indictrans2_lock = threading.Lock()
    if _indictrans2_adapter is None:
        with _indictrans2_lock:
            if _indictrans2_adapter is None:
                try:
                    from .model import get_translator

                    model = get_translator()  # Use centralized singleton
                    _indictrans2_adapter = IndicTrans2Adapter(model)
                    logger.info("IndicTrans2 adapter initialized (using singleton)")
                except Exception as e:
                    logger.warning(f"Could not initialize IndicTrans2 adapter: {e}")
                    _indictrans2_adapter = None
    return _indictrans2_adapter


@dataclass
class TranslatedText:
    """Result of text translation."""

    text: str
    source_language: str
    target_language: str
    subject: str
    script_valid: bool
    semantic_score: float
    metadata: dict[str, Any]

    # Refinement metrics (v1.3.4)
    refinement_iterations: int = 0
    dimension_scores: dict[str, float] | None = None
    target_reached: bool = False


class TranslationEngine:
    """
    Translation engine component that converts content
    into Indian languages using IndicTrans2 model.

    Supports: Hindi, Tamil, Telugu, Bengali, Marathi (MVP)
    Expandable to 10+ Indian languages
    """

    # Supported languages - All 10 major Indian languages
    SUPPORTED_LANGUAGES = {
        "Hindi": {
            "code": "hin_Deva",
            "script": "Devanagari",
            "unicode_range": (0x0900, 0x097F),
        },
        "Tamil": {
            "code": "tam_Taml",
            "script": "Tamil",
            "unicode_range": (0x0B80, 0x0BFF),
        },
        "Telugu": {
            "code": "tel_Telu",
            "script": "Telugu",
            "unicode_range": (0x0C00, 0x0C7F),
        },
        "Bengali": {
            "code": "ben_Beng",
            "script": "Bengali",
            "unicode_range": (0x0980, 0x09FF),
        },
        "Marathi": {
            "code": "mar_Deva",
            "script": "Devanagari",
            "unicode_range": (0x0900, 0x097F),
        },
        "Gujarati": {
            "code": "guj_Gujr",
            "script": "Gujarati",
            "unicode_range": (0x0A80, 0x0AFF),
        },
        "Kannada": {
            "code": "kan_Knda",
            "script": "Kannada",
            "unicode_range": (0x0C80, 0x0CFF),
        },
        "Malayalam": {
            "code": "mal_Mlym",
            "script": "Malayalam",
            "unicode_range": (0x0D00, 0x0D7F),
        },
        "Punjabi": {
            "code": "pan_Guru",
            "script": "Gurmukhi",
            "unicode_range": (0x0A00, 0x0A7F),
        },
        "Odia": {
            "code": "ory_Orya",
            "script": "Odia",
            "unicode_range": (0x0B00, 0x0B7F),
        },
    }

    # Subject-specific technical terminology mappings for all 10 Indian languages
    TECHNICAL_TERMS = {
        "Mathematics": {
            "equation": {
                "Hindi": "समीकरण",
                "Tamil": "சமன்பாடு",
                "Telugu": "సమీకరణం",
                "Bengali": "সমীকরণ",
                "Marathi": "समीकरण",
                "Gujarati": "સમીકરણ",
                "Kannada": "ಸಮೀಕರಣ",
                "Malayalam": "സമവാക്യം",
                "Punjabi": "ਸਮੀਕਰਨ",
                "Odia": "ସମୀକରଣ",
            },
            "variable": {
                "Hindi": "चर",
                "Tamil": "மாறி",
                "Telugu": "చరరాశి",
                "Bengali": "চলক",
                "Marathi": "चल",
                "Gujarati": "ચલ",
                "Kannada": "ಚರ",
                "Malayalam": "ചരം",
                "Punjabi": "ਚਲ",
                "Odia": "ଚଳ",
            },
            "theorem": {
                "Hindi": "प्रमेय",
                "Tamil": "தேற்றம்",
                "Telugu": "సిద్ధాంతం",
                "Bengali": "উপপাদ্য",
                "Marathi": "प्रमेय",
                "Gujarati": "પ્રમેય",
                "Kannada": "ಪ್ರಮೇಯ",
                "Malayalam": "പ്രമേയം",
                "Punjabi": "ਪ੍ਰਮੇਯ",
                "Odia": "ପ୍ରମେୟ",
            },
            "function": {
                "Hindi": "फलन",
                "Tamil": "சார்பு",
                "Telugu": "ఫంక్షన్",
                "Bengali": "অপেক্ষক",
                "Marathi": "फलन",
                "Gujarati": "ફલન",
                "Kannada": "ಫಲನ",
                "Malayalam": "ഫലനം",
                "Punjabi": "ਫਲਨ",
                "Odia": "ଫଳନ",
            },
            "graph": {
                "Hindi": "आलेख",
                "Tamil": "வரைபடம்",
                "Telugu": "గ్రాఫ్",
                "Bengali": "লেখচিত্র",
                "Marathi": "आलेख",
                "Gujarati": "આલેખ",
                "Kannada": "ಗ್ರಾಫ್",
                "Malayalam": "ഗ്രാഫ്",
                "Punjabi": "ਗ੍ਰਾਫ਼",
                "Odia": "ଗ୍ରାଫ୍",
            },
        },
        "Science": {
            "photosynthesis": {
                "Hindi": "प्रकाश संश्लेषण",
                "Tamil": "ஒளிச்சேர்க்கை",
                "Telugu": "కిరణజన్య సంయోగక్రియ",
                "Bengali": "সালোকসংশ্লেষ",
                "Marathi": "प्रकाश संश्लेषण",
                "Gujarati": "પ્રકાશસંશ્લેષણ",
                "Kannada": "ದ್ಯುತಿಸಂಶ್ಲೇಷಣೆ",
                "Malayalam": "പ്രകാശസംശ്ലേഷണം",
                "Punjabi": "ਪ੍ਰਕਾਸ਼ ਸੰਸ਼ਲੇਸ਼ਣ",
                "Odia": "ସାଲୋକସଂଶ୍ଲେଷଣ",
            },
            "molecule": {
                "Hindi": "अणु",
                "Tamil": "மூலக்கூறு",
                "Telugu": "అణువు",
                "Bengali": "অণু",
                "Marathi": "रेणू",
                "Gujarati": "અણુ",
                "Kannada": "ಅಣು",
                "Malayalam": "തന്മാത്ര",
                "Punjabi": "ਅਣੂ",
                "Odia": "ଅଣୁ",
            },
            "atom": {
                "Hindi": "परमाणु",
                "Tamil": "அணு",
                "Telugu": "పరమాణువు",
                "Bengali": "পরমাণু",
                "Marathi": "अणु",
                "Gujarati": "પરમાણુ",
                "Kannada": "ಪರಮಾಣು",
                "Malayalam": "അണു",
                "Punjabi": "ਪਰਮਾਣੂ",
                "Odia": "ପରମାଣୁ",
            },
            "cell": {
                "Hindi": "कोशिका",
                "Tamil": "செல்",
                "Telugu": "కణం",
                "Bengali": "কোষ",
                "Marathi": "पेशी",
                "Gujarati": "કોષ",
                "Kannada": "ಕೋಶ",
                "Malayalam": "കോശം",
                "Punjabi": "ਸੈੱਲ",
                "Odia": "କୋଷ",
            },
            "energy": {
                "Hindi": "ऊर्जा",
                "Tamil": "ஆற்றல்",
                "Telugu": "శక్తి",
                "Bengali": "শক্তি",
                "Marathi": "ऊर्जा",
                "Gujarati": "ઊર્જા",
                "Kannada": "ಶಕ್ತಿ",
                "Malayalam": "ഊർജം",
                "Punjabi": "ਊਰਜਾ",
                "Odia": "ଶକ୍ତି",
            },
        },
        "Social Studies": {
            "democracy": {
                "Hindi": "लोकतंत्र",
                "Tamil": "ஜனநாயகம்",
                "Telugu": "ప్రజాస్వామ్యం",
                "Bengali": "গণতন্ত্র",
                "Marathi": "लोकशाही",
                "Gujarati": "લોકશાહી",
                "Kannada": "ಪ್ರಜಾಪ್ರಭುತ್ವ",
                "Malayalam": "ജനാധിപത്യം",
                "Punjabi": "ਲੋਕਤੰਤਰ",
                "Odia": "ଗଣତନ୍ତ୍ର",
            },
            "constitution": {
                "Hindi": "संविधान",
                "Tamil": "அரசியலமைப்பு",
                "Telugu": "రాజ్యాంగం",
                "Bengali": "সংবিধান",
                "Marathi": "संविधान",
                "Gujarati": "બંધારણ",
                "Kannada": "ಸಂವಿಧಾನ",
                "Malayalam": "ഭരണഘടന",
                "Punjabi": "ਸੰਵਿਧਾਨ",
                "Odia": "ସମ୍ବିଧାନ",
            },
            "government": {
                "Hindi": "सरकार",
                "Tamil": "அரசாங்கம்",
                "Telugu": "ప్రభుత్వం",
                "Bengali": "সরকার",
                "Marathi": "सरकार",
                "Gujarati": "સરકાર",
                "Kannada": "ಸರ್ಕಾರ",
                "Malayalam": "സർക്കാർ",
                "Punjabi": "ਸਰਕਾਰ",
                "Odia": "ସରକାର",
            },
            "economy": {
                "Hindi": "अर्थव्यवस्था",
                "Tamil": "பொருளாதாரம்",
                "Telugu": "ఆర్థిక వ్యవస్థ",
                "Bengali": "অর্থনীতি",
                "Marathi": "अर्थव्यवस्था",
                "Gujarati": "અર્થતંત્ર",
                "Kannada": "ಆರ್ಥಿಕತೆ",
                "Malayalam": "സമ്പദ്‌വ്യവസ്ഥ",
                "Punjabi": "ਅਰਥਵਿਵਸਥਾ",
                "Odia": "ଅର୍ଥନୀତି",
            },
        },
    }

    def __init__(
        self,
        model_client=None,
        enable_refinement: bool = True,
        target_semantic_score: float = 9.0,  # M4-optimized target
    ):
        """
        Initialize the Translation Engine.

        Args:
            model_client: Optional IndicTrans2 model client for inference
                         If None, will attempt to use local IndicTrans2 model
            enable_refinement: Enable iterative refinement for semantic accuracy
            target_semantic_score: Minimum semantic score target (default: 9.0 for M4)
        """
        # Use provided client or attempt to load local model
        if model_client is None:
            self.model_client = get_indictrans2_model()
        else:
            self.model_client = model_client

        # Refinement configuration (v1.3.4)
        self.enable_refinement = enable_refinement and REFINEMENT_AVAILABLE
        self.target_semantic_score = target_semantic_score
        self._refinement_pipeline = None

        logger.info(
            "TranslationEngine initialized with support for: %s",
            ", ".join(self.SUPPORTED_LANGUAGES.keys()),
        )
        if self.model_client:
            logger.info("IndicTrans2 model client available")
        else:
            logger.warning("No model client available, will use fallback translation")

        if self.enable_refinement:
            logger.info(
                f"Translation refinement enabled (target: {target_semantic_score})"
            )

    def _get_refinement_pipeline(self) -> "SemanticRefinementPipeline":
        """Lazy-load refinement pipeline."""
        if self._refinement_pipeline is None and REFINEMENT_AVAILABLE:
            config = RefinementConfig(
                target_score=self.target_semantic_score, max_iterations=3
            )
            self._refinement_pipeline = SemanticRefinementPipeline(config=config)
        return self._refinement_pipeline

    def translate(
        self,
        text: str,
        target_language: str,
        subject: str,
        source_language: str = "English",
        use_refinement: bool | None = None,
        use_cache: bool = True,
    ) -> TranslatedText:
        """
        Translate text to target Indian language.

        M4 Optimization: Caches translations for 10-100x speedup on repeated queries.

        Args:
            text: Source text to translate
            target_language: Target Indian language (Hindi, Tamil, Telugu, Bengali, Marathi)
            subject: Subject area for technical terminology handling
            source_language: Source language (default: English)
            use_refinement: Override refinement setting (None = use instance default)
            use_cache: Whether to use translation cache (default: True)

        Returns:
            TranslatedText object with translated content and validation results

        Raises:
            ValueError: If target language is not supported or text is empty
        """
        # Validate inputs
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")

        if target_language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{target_language}' not supported. "
                f"Supported languages: {', '.join(self.SUPPORTED_LANGUAGES.keys())}"
            )

        # M4 Optimization: Check cache first
        if use_cache:
            cache_key = _get_cache_key(text, source_language, target_language, subject)
            cached = _cache_get(cache_key)
            if cached is not None:
                logger.debug(f"Translation cache hit for {target_language}")
                return cached

        logger.info(f"Translating text to {target_language} for subject {subject}")

        # Determine if refinement should be used
        should_refine = (
            use_refinement if use_refinement is not None else self.enable_refinement
        )

        if should_refine and REFINEMENT_AVAILABLE:
            # For async contexts, run synchronously to avoid event loop issues
            # The refinement pipeline will handle its own async internally if needed
            try:
                # Try to get existing loop - if running, use sync approach
                try:
                    asyncio.get_running_loop()
                    # We're in an async context, use sync version
                    result = self._translate_single_pass(
                        text, target_language, subject, source_language
                    )
                except RuntimeError:
                    # No running loop, can use asyncio.run
                    result = asyncio.run(
                        self._translate_with_refinement(
                            text, target_language, subject, source_language
                        )
                    )
            except Exception as e:
                logger.warning(f"Refinement failed, using single pass: {e}")
                result = self._translate_single_pass(
                    text, target_language, subject, source_language
                )

            # Cache the result
            if use_cache:
                _cache_set(cache_key, result)
            return result

        result = self._translate_single_pass(
            text, target_language, subject, source_language
        )

        # Cache the result
        if use_cache:
            cache_key = _get_cache_key(text, source_language, target_language, subject)
            _cache_set(cache_key, result)

        return result

    def _translate_single_pass(
        self, text: str, target_language: str, subject: str, source_language: str
    ) -> TranslatedText:
        """Single-pass translation without refinement (original behavior)."""
        # Preserve technical terms before translation
        text_with_markers, term_map = self._mark_technical_terms(text, subject)

        # Perform translation
        if self.model_client:
            try:
                translated_text = self.model_client.process(
                    text_with_markers, target_language
                )
            except Exception as e:
                logger.warning(f"Model inference failed, using fallback: {e}")
                translated_text = self._fallback_translation(text, target_language)
        else:
            translated_text = self._fallback_translation(text, target_language)

        # Replace technical term markers with correct translations
        translated_text = self._restore_technical_terms(
            translated_text, term_map, target_language, subject
        )

        # Validate script rendering
        script_valid = self.validate_script_rendering(translated_text, target_language)

        # Calculate semantic equivalence (placeholder score if no validation model)
        semantic_score = self._calculate_semantic_equivalence(text, translated_text)

        logger.info(
            f"Translation complete - script_valid: {script_valid}, "
            f"semantic_score: {semantic_score:.2f}"
        )

        return TranslatedText(
            text=translated_text,
            source_language=source_language,
            target_language=target_language,
            subject=subject,
            script_valid=script_valid,
            semantic_score=semantic_score,
            metadata={
                "language_code": self.SUPPORTED_LANGUAGES[target_language]["code"],
                "script": self.SUPPORTED_LANGUAGES[target_language]["script"],
                "technical_terms_preserved": len(term_map),
                "refinement_enabled": False,
            },
        )

    async def _translate_with_refinement(
        self, text: str, target_language: str, subject: str, source_language: str
    ) -> TranslatedText:
        """
        Translate with iterative refinement to achieve target semantic accuracy.
        """
        # First, generate initial translation
        text_with_markers, term_map = self._mark_technical_terms(text, subject)

        if self.model_client:
            try:
                initial_translation = self.model_client.process(
                    text_with_markers, target_language
                )
            except Exception as e:
                logger.warning(f"Initial translation failed: {e}")
                initial_translation = self._fallback_translation(text, target_language)
        else:
            initial_translation = self._fallback_translation(text, target_language)

        initial_translation = self._restore_technical_terms(
            initial_translation, term_map, target_language, subject
        )

        # Run refinement pipeline
        try:
            pipeline = self._get_refinement_pipeline()
            result = await pipeline.refine(
                original_text=text,
                initial_output=initial_translation,
                task=RefinementTask.TRANSLATION,
                complexity_level=8,  # Default complexity level
                subject=subject,
            )

            final_text = result.final_text
            script_valid = self.validate_script_rendering(final_text, target_language)

            return TranslatedText(
                text=final_text,
                source_language=source_language,
                target_language=target_language,
                subject=subject,
                script_valid=script_valid,
                semantic_score=result.final_score,
                metadata={
                    "language_code": self.SUPPORTED_LANGUAGES[target_language]["code"],
                    "script": self.SUPPORTED_LANGUAGES[target_language]["script"],
                    "technical_terms_preserved": len(term_map),
                    "refinement_enabled": True,
                    "target_reached": result.achieved_target,
                    "initial_score": result.iteration_history[0].score
                    if result.iteration_history
                    else 0,
                    "iterations_history": [
                        {"iteration": it.iteration, "score": it.score}
                        for it in result.iteration_history
                    ],
                },
                refinement_iterations=result.iterations_used,
                dimension_scores=dict(result.iteration_history[-1].dimension_scores)
                if result.iteration_history
                else None,
                target_reached=result.achieved_target,
            )

        except Exception as e:
            logger.error(f"Refinement pipeline failed: {e}, returning initial output")
            script_valid = self.validate_script_rendering(
                initial_translation, target_language
            )
            semantic_score = self._calculate_semantic_equivalence(
                text, initial_translation
            )

            return TranslatedText(
                text=initial_translation,
                source_language=source_language,
                target_language=target_language,
                subject=subject,
                script_valid=script_valid,
                semantic_score=semantic_score,
                metadata={
                    "language_code": self.SUPPORTED_LANGUAGES[target_language]["code"],
                    "script": self.SUPPORTED_LANGUAGES[target_language]["script"],
                    "technical_terms_preserved": len(term_map),
                    "refinement_enabled": True,
                    "refinement_error": str(e),
                },
            )

    def validate_script_rendering(self, text: str, language: str) -> bool:
        """
        Validate that text uses correct Unicode script for the language.

        Args:
            text: Text to validate
            language: Target language

        Returns:
            True if script rendering is valid, False otherwise
        """
        if language not in self.SUPPORTED_LANGUAGES:
            return False

        lang_info = self.SUPPORTED_LANGUAGES[language]
        unicode_start, unicode_end = lang_info["unicode_range"]

        # Extract all characters in the language's script
        script_chars = []
        for char in text:
            code_point = ord(char)
            if unicode_start <= code_point <= unicode_end:
                script_chars.append(char)

        # Check if we have significant content in the correct script
        # Allow for punctuation, numbers, and English technical terms
        total_alpha_chars = sum(1 for c in text if c.isalpha())

        if total_alpha_chars == 0:
            return False

        # At least 50% of alphabetic characters should be in the target script
        script_ratio = (
            len(script_chars) / total_alpha_chars if total_alpha_chars > 0 else 0
        )

        is_valid = script_ratio >= 0.5

        logger.debug(
            f"Script validation for {language}: {len(script_chars)}/{total_alpha_chars} "
            f"chars in {lang_info['script']} script (ratio: {script_ratio:.2f})"
        )

        return is_valid

    def _mark_technical_terms(
        self, text: str, subject: str
    ) -> tuple[str, dict[str, str]]:
        """
        Mark technical terms in text for preservation during translation.

        Args:
            text: Source text
            subject: Subject area

        Returns:
            Tuple of (marked_text, term_map)
        """
        term_map = {}
        marked_text = text

        if subject not in self.TECHNICAL_TERMS:
            return text, term_map

        subject_terms = self.TECHNICAL_TERMS[subject]

        # Find and mark technical terms
        for term_en in subject_terms:
            # Case-insensitive search for the term
            pattern = re.compile(r"\b" + re.escape(term_en) + r"\b", re.IGNORECASE)
            matches = pattern.findall(text)

            if matches:
                # Create a unique marker for this term
                marker = f"__TERM_{len(term_map)}__"
                term_map[marker] = term_en
                marked_text = pattern.sub(marker, marked_text)

        logger.debug(f"Marked {len(term_map)} technical terms for preservation")

        return marked_text, term_map

    def _restore_technical_terms(
        self, text: str, term_map: dict[str, str], target_language: str, subject: str
    ) -> str:
        """
        Restore technical terms with correct translations.

        Args:
            text: Translated text with markers
            term_map: Map of markers to original terms
            target_language: Target language
            subject: Subject area

        Returns:
            Text with technical terms properly translated
        """
        restored_text = text

        if subject not in self.TECHNICAL_TERMS:
            return text

        subject_terms = self.TECHNICAL_TERMS[subject]

        # Replace markers with translated technical terms
        for marker, term_en in term_map.items():
            if term_en in subject_terms:
                term_translations = subject_terms[term_en]
                if target_language in term_translations:
                    translated_term = term_translations[target_language]
                    restored_text = restored_text.replace(marker, translated_term)
                else:
                    # Fallback to English term if translation not available
                    restored_text = restored_text.replace(marker, term_en)

        return restored_text

    def _calculate_semantic_equivalence(
        self, source_text: str, translated_text: str
    ) -> float:
        """
        Calculate semantic equivalence between source and translated text.

        Args:
            source_text: Original text
            translated_text: Translated text

        Returns:
            Semantic similarity score (0-1)
        """
        # Placeholder implementation
        # In production, this would use BERT embeddings for semantic comparison

        # Simple heuristic: check if lengths are proportional
        source_len = len(source_text.split())
        translated_len = len(translated_text.split())

        if source_len == 0:
            return 0.0

        # Expect translated text to be within 50-200% of source length
        length_ratio = translated_len / source_len

        if 0.5 <= length_ratio <= 2.0:
            # Good length ratio suggests reasonable translation
            score = 0.85
        elif 0.3 <= length_ratio <= 3.0:
            # Acceptable but not ideal
            score = 0.70
        else:
            # Suspicious length difference
            score = 0.50

        logger.debug(
            f"Semantic equivalence (heuristic): {score:.2f} "
            f"(length ratio: {length_ratio:.2f})"
        )

        return score

    def _fallback_translation(self, text: str, target_language: str) -> str:
        """
        Fallback translation when model is not available.

        This is a placeholder that returns a marked version of the text.
        In production, this could use a simpler translation service or
        return the original text with a warning.

        Args:
            text: Source text
            target_language: Target language

        Returns:
            Translated text (or marked original)
        """
        logger.warning(
            f"Using fallback translation for {target_language}. "
            "Model-based translation not available."
        )

        # Return original text with language marker
        return f"[{target_language}] {text}"

    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported languages.

        Returns:
            List of supported language names
        """
        return list(self.SUPPORTED_LANGUAGES.keys())

    def get_language_info(self, language: str) -> dict[str, Any] | None:
        """
        Get information about a supported language.

        Args:
            language: Language name

        Returns:
            Dictionary with language information or None if not supported
        """
        return self.SUPPORTED_LANGUAGES.get(language)
