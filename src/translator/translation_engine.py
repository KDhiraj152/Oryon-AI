"""Translation Engine component using IndicTrans2 for multi-language translation."""
import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy import to avoid loading model at module import time
_indictrans2_model = None

def get_indictrans2_model():
    """Get or create singleton IndicTrans2 model instance."""
    global _indictrans2_model
    if _indictrans2_model is None:
        try:
            from .indictrans2_model import IndicTrans2Model
            _indictrans2_model = IndicTrans2Model()
            logger.info("IndicTrans2 model client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize IndicTrans2 model: {e}")
            _indictrans2_model = None
    return _indictrans2_model


@dataclass
class TranslatedText:
    """Result of text translation."""
    text: str
    source_language: str
    target_language: str
    subject: str
    script_valid: bool
    semantic_score: float
    metadata: Dict[str, Any]


class TranslationEngine:
    """
    Translation engine component that converts educational content
    into Indian languages using IndicTrans2 model.
    
    Supports: Hindi, Tamil, Telugu, Bengali, Marathi (MVP)
    Expandable to 10+ Indian languages
    """
    
    # Supported languages in MVP phase
    SUPPORTED_LANGUAGES = {
        'Hindi': {
            'code': 'hin_Deva',
            'script': 'Devanagari',
            'unicode_range': (0x0900, 0x097F)
        },
        'Tamil': {
            'code': 'tam_Taml',
            'script': 'Tamil',
            'unicode_range': (0x0B80, 0x0BFF)
        },
        'Telugu': {
            'code': 'tel_Telu',
            'script': 'Telugu',
            'unicode_range': (0x0C00, 0x0C7F)
        },
        'Bengali': {
            'code': 'ben_Beng',
            'script': 'Bengali',
            'unicode_range': (0x0980, 0x09FF)
        },
        'Marathi': {
            'code': 'mar_Deva',
            'script': 'Devanagari',
            'unicode_range': (0x0900, 0x097F)
        }
    }
    
    # Subject-specific technical terminology mappings
    TECHNICAL_TERMS = {
        'Mathematics': {
            'equation': {'Hindi': 'समीकरण', 'Tamil': 'சமன்பாடு', 'Telugu': 'సమీకరణం', 'Bengali': 'সমীকরণ', 'Marathi': 'समीकरण'},
            'variable': {'Hindi': 'चर', 'Tamil': 'மாறி', 'Telugu': 'చరరాశి', 'Bengali': 'চলক', 'Marathi': 'चल'},
            'theorem': {'Hindi': 'प्रमेय', 'Tamil': 'தேற்றம்', 'Telugu': 'సిద్ధాంతం', 'Bengali': 'উপপাদ্য', 'Marathi': 'प्रमेय'},
            'function': {'Hindi': 'फलन', 'Tamil': 'சார்பு', 'Telugu': 'ఫంక్షన్', 'Bengali': 'অপেক্ষক', 'Marathi': 'फलन'},
            'graph': {'Hindi': 'आलेख', 'Tamil': 'வரைபடம்', 'Telugu': 'గ్రాఫ్', 'Bengali': 'লেখচিত্র', 'Marathi': 'आलेख'}
        },
        'Science': {
            'photosynthesis': {'Hindi': 'प्रकाश संश्लेषण', 'Tamil': 'ஒளிச்சேர்க்கை', 'Telugu': 'కిరణజన్య సంయోగక్రియ', 'Bengali': 'সালোকসংশ্লেষ', 'Marathi': 'प्रकाश संश्लेषण'},
            'molecule': {'Hindi': 'अणु', 'Tamil': 'மூலக்கூறு', 'Telugu': 'అణువు', 'Bengali': 'অণু', 'Marathi': 'रेणू'},
            'atom': {'Hindi': 'परमाणु', 'Tamil': 'அணு', 'Telugu': 'పరమాణువు', 'Bengali': 'পরমাণু', 'Marathi': 'अणु'},
            'cell': {'Hindi': 'कोशिका', 'Tamil': 'செல்', 'Telugu': 'కణం', 'Bengali': 'কোষ', 'Marathi': 'पेशी'},
            'energy': {'Hindi': 'ऊर्जा', 'Tamil': 'ஆற்றல்', 'Telugu': 'శక్తి', 'Bengali': 'শক্তি', 'Marathi': 'ऊर्जा'}
        },
        'Social Studies': {
            'democracy': {'Hindi': 'लोकतंत्र', 'Tamil': 'ஜனநாயகம்', 'Telugu': 'ప్రజాస్వామ్యం', 'Bengali': 'গণতন্ত্র', 'Marathi': 'लोकशाही'},
            'constitution': {'Hindi': 'संविधान', 'Tamil': 'அரசியலமைப்பு', 'Telugu': 'రాజ్యాంగం', 'Bengali': 'সংবিধান', 'Marathi': 'संविधान'},
            'government': {'Hindi': 'सरकार', 'Tamil': 'அரசாங்கம்', 'Telugu': 'ప్రభుత్వం', 'Bengali': 'সরকার', 'Marathi': 'सरकार'},
            'economy': {'Hindi': 'अर्थव्यवस्था', 'Tamil': 'பொருளாதாரம்', 'Telugu': 'ఆర్థిక వ్యవస్థ', 'Bengali': 'অর্থনীতি', 'Marathi': 'अर्थव्यवस्था'}
        }
    }
    
    def __init__(self, model_client=None):
        """
        Initialize the Translation Engine.
        
        Args:
            model_client: Optional IndicTrans2 model client for inference
                         If None, will attempt to use local IndicTrans2 model
        """
        # Use provided client or attempt to load local model
        if model_client is None:
            self.model_client = get_indictrans2_model()
        else:
            self.model_client = model_client
            
        logger.info("TranslationEngine initialized with support for: %s", 
                   ', '.join(self.SUPPORTED_LANGUAGES.keys()))
        if self.model_client:
            logger.info("IndicTrans2 model client available")
        else:
            logger.warning("No model client available, will use fallback translation")
    
    def translate(
        self,
        text: str,
        target_language: str,
        subject: str,
        source_language: str = 'English'
    ) -> TranslatedText:
        """
        Translate text to target Indian language.
        
        Args:
            text: Source text to translate
            target_language: Target Indian language (Hindi, Tamil, Telugu, Bengali, Marathi)
            subject: Subject area for technical terminology handling
            source_language: Source language (default: English)
        
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
        
        logger.info(f"Translating text to {target_language} for subject {subject}")
        
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
                translated_text = self._fallback_translation(text, target_language, subject)
        else:
            translated_text = self._fallback_translation(text, target_language, subject)
        
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
                'language_code': self.SUPPORTED_LANGUAGES[target_language]['code'],
                'script': self.SUPPORTED_LANGUAGES[target_language]['script'],
                'technical_terms_preserved': len(term_map)
            }
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
        unicode_start, unicode_end = lang_info['unicode_range']
        
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
        script_ratio = len(script_chars) / total_alpha_chars if total_alpha_chars > 0 else 0
        
        is_valid = script_ratio >= 0.5
        
        logger.debug(
            f"Script validation for {language}: {len(script_chars)}/{total_alpha_chars} "
            f"chars in {lang_info['script']} script (ratio: {script_ratio:.2f})"
        )
        
        return is_valid
    
    def _mark_technical_terms(
        self,
        text: str,
        subject: str
    ) -> tuple[str, Dict[str, str]]:
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
        for term_en in subject_terms.keys():
            # Case-insensitive search for the term
            pattern = re.compile(r'\b' + re.escape(term_en) + r'\b', re.IGNORECASE)
            matches = pattern.findall(text)
            
            if matches:
                # Create a unique marker for this term
                marker = f"__TERM_{len(term_map)}__"
                term_map[marker] = term_en
                marked_text = pattern.sub(marker, marked_text)
        
        logger.debug(f"Marked {len(term_map)} technical terms for preservation")
        
        return marked_text, term_map
    
    def _restore_technical_terms(
        self,
        text: str,
        term_map: Dict[str, str],
        target_language: str,
        subject: str
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
        self,
        source_text: str,
        translated_text: str
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
    
    def _fallback_translation(
        self,
        text: str,
        target_language: str,
        subject: str
    ) -> str:
        """
        Fallback translation when model is not available.
        
        This is a placeholder that returns a marked version of the text.
        In production, this could use a simpler translation service or
        return the original text with a warning.
        
        Args:
            text: Source text
            target_language: Target language
            subject: Subject area
        
        Returns:
            Translated text (or marked original)
        """
        logger.warning(
            f"Using fallback translation for {target_language}. "
            "Model-based translation not available."
        )
        
        # Return original text with language marker
        return f"[{target_language}] {text}"
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language names
        """
        return list(self.SUPPORTED_LANGUAGES.keys())
    
    def get_language_info(self, language: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a supported language.
        
        Args:
            language: Language name
        
        Returns:
            Dictionary with language information or None if not supported
        """
        return self.SUPPORTED_LANGUAGES.get(language)
