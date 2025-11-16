"""Simple IndicTrans2/NLLB model wrapper for local translation."""
import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger(__name__)


class IndicTrans2Model:
    """Lightweight wrapper for IndicTrans2/NLLB translation model."""
    
    # Language code mappings for NLLB model
    NLLB_LANG_CODES = {
        'Hindi': 'hin_Deva',
        'Tamil': 'tam_Taml',
        'Telugu': 'tel_Telu',
        'Bengali': 'ben_Beng',
        'Marathi': 'mar_Deva',
        'Gujarati': 'guj_Gujr',
        'Kannada': 'kan_Knda',
        'Malayalam': 'mal_Mlym',
        'Punjabi': 'pan_Guru',
        'Urdu': 'urd_Arab'
    }
    
    def __init__(self, model_name: str = None):
        """
        Initialize translation model.
        
        Args:
            model_name: HuggingFace model identifier
                       Default: facebook/nllb-200-distilled-600M (open model)
        """
        if model_name is None:
            # Use NLLB as default (open, supports 200+ languages including all Indian languages)
            model_name = "facebook/nllb-200-distilled-600M"
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"IndicTrans2Model initialized with: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info("Model will be loaded on first translation request")
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            logger.info(f"Loading translation model: {self.model_name}")
            logger.info("First run may take a few minutes (~600MB download)...")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    src_lang="eng_Latn"  # English source
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name
                ).to(self.device)
                
                logger.info("Translation model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load translation model: {e}")
                raise
    
    def translate(self, text: str, target_language_code: str) -> str:
        """
        Translate text to target language.
        
        Args:
            text: Source text in English
            target_language_code: Language code (e.g., 'hin_Deva', 'tam_Taml')
        
        Returns:
            Translated text
        """
        # Load model if not already loaded
        if self.model is None:
            self._load_model()
        
        try:
            # Set target language for tokenizer
            self.tokenizer.src_lang = "eng_Latn"
            
            # Prepare input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get target language token ID
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_language_code)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode output
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Translation successful: {len(text)} chars -> {len(translated)} chars")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Return fallback
            return f"[Translation Error] {text}"
    
    def process(self, text: str, target_language: str) -> str:
        """
        Process method compatible with existing model client interface.
        
        Args:
            text: Source text
            target_language: Language name (Hindi, Tamil, etc.)
        
        Returns:
            Translated text
        """
        target_code = self.NLLB_LANG_CODES.get(target_language, 'hin_Deva')
        return self.translate(text, target_code)
