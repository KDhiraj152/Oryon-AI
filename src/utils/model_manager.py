"""Enhanced ML model manager with all models."""
import os
import logging
from typing import Optional, Dict, Any, List
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """Centralized ML model management."""
    
    def __init__(self, cache_dir: str = "data/models"):
        """Initialize model manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model Manager initialized on device: {self.device}")
        
        self._models = {}
    
    def load_flan_t5(self, model_size: str = "base") -> Any:
        """Load Flan-T5 for text simplification."""
        key = f"flan-t5-{model_size}"
        
        if key in self._models:
            return self._models[key]
        
        logger.info(f"Loading Flan-T5 {model_size}...")
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = f"google/flan-t5-{model_size}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self._models[key] = {"model": model, "tokenizer": tokenizer}
        logger.info(f"Flan-T5 {model_size} loaded")
        
        return self._models[key]
    
    def load_indictrans2(self) -> Any:
        """Load IndicTrans2 for Indian language translation."""
        key = "indictrans2"
        
        if key in self._models:
            return self._models[key]
        
        logger.info("Loading IndicTrans2...")
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = "ai4bharat/indictrans2-en-indic-1B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self._models[key] = {"model": model, "tokenizer": tokenizer}
        logger.info("IndicTrans2 loaded")
        
        return self._models[key]
    
    def load_bert_multilingual(self) -> Any:
        """Load BERT for semantic validation."""
        key = "bert-multilingual"
        
        if key in self._models:
            return self._models[key]
        
        logger.info("Loading BERT multilingual...")
        
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        ).to(self.device)
        
        self._models[key] = {"model": model, "tokenizer": tokenizer}
        logger.info("BERT multilingual loaded")
        
        return self._models[key]
    
    def load_sentence_bert(self) -> Any:
        """Load Sentence-BERT for semantic similarity."""
        key = "sentence-bert"
        
        if key in self._models:
            return self._models[key]
        
        logger.info("Loading Sentence-BERT...")
        
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(
            'paraphrase-multilingual-mpnet-base-v2',
            cache_folder=str(self.cache_dir),
            device=self.device
        )
        
        self._models[key] = model
        logger.info("Sentence-BERT loaded")
        
        return self._models[key]
    
    def load_vits_tts(self, language: str = "hin") -> Any:
        """Load VITS TTS model."""
        key = f"vits-{language}"
        
        if key in self._models:
            return self._models[key]
        
        logger.info(f"Loading VITS TTS for {language}...")
        
        from transformers import VitsModel, AutoTokenizer
        
        model_name = f"facebook/mms-tts-{language}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        model = VitsModel.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        ).to(self.device)
        
        self._models[key] = {"model": model, "tokenizer": tokenizer}
        logger.info(f"VITS TTS {language} loaded")
        
        return self._models[key]
    
    def load_coqui_tts(self) -> Any:
        """Load Coqui TTS for high-quality speech synthesis."""
        key = "coqui-tts"
        
        if key in self._models:
            return self._models[key]
        
        logger.info("Loading Coqui TTS...")
        
        from TTS.api import TTS
        
        # Use multilingual model
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False
        ).to(self.device)
        
        self._models[key] = tts
        logger.info("Coqui TTS loaded")
        
        return self._models[key]
    
    def load_whisper(self, model_size: str = "base") -> Any:
        """Load Whisper ASR model."""
        key = f"whisper-{model_size}"
        
        if key in self._models:
            return self._models[key]
        
        logger.info(f"Loading Whisper {model_size}...")
        
        import whisper
        
        model = whisper.load_model(model_size, device=self.device)
        
        self._models[key] = model
        logger.info(f"Whisper {model_size} loaded")
        
        return self._models[key]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "loaded_models": list(self._models.keys()),
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cache_dir": str(self.cache_dir)
        }
    
    def unload_model(self, key: str) -> bool:
        """Unload a specific model to free memory."""
        if key in self._models:
            del self._models[key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Model {key} unloaded")
            return True
        return False
    
    def unload_all(self):
        """Unload all models."""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models unloaded")


# Singleton instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get singleton model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Export
__all__ = ['ModelManager', 'get_model_manager']
