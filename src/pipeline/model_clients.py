"""Hugging Face model client wrappers with authentication and rate limiting."""
import os
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        now = time.time()
        # Remove calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.calls = []
        
        self.calls.append(now)


class BaseModelClient(ABC):
    """Base class for Hugging Face model clients."""
    
    def __init__(self, model_id: str, api_key: Optional[str] = None):
        self.model_id = model_id
        self.api_key = api_key or os.getenv('HUGGINGFACE_API_KEY')
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60)
        
        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with rate limiting and error handling."""
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.post(
                self.api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            
            if response.status_code == 503:
                # Model is loading, wait and retry
                time.sleep(20)
                response = self.session.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=30
                )
            
            # Check for deprecated/moved models (410)
            if response.status_code == 410:
                raise RuntimeError(f"Model {self.model_id} is deprecated or moved. Using fallback.")
            
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.HTTPError, RuntimeError) as e:
            # Log the error and raise for fallback handling
            raise RuntimeError(f"API request failed: {e}")
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Process input through the model."""
        pass


class FlanT5Client(BaseModelClient):
    """Client for Flan-T5 text simplification model."""
    
    def __init__(self, api_key: Optional[str] = None):
        model_id = os.getenv('FLANT5_MODEL_ID', 'google/flan-t5-base')
        super().__init__(model_id, api_key)
    
    def process(self, text: str, grade_level: int, subject: str) -> str:
        """Simplify text for the specified grade level."""
        try:
            prompt = self._create_prompt(text, grade_level, subject)
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 512,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            result = self._make_request(payload)
            return result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')
        except RuntimeError:
            # Fallback to rule-based simplification
            return self._fallback_simplification(text, grade_level)
    
    def _fallback_simplification(self, text: str, grade_level: int) -> str:
        """Simple rule-based text simplification as fallback."""
        # Basic simplification: shorter sentences, simpler words
        import re
        sentences = re.split(r'[.!?]+', text)
        simplified_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Break long sentences at conjunctions for lower grades
            if grade_level <= 8 and len(sentence.split()) > 20:
                parts = re.split(r'\s+(and|but|because|so|which)\s+', sentence, maxsplit=1)
                if len(parts) >= 3:
                    simplified_sentences.append(parts[0].strip() + '.')
                    simplified_sentences.append(parts[2].strip())
                else:
                    simplified_sentences.append(sentence)
            else:
                simplified_sentences.append(sentence)
        
        return '. '.join(simplified_sentences) + '.'
    
    def _create_prompt(self, text: str, grade_level: int, subject: str) -> str:
        """Create grade-appropriate simplification prompt."""
        return f"Simplify the following {subject} text for grade {grade_level} students: {text}"


class IndicTrans2Client(BaseModelClient):
    """Client for IndicTrans2 translation model."""
    
    def __init__(self, api_key: Optional[str] = None):
        model_id = os.getenv('INDICTRANS2_MODEL_ID', 'ai4bharat/indictrans2-en-indic-1B')
        super().__init__(model_id, api_key)
    
    def process(self, text: str, target_language: str) -> str:
        """Translate text to target Indian language."""
        try:
            payload = {
                "inputs": text,
                "parameters": {
                    "src_lang": "eng_Latn",
                    "tgt_lang": self._get_language_code(target_language)
                }
            }
            
            result = self._make_request(payload)
            return result[0]['translation_text'] if isinstance(result, list) else result.get('translation_text', '')
        except RuntimeError:
            # Return marked text as fallback
            return f"[Translation to {target_language}] {text}"
    
    def _get_language_code(self, language: str) -> str:
        """Map language names to IndicTrans2 codes."""
        language_map = {
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
        return language_map.get(language, 'hin_Deva')


class BERTClient(BaseModelClient):
    """Client for BERT validation model."""
    
    def __init__(self, api_key: Optional[str] = None):
        model_id = os.getenv('BERT_MODEL_ID', 'bert-base-multilingual-cased')
        super().__init__(model_id, api_key)
    
    def process(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            payload = {
                "inputs": {
                    "source_sentence": text1,
                    "sentences": [text2]
                }
            }
            
            result = self._make_request(payload)
            # Return similarity score (0-1)
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            return 0.0
        except RuntimeError:
            # Fallback heuristic based on word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0


class VITSClient(BaseModelClient):
    """Client for VITS text-to-speech model."""
    
    DEFAULT_TTS_MODEL = 'facebook/mms-tts-hin'
    
    def __init__(self, api_key: Optional[str] = None):
        model_id = os.getenv('VITS_MODEL_ID', self.DEFAULT_TTS_MODEL)
        super().__init__(model_id, api_key)
    
    def process(self, text: str, language: str) -> bytes:
        """Generate speech audio from text."""
        try:
            # Update model ID based on language
            self.model_id = self._get_model_for_language(language)
            self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            
            payload = {"inputs": text}
            
            response = self.session.post(
                self.api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=60
            )
            
            if response.status_code == 503:
                time.sleep(20)
                response = self.session.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60
                )
            
            # Check for deprecated models
            if response.status_code == 410:
                raise RuntimeError(f"TTS model {self.model_id} is deprecated")
            
            response.raise_for_status()
            return response.content
        except RuntimeError:
            # Return empty audio as fallback (or could generate silence)
            return b''  # Empty bytes for testing
    
    def _get_model_for_language(self, language: str) -> str:
        """Get appropriate TTS model for language."""
        model_map = {
            'Hindi': self.DEFAULT_TTS_MODEL,
            'Tamil': 'facebook/mms-tts-tam',
            'Telugu': 'facebook/mms-tts-tel',
            'Bengali': 'facebook/mms-tts-ben',
            'Marathi': 'facebook/mms-tts-mar'
        }
        return model_map.get(language, self.DEFAULT_TTS_MODEL)


class BhashiniTTSClient:
    """Client for Bhashini TTS API (alternative to VITS)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('BHASHINI_API_KEY')
        self.api_url = os.getenv('BHASHINI_API_URL', 'https://dhruva-api.bhashini.gov.in/services/inference/pipeline')
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60)
    
    def process(self, text: str, language: str) -> bytes:
        """Generate speech using Bhashini TTS."""
        self.rate_limiter.wait_if_needed()
        
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {
                            "sourceLanguage": self._get_language_code(language)
                        },
                        "gender": "female"
                    }
                }
            ],
            "inputData": {
                "input": [{"source": text}]
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        # Extract audio from response
        result = response.json()
        audio_content = result.get('pipelineResponse', [{}])[0].get('audio', [{}])[0].get('audioContent', '')
        
        # Decode base64 audio if needed
        import base64
        return base64.b64decode(audio_content) if audio_content else b''
    
    def _get_language_code(self, language: str) -> str:
        """Map language names to Bhashini codes."""
        language_map = {
            'Hindi': 'hi',
            'Tamil': 'ta',
            'Telugu': 'te',
            'Bengali': 'bn',
            'Marathi': 'mr'
        }
        return language_map.get(language, 'hi')
