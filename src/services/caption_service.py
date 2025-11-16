"""Caption generation service using Whisper ASR."""
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import timedelta
import tempfile

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import webvtt
    WEBVTT_AVAILABLE = True
except ImportError:
    WEBVTT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Caption:
    """Single caption with timing."""
    start: float
    end: float
    text: str
    
    def to_vtt_format(self) -> str:
        """Convert to WebVTT format."""
        start_time = self._format_time(self.start)
        end_time = self._format_time(self.end)
        return f"{start_time} --> {end_time}\n{self.text}"
    
    def to_srt_format(self, index: int) -> str:
        """Convert to SRT format."""
        start_time = self._format_time(self.start, use_comma=True)
        end_time = self._format_time(self.end, use_comma=True)
        return f"{index}\n{start_time} --> {end_time}\n{self.text}"
    
    @staticmethod
    def _format_time(seconds: float, use_comma: bool = False) -> str:
        """Format time as HH:MM:SS.mmm or HH:MM:SS,mmm."""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        
        separator = ',' if use_comma else '.'
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


@dataclass
class CaptionResult:
    """Caption generation result."""
    captions: List[Caption]
    vtt_path: Optional[str]
    srt_path: Optional[str]
    language: str
    detected_language: Optional[str]
    num_segments: int
    duration: float
    confidence: float


class WhisperCaptionService:
    """Service for generating captions from audio using Whisper."""
    
    # Language code mapping (Whisper uses 2-letter ISO codes)
    LANGUAGE_MAP = {
        'Hindi': 'hi',
        'Tamil': 'ta',
        'Telugu': 'te',
        'Bengali': 'bn',
        'Marathi': 'mr',
        'Gujarati': 'gu',
        'Kannada': 'kn',
        'Malayalam': 'ml',
        'Punjabi': 'pa',
        'English': 'en',
        'Odia': 'or',
        'Urdu': 'ur'
    }
    
    def __init__(
        self,
        model_size: str = 'base',
        output_dir: str = 'data/captions',
        device: str = 'cpu'
    ):
        """
        Initialize Whisper caption service.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            output_dir: Directory for caption files
            device: Device to use ('cpu', 'cuda')
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper not available. Install: pip install openai-whisper")
        
        if not WEBVTT_AVAILABLE:
            logger.warning("webvtt-py not available. Install: pip install webvtt-py")
        
        self.model_size = model_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load model
        logger.info(f"Loading Whisper {model_size} model on {device}...")
        self.model = whisper.load_model(model_size, device=device)
        logger.info("Whisper model loaded")
    
    def generate_captions(
        self,
        audio_path: str,
        language: Optional[str] = None,
        format: str = 'both',
        max_line_length: int = 42,
        min_duration: float = 1.0,
        max_duration: float = 5.0
    ) -> CaptionResult:
        """
        Generate captions from audio file.
        
        Args:
            audio_path: Path to audio file
            language: Target language (auto-detect if None)
            format: Output format ('vtt', 'srt', or 'both')
            max_line_length: Maximum characters per caption line
            min_duration: Minimum caption duration in seconds
            max_duration: Maximum caption duration in seconds
            
        Returns:
            CaptionResult with captions and file paths
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Convert language name to code
        language_code = None
        if language:
            language_code = self.LANGUAGE_MAP.get(language, language.lower()[:2])
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Transcribe with Whisper
        transcribe_options = {
            'task': 'transcribe',
            'verbose': False,
            'word_timestamps': True
        }
        
        if language_code:
            transcribe_options['language'] = language_code
        
        result = self.model.transcribe(audio_path, **transcribe_options)
        
        detected_language = result.get('language', 'unknown')
        segments = result.get('segments', [])
        
        logger.info(f"Transcription complete. Language: {detected_language}, Segments: {len(segments)}")
        
        # Process segments into captions
        captions = self._process_segments(
            segments,
            max_line_length=max_line_length,
            min_duration=min_duration,
            max_duration=max_duration
        )
        
        # Calculate average confidence
        confidences = [seg.get('avg_logprob', 0) for seg in segments]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        # Convert log probability to confidence score (0-1)
        confidence = min(1.0, max(0.0, (avg_confidence + 1.0)))
        
        # Get audio duration
        duration = result.get('segments', [{}])[-1].get('end', 0) if segments else 0
        
        # Generate output files
        base_name = Path(audio_path).stem
        vtt_path = None
        srt_path = None
        
        if format in ['vtt', 'both']:
            vtt_path = str(self.output_dir / f"{base_name}.vtt")
            self._write_vtt(captions, vtt_path)
            logger.info(f"VTT captions saved: {vtt_path}")
        
        if format in ['srt', 'both']:
            srt_path = str(self.output_dir / f"{base_name}.srt")
            self._write_srt(captions, srt_path)
            logger.info(f"SRT captions saved: {srt_path}")
        
        return CaptionResult(
            captions=captions,
            vtt_path=vtt_path,
            srt_path=srt_path,
            language=language or detected_language,
            detected_language=detected_language,
            num_segments=len(captions),
            duration=duration,
            confidence=confidence
        )
    
    def _process_segments(
        self,
        segments: List[Dict[str, Any]],
        max_line_length: int,
        min_duration: float,
        max_duration: float
    ) -> List[Caption]:
        """Process Whisper segments into captions."""
        captions = []
        
        for seg in segments:
            text = seg.get('text', '').strip()
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            
            if not text:
                continue
            
            # Split long text into multiple captions
            if len(text) > max_line_length:
                words = seg.get('words', [])
                
                if words:
                    # Use word timestamps for accurate splitting
                    sub_captions = self._split_by_words(
                        words,
                        max_line_length,
                        max_duration
                    )
                    captions.extend(sub_captions)
                else:
                    # Fallback: split by length
                    sub_captions = self._split_by_length(
                        text,
                        start,
                        end,
                        max_line_length
                    )
                    captions.extend(sub_captions)
            else:
                captions.append(Caption(
                    start=start,
                    end=end,
                    text=text
                ))
        
        # Merge very short captions
        captions = self._merge_short_captions(captions, min_duration)
        
        return captions
    
    def _split_by_words(
        self,
        words: List[Dict[str, Any]],
        max_length: int,
        max_duration: float
    ) -> List[Caption]:
        """Split words into captions based on timing and length."""
        captions = []
        current_words = []
        current_length = 0
        start_time = None
        
        for word_info in words:
            word = word_info.get('word', '').strip()
            word_start = word_info.get('start', 0)
            word_end = word_info.get('end', 0)
            
            if not word:
                continue
            
            if start_time is None:
                start_time = word_start
            
            # Check if adding this word exceeds limits
            new_length = current_length + len(word) + (1 if current_words else 0)
            duration = word_end - start_time
            
            if (new_length > max_length or duration > max_duration) and current_words:
                # Create caption from current words
                text = ' '.join(current_words)
                end_time = words[len(current_words) - 1].get('end', start_time)
                
                captions.append(Caption(
                    start=start_time,
                    end=end_time,
                    text=text
                ))
                
                # Reset
                current_words = [word]
                current_length = len(word)
                start_time = word_start
            else:
                current_words.append(word)
                current_length = new_length
        
        # Add remaining words
        if current_words:
            text = ' '.join(current_words)
            end_time = words[len(words) - 1].get('end', start_time)
            captions.append(Caption(
                start=start_time,
                end=end_time,
                text=text
            ))
        
        return captions
    
    def _split_by_length(
        self,
        text: str,
        start: float,
        end: float,
        max_length: int
    ) -> List[Caption]:
        """Split text by length (fallback when no word timestamps)."""
        words = text.split()
        duration = end - start
        captions = []
        
        current_words = []
        current_length = 0
        
        for i, word in enumerate(words):
            new_length = current_length + len(word) + (1 if current_words else 0)
            
            if new_length > max_length and current_words:
                # Estimate timing proportionally
                word_ratio = len(current_words) / len(words)
                caption_end = start + (duration * word_ratio)
                
                captions.append(Caption(
                    start=start,
                    end=caption_end,
                    text=' '.join(current_words)
                ))
                
                start = caption_end
                current_words = [word]
                current_length = len(word)
            else:
                current_words.append(word)
                current_length = new_length
        
        # Add remaining
        if current_words:
            captions.append(Caption(
                start=start,
                end=end,
                text=' '.join(current_words)
            ))
        
        return captions
    
    def _merge_short_captions(
        self,
        captions: List[Caption],
        min_duration: float
    ) -> List[Caption]:
        """Merge captions that are too short."""
        if not captions:
            return []
        
        merged = []
        current = captions[0]
        
        for next_caption in captions[1:]:
            current_duration = current.end - current.start
            
            if current_duration < min_duration:
                # Merge with next
                current = Caption(
                    start=current.start,
                    end=next_caption.end,
                    text=f"{current.text} {next_caption.text}"
                )
            else:
                merged.append(current)
                current = next_caption
        
        merged.append(current)
        return merged
    
    def _write_vtt(self, captions: List[Caption], output_path: str):
        """Write captions to WebVTT file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for caption in captions:
                f.write(caption.to_vtt_format())
                f.write("\n\n")
    
    def _write_srt(self, captions: List[Caption], output_path: str):
        """Write captions to SRT file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, caption in enumerate(captions, 1):
                f.write(caption.to_srt_format(i))
                f.write("\n\n")
    
    @staticmethod
    def validate_captions(captions: List[Caption]) -> bool:
        """Validate caption timing and content."""
        if not captions:
            return False
        
        for i, caption in enumerate(captions):
            # Check timing
            if caption.end <= caption.start:
                logger.warning(f"Invalid timing at caption {i}: {caption.start} -> {caption.end}")
                return False
            
            # Check text
            if not caption.text or not caption.text.strip():
                logger.warning(f"Empty text at caption {i}")
                return False
            
            # Check overlap with next caption
            if i < len(captions) - 1:
                next_caption = captions[i + 1]
                if caption.end > next_caption.start:
                    logger.warning(f"Overlap at captions {i}-{i+1}")
                    return False
        
        return True


# Export
__all__ = ['WhisperCaptionService', 'CaptionResult', 'Caption']
