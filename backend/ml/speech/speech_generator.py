"""Speech Generator component for converting text to audio in multiple Indian languages.

Hardware Optimization:
- Uses MPS (Metal Performance Shaders) on Apple Silicon for GPU acceleration
- Falls back to CPU for unsupported operations
- Optimized for M4 Mac with unified memory
"""

import hashlib
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Audio processing libraries
try:
    from pydub import AudioSegment
    from pydub.effects import normalize
except ImportError:
    AudioSegment = None
    normalize = None

# ASR validation
try:
    import speech_recognition as sr
except ImportError:
    sr = None

# Try to use local MMS-TTS service first (GPU accelerated)
try:
    from .tts.mms_tts import MMSTTSService

    LOCAL_TTS_AVAILABLE = True
except ImportError:
    LOCAL_TTS_AVAILABLE = False

# Fallback to API-based client - use async client with sync wrapper
try:
    import asyncio

    from ..pipeline.model_clients_async import MMSTTSAsyncClient

    class MMSTTSClient:
        """Sync wrapper around MMSTTSAsyncClient for backward compatibility."""

        def __init__(self):
            self._async_client = MMSTTSAsyncClient()

        def process(self, text: str, language: str = "Hindi", **kwargs) -> bytes:
            """Synchronously generate speech."""
            try:
                try:
                    asyncio.get_running_loop()
                    # Already in async context, use shared ML pool thread
                    from ..core.optimized.thread_pool_manager import get_ml_executor

                    pool = get_ml_executor()
                    future = pool.submit(
                        asyncio.run,
                        self._async_client.process(text, language, **kwargs),
                    )
                    return bytes(future.result(timeout=60))
                except RuntimeError:
                    # No running loop, create one
                    return bytes(
                        asyncio.run(
                            self._async_client.process(text, language, **kwargs)
                        )
                    )
            except (RuntimeError, ValueError, OSError) as e:
                logging.getLogger(__name__).warning(f"TTS client error: {e}")
                return b""
except ImportError:
    MMSTTSClient = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

@dataclass
class AudioFile:
    """Represents an audio file with metadata."""

    content: bytes
    format: str
    size_mb: float
    duration_seconds: float
    sample_rate: int
    file_path: str | None = None
    language: str | None = None
    accuracy_score: float | None = None

class TechnicalTermHandler:
    """Handles pronunciation of technical terms in Indic languages."""

    def __init__(self):
        self.term_mappings = self._load_technical_terms()

    def _load_technical_terms(self) -> dict[str, dict[str, dict[str, str]]]:
        """Load technical term pronunciation mappings for each language."""
        # This would typically load from a configuration file or database
        return {
            "Mathematics": {
                "Hindi": {
                    "equation": "समीकरण",
                    "algebra": "बीजगणित",
                    "geometry": "ज्यामिति",
                    "calculus": "कैलकुलस",
                    "theorem": "प्रमेय",
                    "coefficient": "गुणांक",
                    "polynomial": "बहुपद",
                },
                "Tamil": {
                    "equation": "சமன்பாடு",
                    "algebra": "இயற்கணிதம்",
                    "geometry": "வடிவியல்",
                    "calculus": "கணிதம்",
                    "theorem": "தேற்றம்",
                },
                "Telugu": {
                    "equation": "సమీకరణం",
                    "algebra": "బీజగణితం",
                    "geometry": "రేఖాగణితం",
                },
                "Bengali": {
                    "equation": "সমীকরণ",
                    "algebra": "বীজগণিত",
                    "geometry": "জ্যামিতি",
                },
                "Marathi": {
                    "equation": "समीकरण",
                    "algebra": "बीजगणित",
                    "geometry": "भूमिती",
                },
            },
            "Science": {
                "Hindi": {
                    "photosynthesis": "प्रकाश संश्लेषण",
                    "molecule": "अणु",
                    "atom": "परमाणु",
                    "electron": "इलेक्ट्रॉन",
                    "nucleus": "नाभिक",
                    "chromosome": "गुणसूत्र",
                },
                "Tamil": {
                    "photosynthesis": "ஒளிச்சேர்க்கை",
                    "molecule": "மூலக்கூறு",
                    "atom": "அணு",
                    "electron": "எலக்ட்ரான்",
                },
                "Telugu": {
                    "photosynthesis": "కాంతిసంశ్లేషణ",
                    "molecule": "అణువు",
                    "atom": "పరమాణువు",
                },
                "Bengali": {
                    "photosynthesis": "সালোকসংশ্লেষণ",
                    "molecule": "অণু",
                    "atom": "পরমাণু",
                },
                "Marathi": {
                    "photosynthesis": "प्रकाशसंश्लेषण",
                    "molecule": "रेणू",
                    "atom": "अणू",
                },
            },
        }

    def process_technical_terms(self, text: str, language: str, subject: str) -> str:
        """Replace technical terms with language-appropriate pronunciations."""
        if subject not in self.term_mappings:
            return text

        if language not in self.term_mappings[subject]:
            return text

        processed_text = text
        term_map = self.term_mappings[subject][language]

        for english_term, local_term in term_map.items():
            # Replace English technical terms with local equivalents for better pronunciation
            processed_text = processed_text.replace(english_term, local_term)

        return processed_text

class AudioOptimizer:
    """Optimizes audio files for low-end devices and bandwidth constraints."""

    def __init__(self):
        self.target_size_mb = 5.0  # Target size for 10 minutes of audio
        self.min_quality = 32  # Minimum bitrate in kbps

    def optimize_for_low_end_devices(self, audio_content: bytes) -> bytes:
        """Optimize audio for devices with ≤2 GB RAM."""
        if not AudioSegment:
            logger.warning("pydub not available, returning original audio")
            return audio_content

        try:
            # Load audio
            audio = AudioSegment.from_file(io.BytesIO(audio_content))

            # Calculate target bitrate based on size constraints
            current_size_mb = len(audio_content) / (1024 * 1024)
            target_size_mb = min(self.target_size_mb, current_size_mb)

            # Calculate optimal bitrate
            duration_minutes = len(audio) / (1000 * 60)
            target_bitrate = int((target_size_mb * 8 * 1024) / duration_minutes)
            target_bitrate = max(target_bitrate, self.min_quality)

            # Optimize audio settings
            optimized_audio = audio.set_frame_rate(22050)  # Reduce sample rate
            optimized_audio = optimized_audio.set_channels(1)  # Mono audio

            # Normalize audio levels
            if normalize:
                optimized_audio = normalize(optimized_audio)

            # Export with optimized settings
            output_buffer = io.BytesIO()
            optimized_audio.export(
                output_buffer,
                format="mp3",
                bitrate=f"{target_bitrate}k",
                parameters=["-q:a", "2"],  # Good quality/size balance
            )

            return output_buffer.getvalue()

        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Audio optimization failed: %s", e)
            return audio_content

    def compress_audio(
        self, audio_content: bytes, compression_level: str = "medium"
    ) -> bytes:
        """Apply compression to reduce file size."""
        if not AudioSegment:
            return audio_content

        compression_settings = {
            "low": {"bitrate": "64k", "quality": "5"},
            "medium": {"bitrate": "48k", "quality": "3"},
            "high": {"bitrate": "32k", "quality": "2"},
        }

        settings = compression_settings.get(
            compression_level, compression_settings["medium"]
        )

        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_content))
            output_buffer = io.BytesIO()

            audio.export(
                output_buffer,
                format="mp3",
                bitrate=settings["bitrate"],
                parameters=["-q:a", settings["quality"]],
            )

            return output_buffer.getvalue()

        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Audio compression failed: %s", e)
            return audio_content

class ASRValidator:
    """Validates audio accuracy using Automatic Speech Recognition."""

    def __init__(self):
        self.recognizer = sr.Recognizer() if sr else None
        self.target_accuracy = 0.90  # 90% accuracy target

    def validate_audio_accuracy(
        self, audio_file: AudioFile, reference_text: str
    ) -> float:
        """Validate audio accuracy using ASR and return accuracy score."""
        if not self.recognizer or not sr:
            logger.warning(
                "speech_recognition not available, returning default accuracy"
            )
            return 0.85  # Return reasonable default

        try:
            # Convert audio to WAV format for speech recognition
            if AudioSegment:
                audio = AudioSegment.from_file(io.BytesIO(audio_file.content))
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
            else:
                wav_buffer = io.BytesIO(audio_file.content)

            # Perform speech recognition
            with sr.AudioFile(wav_buffer) as source:
                audio_data = self.recognizer.record(source)

            # Use Google Speech Recognition (supports multiple languages)
            language_codes = {
                "Hindi": "hi-IN",
                "Tamil": "ta-IN",
                "Telugu": "te-IN",
                "Bengali": "bn-IN",
                "Marathi": "mr-IN",
            }

            language_code = language_codes.get(audio_file.language or "", "hi-IN")

            try:
                recognized_text = self.recognizer.recognize_google(
                    audio_data, language=language_code
                )

                # Calculate accuracy by comparing with reference text
                accuracy = self._calculate_text_similarity(
                    recognized_text, reference_text
                )
                return accuracy

            except sr.UnknownValueError:
                logger.warning("Speech recognition could not understand audio")
                return 0.0
            except sr.RequestError as e:
                logger.error("Speech recognition service error: %s", e)
                return 0.0

        except (RuntimeError, ValueError, OSError) as e:
            logger.error("ASR validation failed: %s", e)
            return 0.0

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple word-based similarity calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

class SpeechGenerator:
    """Main Speech Generator class for converting text to audio.

    Uses local MMS-TTS service with MPS (Metal) acceleration on Apple Silicon.
    Falls back to HuggingFace API if local model unavailable.
    """

    def __init__(self):
        """Initialize Speech Generator with MMS-TTS service singleton."""
        # Prefer local GPU-accelerated TTS (use singleton)
        if LOCAL_TTS_AVAILABLE:
            try:
                from .tts import get_mms_tts_service

                self._local_tts = get_mms_tts_service()
                self.tts_client = None  # Don't use API client
                logger.info(
                    "Using local MMS-TTS service singleton (MPS GPU accelerated)"
                )
            except (ImportError, RuntimeError) as e:
                logger.warning("Local TTS initialization failed: %s", e)
                self._local_tts = None
                self.tts_client = MMSTTSClient()
        else:
            self._local_tts = None
            self.tts_client = MMSTTSClient()
            logger.info("Using HuggingFace API for TTS")

        self.term_handler = TechnicalTermHandler()
        self.audio_optimizer = AudioOptimizer()
        self.asr_validator = ASRValidator()

        # Supported languages
        self.supported_languages = [
            "Hindi",
            "Tamil",
            "Telugu",
            "Bengali",
            "Marathi",
            "Gujarati",
            "Kannada",
            "Malayalam",
            "Punjabi",
            "Urdu",
        ]

        # Audio storage directory
        self.audio_dir = Path(os.getenv("AUDIO_STORAGE_DIR", "data/audio"))
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def generate_speech(self, text: str, language: str, subject: str) -> AudioFile:
        """Generate speech audio from text with optimization and validation.

        Uses local MMS-TTS with MPS acceleration on Apple Silicon for fast inference.
        """
        if language not in self.supported_languages:
            raise ValueError(
                f"Language '{language}' not supported. Supported: {self.supported_languages}"
            )

        logger.info("Generating speech for %s text in %s", language, subject)

        try:
            # Process technical terms for better pronunciation
            processed_text = self.term_handler.process_technical_terms(
                text, language, subject
            )

            # Generate audio - prefer local GPU-accelerated TTS
            if self._local_tts is not None:
                try:
                    audio_content = self._local_tts.synthesize(processed_text, language)
                    logger.info("Speech generated using local MPS-accelerated TTS")
                except (RuntimeError, ValueError, OSError) as e:
                    logger.warning("Local TTS failed, falling back to API: %s", e)
                    if self.tts_client:
                        audio_content = self.tts_client.process(
                            processed_text, language
                        )
                    else:
                        raise RuntimeError(f"No TTS backend available: {e}")
            else:
                audio_content = self.tts_client.process(processed_text, language)

            if not audio_content:
                raise RuntimeError("TTS service returned empty audio content")

            # Create initial AudioFile object
            audio_file = self._create_audio_file(audio_content, language)

            # Optimize audio for low-end devices
            optimized_content = self.audio_optimizer.optimize_for_low_end_devices(
                audio_content
            )
            audio_file.content = optimized_content
            audio_file.size_mb = len(optimized_content) / (1024 * 1024)

            # Validate audio accuracy using ASR
            accuracy_score = self.asr_validator.validate_audio_accuracy(
                audio_file, text
            )
            audio_file.accuracy_score = accuracy_score

            # Check if accuracy meets requirements (≥90%)
            if accuracy_score < self.asr_validator.target_accuracy:
                logger.warning(
                    f"Audio accuracy {accuracy_score:.2%} below target {self.asr_validator.target_accuracy:.2%}"
                )

            # Save audio file
            audio_file.file_path = self._save_audio_file(
                audio_file, text, language, subject
            )

            logger.info(
                f"Speech generation completed. Size: {audio_file.size_mb:.2f}MB, Accuracy: {accuracy_score:.2%}"
            )

            return audio_file

        except Exception as e:
            logger.error("Speech generation failed: %s", e)
            raise RuntimeError(f"Failed to generate speech: {e}")

    def _create_audio_file(self, audio_content: bytes, language: str) -> AudioFile:
        """Create AudioFile object with metadata."""
        size_mb = len(audio_content) / (1024 * 1024)

        # Extract audio metadata if possible
        duration_seconds = 0.0
        sample_rate = 22050  # Default

        if AudioSegment:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_content))
                duration_seconds = len(audio) / 1000.0  # Convert ms to seconds
                sample_rate = audio.frame_rate
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Could not extract audio metadata: %s", e)

        return AudioFile(
            content=audio_content,
            format="mp3",
            size_mb=size_mb,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            language=language,
        )

    def _save_audio_file(
        self, audio_file: AudioFile, text: str, language: str, subject: str
    ) -> str:
        """Save audio file to storage and return file path."""
        # Create filename based on content hash
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]
        filename = f"{language}_{subject}_{text_hash}.mp3"
        file_path = self.audio_dir / filename

        # Save audio content
        with open(file_path, "wb") as f:
            f.write(audio_file.content)

        return str(file_path)

    def validate_audio_quality(self, audio_file: AudioFile) -> bool:
        """Validate that audio meets quality requirements."""
        # Check file size (should be optimized for low-end devices)
        if audio_file.size_mb > 5.0:  # 5MB for 10 minutes target
            logger.warning("Audio file size %.2fMB exceeds target", audio_file.size_mb)
            return False

        # Check accuracy score
        if audio_file.accuracy_score and audio_file.accuracy_score < 0.90:
            logger.warning(
                f"Audio accuracy {audio_file.accuracy_score:.2%} below 90% target"
            )
            return False

        return True

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        return list(self.supported_languages)

    def estimate_audio_size(self, text: str) -> float:
        """Estimate audio file size in MB for given text."""
        # Rough estimation: ~1MB per minute of speech, ~150 words per minute
        word_count = len(text.split())
        estimated_minutes = word_count / 150
        estimated_size_mb = estimated_minutes * 1.0  # Before optimization

        # Account for optimization (typically reduces size by 60-70%)
        optimized_size_mb = estimated_size_mb * 0.35

        return optimized_size_mb
