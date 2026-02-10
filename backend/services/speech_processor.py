"""Audio processing utilities for speech generation and optimization."""

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Constants
MP3_PATTERN = "*.mp3"

try:
    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range, normalize
    from pydub.silence import split_on_silence
except ImportError:
    AudioSegment = None
    normalize = None
    compress_dynamic_range = None
    split_on_silence = None

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio format conversion, compression, and optimization."""

    def __init__(self):
        self.supported_formats = ["mp3", "wav", "ogg", "m4a"]
        self.target_sample_rate = 22050  # Optimized for speech
        self.target_channels = 1  # Mono for smaller file size

    def convert_format(
        self, audio_content: bytes, source_format: str, target_format: str
    ) -> bytes:
        """Convert audio from one format to another."""
        if not AudioSegment:
            logger.warning("pydub not available, cannot convert audio format")
            return audio_content

        if target_format not in self.supported_formats:
            raise ValueError(f"Unsupported target format: {target_format}")

        try:
            # Load audio in source format
            audio = AudioSegment.from_file(
                io.BytesIO(audio_content), format=source_format
            )

            # Export in target format
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format=target_format)

            return output_buffer.getvalue()

        except Exception as e:
            logger.error(f"Audio format conversion failed: {e}")
            return audio_content

    def optimize_for_speech(self, audio_content: bytes) -> bytes:
        """Optimize audio specifically for speech content."""
        if not AudioSegment:
            return audio_content

        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_content))

            # Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Optimize sample rate for speech
            if audio.frame_rate != self.target_sample_rate:
                audio = audio.set_frame_rate(self.target_sample_rate)

            # Normalize audio levels
            if normalize:
                audio = normalize(audio)

            # Apply dynamic range compression for better clarity
            if compress_dynamic_range:
                audio = compress_dynamic_range(audio)

            # Remove silence at beginning and end
            audio = self._trim_silence(audio)

            # Export optimized audio
            output_buffer = io.BytesIO()
            audio.export(
                output_buffer, format="mp3", bitrate="48k", parameters=["-q:a", "2"]
            )

            return output_buffer.getvalue()

        except Exception as e:
            logger.error(f"Speech optimization failed: {e}")
            return audio_content

    def _trim_silence(self, audio: "AudioSegment") -> "AudioSegment":
        """Remove silence from beginning and end of audio."""
        try:
            # Remove silence with threshold of -40dB
            silence_threshold = -40
            chunk_size = 10  # ms

            # Find start of audio (first non-silent chunk)
            start_trim = 0
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                if chunk.dBFS > silence_threshold:
                    start_trim = i
                    break

            # Find end of audio (last non-silent chunk)
            end_trim = len(audio)
            for i in range(len(audio) - chunk_size, 0, -chunk_size):
                chunk = audio[i : i + chunk_size]
                if chunk.dBFS > silence_threshold:
                    end_trim = i + chunk_size
                    break

            return audio[start_trim:end_trim]

        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio

    def split_long_audio(
        self, audio_content: bytes, max_duration_minutes: float = 10.0
    ) -> list:
        """Split long audio into smaller chunks for better handling."""
        if not AudioSegment:
            return [audio_content]

        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_content))
            duration_minutes = len(audio) / (1000 * 60)

            if duration_minutes <= max_duration_minutes:
                return [audio_content]

            # Split audio into chunks
            max_duration_ms = int(max_duration_minutes * 60 * 1000)
            chunks = []

            for i in range(0, len(audio), max_duration_ms):
                chunk = audio[i : i + max_duration_ms]

                # Export chunk
                output_buffer = io.BytesIO()
                chunk.export(output_buffer, format="mp3")
                chunks.append(output_buffer.getvalue())

            return chunks

        except Exception as e:
            logger.error(f"Audio splitting failed: {e}")
            return [audio_content]

    def get_audio_info(self, audio_content: bytes) -> dict[str, Any]:
        """Extract metadata from audio content."""
        info = {
            "size_bytes": len(audio_content),
            "size_mb": len(audio_content) / (1024 * 1024),
            "duration_seconds": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "bitrate": 0,
            "format": "unknown",
        }

        if not AudioSegment:
            return info

        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_content))

            info.update(
                {
                    "duration_seconds": len(audio) / 1000.0,
                    "sample_rate": audio.frame_rate,
                    "channels": audio.channels,
                    "bitrate": audio.frame_rate
                    * audio.frame_width
                    * 8
                    * audio.channels,
                    "format": "detected",
                }
            )

        except Exception as e:
            logger.warning(f"Could not extract audio info: {e}")

        return info

    def validate_audio_quality(
        self, audio_content: bytes, min_duration: float = 0.5
    ) -> tuple[bool, str]:
        """Validate audio quality and return status with message."""
        if not audio_content:
            return False, "Empty audio content"

        info = self.get_audio_info(audio_content)

        # Check minimum duration
        if info["duration_seconds"] < min_duration:
            return (
                False,
                f"Audio too short: {info['duration_seconds']:.1f}s < {min_duration}s",
            )

        # Check file size (should not be too large for low-end devices)
        if info["size_mb"] > 10.0:  # 10MB limit
            return False, f"Audio file too large: {info['size_mb']:.1f}MB > 10MB"

        # Check if audio has content (not just silence)
        if AudioSegment:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_content))
                if audio.dBFS < -60:  # Very quiet audio
                    return False, "Audio appears to be silent or very quiet"
            except Exception as e:
                logger.debug(f"Could not analyze audio volume: {e}")

        return True, "Audio quality validation passed"


class AudioCache:
    """Manages caching of generated audio files."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(
            cache_dir or os.getenv("AUDIO_CACHE_DIR", "data/cache/audio")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_mb = 1000  # 1GB cache limit

    def get_cache_key(self, text: str, language: str, subject: str) -> str:
        """Generate cache key for audio content."""
        import hashlib

        content = f"{text}_{language}_{subject}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    def get_cached_audio(self, cache_key: str) -> bytes | None:
        """Retrieve cached audio if available."""
        cache_file = self.cache_dir / f"{cache_key}.mp3"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read cached audio: {e}")

        return None

    def cache_audio(self, cache_key: str, audio_content: bytes) -> bool:
        """Cache audio content for future use."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.mp3"

            # Check cache size limit
            if self._get_cache_size_mb() > self.max_cache_size_mb:
                self._cleanup_old_files()

            with open(cache_file, "wb") as f:
                f.write(audio_content)

            logger.debug(f"Cached audio: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache audio: {e}")
            return False

    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB."""
        total_size = 0
        for file_path in self.cache_dir.glob(MP3_PATTERN):
            total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)

    def _cleanup_old_files(self):
        """Remove oldest cached files to free space."""
        cache_files = list(self.cache_dir.glob(MP3_PATTERN))

        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda f: f.stat().st_mtime)

        # Remove oldest 20% of files
        files_to_remove = cache_files[: len(cache_files) // 5]

        for file_path in files_to_remove:
            try:
                file_path.unlink()
                logger.debug(f"Removed old cached audio: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove cached file: {e}")

    def clear_cache(self):
        """Clear all cached audio files."""
        for file_path in self.cache_dir.glob(MP3_PATTERN):
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cached file: {e}")

        logger.info("Audio cache cleared")


class BatchAudioProcessor:
    """Handles batch processing of multiple audio files."""

    def __init__(self):
        self.processor = AudioProcessor()
        self.cache = AudioCache()

    def process_batch(self, audio_requests: list) -> list:
        """Process multiple audio generation requests."""
        results = []

        for request in audio_requests:
            try:
                # Check cache first
                cache_key = self.cache.get_cache_key(
                    request["text"], request["language"], request["subject"]
                )

                cached_audio = self.cache.get_cached_audio(cache_key)
                if cached_audio:
                    results.append(
                        {
                            "success": True,
                            "audio_content": cached_audio,
                            "cached": True,
                            "request": request,
                        }
                    )
                    continue

                # Process new request (this would integrate with SpeechGenerator)
                # For now, just return placeholder
                results.append(
                    {
                        "success": False,
                        "error": "Batch processing not fully implemented",
                        "cached": False,
                        "request": request,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "error": str(e),
                        "cached": False,
                        "request": request,
                    }
                )

        return results
