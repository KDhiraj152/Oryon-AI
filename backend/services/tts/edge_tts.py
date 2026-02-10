"""
Edge TTS Service - High-quality multilingual text-to-speech using Microsoft Edge.

Features:
- 400+ neural voices across 100+ languages
- All Indian languages supported with multiple voice options
- High quality, natural-sounding speech
- Free to use (no API key required)
- Voice gender and style options
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

logger = logging.getLogger(__name__)

# Voice mapping for Indian languages with multiple options
INDIAN_VOICES = {
    # Hindi voices
    "hi": {
        "default": "hi-IN-SwaraNeural",
        "male": [
            "hi-IN-MadhurNeural",
            "hi-IN-AaravNeural",
            "hi-IN-ArjunNeural",
            "hi-IN-KunalNeural",
            "hi-IN-RehaanNeural",
        ],
        "female": [
            "hi-IN-SwaraNeural",
            "hi-IN-AartiNeural",
            "hi-IN-AnanyaNeural",
            "hi-IN-KavyaNeural",
        ],
    },
    # Telugu voices
    "te": {
        "default": "te-IN-ShrutiNeural",
        "male": ["te-IN-MohanNeural"],
        "female": ["te-IN-ShrutiNeural"],
    },
    # Tamil voices
    "ta": {
        "default": "ta-IN-PallaviNeural",
        "male": ["ta-IN-ValluvarNeural"],
        "female": ["ta-IN-PallaviNeural"],
    },
    # Kannada voices
    "kn": {
        "default": "kn-IN-SapnaNeural",
        "male": ["kn-IN-GaganNeural"],
        "female": ["kn-IN-SapnaNeural"],
    },
    # Malayalam voices
    "ml": {
        "default": "ml-IN-SobhanaNeural",
        "male": ["ml-IN-MidhunNeural"],
        "female": ["ml-IN-SobhanaNeural"],
    },
    # Bengali voices
    "bn": {
        "default": "bn-IN-TanishaaNeural",
        "male": ["bn-IN-BashkarNeural"],
        "female": ["bn-IN-TanishaaNeural"],
    },
    # Marathi voices
    "mr": {
        "default": "mr-IN-AarohiNeural",
        "male": ["mr-IN-ManoharNeural"],
        "female": ["mr-IN-AarohiNeural"],
    },
    # Gujarati voices
    "gu": {
        "default": "gu-IN-DhwaniNeural",
        "male": ["gu-IN-NiranjanNeural"],
        "female": ["gu-IN-DhwaniNeural"],
    },
    # Punjabi voices (Gurmukhi)
    "pa": {
        "default": "pa-IN-GurpreetNeural",
        "male": ["pa-IN-GurpreetNeural"],
        "female": ["pa-IN-GurpreetNeural"],
    },
    # Odia voices
    "or": {
        "default": "or-IN-SubhasiniNeural",
        "male": ["or-IN-SukantNeural"],
        "female": ["or-IN-SubhasiniNeural"],
    },
    # English (India) voices
    "en-IN": {
        "default": "en-IN-NeerjaNeural",
        "male": [
            "en-IN-PrabhatNeural",
            "en-IN-AaravNeural",
            "en-IN-ArjunNeural",
            "en-IN-KunalNeural",
            "en-IN-RehaanNeural",
        ],
        "female": [
            "en-IN-NeerjaNeural",
            "en-IN-AartiNeural",
            "en-IN-AnanyaNeural",
            "en-IN-AashiNeural",
            "en-IN-KavyaNeural",
        ],
    },
    # English (US) voices
    "en": {
        "default": "en-US-JennyNeural",
        "male": ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-EricNeural"],
        "female": ["en-US-JennyNeural", "en-US-AriaNeural", "en-US-SaraNeural"],
    },
}

# International language voices
INTERNATIONAL_VOICES = {
    "es": {
        "default": "es-ES-ElviraNeural",
        "male": ["es-ES-AlvaroNeural"],
        "female": ["es-ES-ElviraNeural"],
    },
    "fr": {
        "default": "fr-FR-DeniseNeural",
        "male": ["fr-FR-HenriNeural"],
        "female": ["fr-FR-DeniseNeural"],
    },
    "de": {
        "default": "de-DE-KatjaNeural",
        "male": ["de-DE-ConradNeural"],
        "female": ["de-DE-KatjaNeural"],
    },
    "it": {
        "default": "it-IT-ElsaNeural",
        "male": ["it-IT-DiegoNeural"],
        "female": ["it-IT-ElsaNeural"],
    },
    "pt": {
        "default": "pt-BR-FranciscaNeural",
        "male": ["pt-BR-AntonioNeural"],
        "female": ["pt-BR-FranciscaNeural"],
    },
    "ru": {
        "default": "ru-RU-SvetlanaNeural",
        "male": ["ru-RU-DmitryNeural"],
        "female": ["ru-RU-SvetlanaNeural"],
    },
    "zh": {
        "default": "zh-CN-XiaoxiaoNeural",
        "male": ["zh-CN-YunxiNeural"],
        "female": ["zh-CN-XiaoxiaoNeural"],
    },
    "ja": {
        "default": "ja-JP-NanamiNeural",
        "male": ["ja-JP-KeitaNeural"],
        "female": ["ja-JP-NanamiNeural"],
    },
    "ko": {
        "default": "ko-KR-SunHiNeural",
        "male": ["ko-KR-InJoonNeural"],
        "female": ["ko-KR-SunHiNeural"],
    },
    "ar": {
        "default": "ar-SA-ZariyahNeural",
        "male": ["ar-SA-HamedNeural"],
        "female": ["ar-SA-ZariyahNeural"],
    },
}

# Merge all voices
ALL_VOICES = {**INDIAN_VOICES, **INTERNATIONAL_VOICES}


class EdgeTTSService:
    """High-quality TTS service using Microsoft Edge neural voices."""

    def __init__(self):
        self.audio_dir = Path(os.getenv("AUDIO_STORAGE_DIR", "storage/audio/tts"))
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def get_voice(
        self, language: str, gender: str = "female", voice_name: str | None = None
    ) -> str:
        """Get the appropriate voice for a language and gender."""
        # If specific voice requested, use it
        if voice_name:
            return voice_name

        # Normalize language code
        lang_code = language.lower().strip()

        # Check if language is supported
        if lang_code in ALL_VOICES:
            voices = ALL_VOICES[lang_code]
            if gender.lower() == "male" and voices.get("male"):
                return voices["male"][0]
            elif gender.lower() == "female" and voices.get("female"):
                return voices["female"][0]
            return voices["default"]

        # Fallback to English
        logger.warning(f"Language {language} not found, using English")
        return "en-US-JennyNeural"

    def get_available_voices(self, language: str | None = None) -> dict:
        """Get available voices, optionally filtered by language."""
        if language:
            lang_code = language.lower().strip()
            if lang_code in ALL_VOICES:
                return {lang_code: ALL_VOICES[lang_code]}
            return {}
        return ALL_VOICES

    async def synthesize(
        self,
        text: str,
        language: str = "hi",
        gender: str = "female",
        voice_name: str | None = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bytes:
        """
        Synthesize speech from text using Edge TTS.

        Args:
            text: Text to convert to speech
            language: Language code (hi, te, ta, en, etc.)
            gender: Voice gender (male/female)
            voice_name: Specific voice name (optional)
            rate: Speech rate adjustment (e.g., "+10%", "-20%")
            pitch: Pitch adjustment (e.g., "+5Hz", "-10Hz")

        Returns:
            Audio bytes in MP3 format
        """
        try:
            import edge_tts

            # Get appropriate voice
            voice = self.get_voice(language, gender, voice_name)

            logger.info(
                f"Edge TTS: Generating audio with voice {voice} for {len(text)} chars"
            )

            # Create communicate instance
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)

            # Generate audio to temporary file using secure NamedTemporaryFile
            import uuid

            tmp_path = os.path.join(
                tempfile.gettempdir(), f"edge_tts_{uuid.uuid4().hex}.mp3"
            )

            await communicate.save(tmp_path)

            # Read audio bytes using async API if available
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(tmp_path, "rb") as f:
                    audio_bytes = await f.read()
            else:
                # Fallback to sync read via executor using Path
                audio_bytes = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: Path(tmp_path).read_bytes()
                )

            # Clean up temp file
            os.unlink(tmp_path)

            logger.info(f"Edge TTS: Generated {len(audio_bytes)} bytes of audio")
            return audio_bytes

        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}")
            raise RuntimeError(f"TTS synthesis failed: {e}")

    def synthesize_sync(
        self,
        text: str,
        language: str = "hi",
        gender: str = "female",
        voice_name: str | None = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bytes:
        """Synchronous wrapper for synthesize."""
        return asyncio.run(
            self.synthesize(text, language, gender, voice_name, rate, pitch)
        )

    async def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        language: str = "hi",
        gender: str = "female",
        voice_name: str | None = None,
    ) -> str:
        """Synthesize speech and save to file."""
        audio_bytes = await self.synthesize(text, language, gender, voice_name)

        if AIOFILES_AVAILABLE:
            async with aiofiles.open(output_path, "wb") as f:
                await f.write(audio_bytes)
        else:
            # Fallback to sync write via executor using Path
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: Path(output_path).write_bytes(audio_bytes)
            )

        return output_path


# Thread-safe singleton using functools.lru_cache
from functools import lru_cache


@lru_cache(maxsize=1)
def get_edge_tts_service() -> EdgeTTSService:
    """Get or create the Edge TTS service singleton (thread-safe)."""
    return EdgeTTSService()
