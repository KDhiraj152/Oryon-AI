"""Translation service with async support for streaming."""

import asyncio

from ...utils.logging import get_logger
from .engine import TranslationEngine

logger = get_logger(__name__)


class TranslationService:
    """Wrapper service for translation with async streaming support."""

    def __init__(self):
        self.engine = TranslationEngine()

    async def translate_async(
        self, text: str, source_lang: str, target_lang: str, subject: str = "General"
    ) -> str:
        """
        Async translation method for WebSocket streaming.

        Uses IndicTrans2 engine for English → Indian language translation.

        Args:
            text: Text to translate
            source_lang: Source language (e.g., "English")
            target_lang: Target language (e.g., "Hindi", "Telugu")
            subject: Subject area for technical terms (default: "General")

        Returns:
            Translated text string
        """

        def _translate():
            # TranslationEngine.translate expects: (text, target_language, subject, source_language)
            result = self.engine.translate(
                text=text,
                target_language=target_lang,
                subject=subject,
                source_language=source_lang,
            )
            # Result is a TranslatedText object, extract the text
            return (
                result.translated_text
                if hasattr(result, "translated_text")
                else str(result)
            )

        # Run synchronous translation in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _translate)
        return result
