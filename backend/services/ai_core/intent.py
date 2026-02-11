"""
Intelligent Intent Classification
=================================

Uses the LLM to dynamically understand user intent without hardcoded patterns.
The system learns from the prompt context to determine:
- What the user wants to do
- What language they want responses in
- How complex/simple the response should be
- Whether specialized tools are needed (translation, TTS, code execution, etc.)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """High-level user intents - dynamically determined by LLM."""

    QUESTION = "question"  # General Q&A, explanations
    TRANSLATION = "translation"  # Translate content between languages
    SIMPLIFICATION = "simplification"  # Make content easier to understand
    AUDIO = "audio"  # Text-to-speech request
    QUIZ = "quiz"  # Generate quiz/questions
    CODE = "code"  # Code-related (explain, write, debug)
    CALCULATION = "calculation"  # Math/calculation requests
    CONVERSATION = "conversation"  # General chat/greeting
    CREATIVE = "creative"  # Creative writing, stories
    SUMMARIZATION = "summarization"  # Summarize content
    UNKNOWN = "unknown"  # Let the LLM decide freely


@dataclass
class IntentAnalysis:
    """Result of intelligent intent analysis."""

    primary_intent: Intent
    confidence: float  # 0.0 to 1.0

    # Extracted parameters (all optional, LLM-determined)
    target_language: str | None = None
    source_language: str | None = None
    content_to_process: str | None = None

    # Response preferences
    response_language: str | None = None  # What language to respond in
    response_style: str | None = None  # formal, casual, educational, etc.
    response_length: str | None = None  # brief, detailed, comprehensive

    # Tool hints
    needs_translation_model: bool = False
    needs_tts: bool = False
    needs_code_execution: bool = False
    needs_calculation: bool = False
    needs_web_search: bool = False

    # Raw reasoning from LLM
    reasoning: str = ""

    # Additional context extracted
    extra: dict[str, Any] = field(default_factory=dict)


class IntelligentIntentClassifier:
    """
    Uses the LLM itself to understand user intent dynamically.

    No hardcoded patterns - the LLM analyzes the prompt and determines:
    1. What the user wants
    2. What tools/capabilities are needed
    3. How to best respond
    """

    # Intent classification prompt - teaches the LLM to analyze intent
    INTENT_PROMPT = """You are an intent analyzer for an AI assistant called Shiksha Setu.

Analyze the user's message and determine their intent. Respond ONLY with a JSON object.

User message: "{message}"

Analyze and respond with this exact JSON structure:
{{
  "intent": "<one of: question, translation, simplification, audio, quiz, code, calculation, conversation, creative, summarization>",
  "confidence": <0.0 to 1.0>,
  "target_language": "<ISO code like 'hi', 'te', 'ta', 'en' or null if not applicable>",
  "source_language": "<detected source language ISO code or null>",
  "response_language": "<language code the user wants response in, or null for same as their message>",
  "response_style": "<formal/casual/educational/simple or null>",
  "content_to_process": "<the actual content they want processed, or null>",
  "needs_translation_model": <true if explicit translation between languages is needed>,
  "needs_tts": <true if they want audio/speech output>,
  "needs_code_execution": <true if code needs to be run>,
  "needs_calculation": <true if math calculation is needed>,
  "reasoning": "<brief explanation of your analysis>"
}}

Examples:
- "explain photosynthesis" → intent: question, response_language: null (respond in same language)
- "explain photosynthesis in hindi" → intent: question, response_language: "hi"
- "translate this to tamil: hello world" → intent: translation, target_language: "ta", needs_translation_model: true
- "make this simpler" → intent: simplification
- "read this aloud" → intent: audio, needs_tts: true
- "what is 25 * 4 + 10?" → intent: calculation, needs_calculation: true
- "hello" → intent: conversation

IMPORTANT: Respond with ONLY the JSON object, no other text."""

    def __init__(self, llm_client=None):
        """
        Initialize with optional LLM client.
        If not provided, uses fast heuristics with LLM fallback.
        """
        self._llm_client = llm_client
        self._cache: dict[str, IntentAnalysis] = {}

    def set_llm_client(self, client):
        """Set the LLM client for intelligent classification."""
        self._llm_client = client

    async def analyze(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
        use_llm: bool = True,
    ) -> IntentAnalysis:
        """
        Analyze user message to determine intent.

        Args:
            message: The user's message
            conversation_history: Previous messages for context
            use_llm: Whether to use LLM for classification (vs fast heuristics)

        Returns:
            IntentAnalysis with determined intent and parameters
        """
        # Check cache for identical messages
        cache_key = message.strip().lower()[:200]
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try LLM-based classification if available and enabled
        if use_llm and self._llm_client:
            try:
                result = await self._llm_classify(message, conversation_history)
                self._cache[cache_key] = result
                return result
            except Exception as e:
                logger.warning(
                    f"LLM classification failed, using smart heuristics: {e}"
                )

        # Smart heuristic fallback (still dynamic, not pattern-based)
        result = self._smart_heuristic_classify(message)
        self._cache[cache_key] = result
        return result

    async def _llm_classify(
        self, message: str, history: list[dict[str, str]] | None = None
    ) -> IntentAnalysis:
        """Use LLM to classify intent."""
        prompt = self.INTENT_PROMPT.format(message=message[:500])

        # Generate with low temperature for consistent classification
        response = await self._llm_client.generate(
            prompt=prompt, max_tokens=300, temperature=0.1
        )

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = re.sub(r"^```\w*\n?", "", json_str)
                json_str = re.sub(r"\n?```$", "", json_str)

            data = json.loads(json_str)

            return IntentAnalysis(
                primary_intent=Intent(data.get("intent", "question")),
                confidence=float(data.get("confidence", 0.8)),
                target_language=data.get("target_language"),
                source_language=data.get("source_language"),
                content_to_process=data.get("content_to_process"),
                response_language=data.get("response_language"),
                response_style=data.get("response_style"),
                response_length=data.get("response_length"),
                needs_translation_model=data.get("needs_translation_model", False),
                needs_tts=data.get("needs_tts", False),
                needs_code_execution=data.get("needs_code_execution", False),
                needs_calculation=data.get("needs_calculation", False),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM intent response: {e}")
            # Fallback to heuristics
            return self._smart_heuristic_classify(message)

    def _smart_heuristic_classify(self, message: str) -> IntentAnalysis:
        """
        Smart heuristic classification - lightweight but still intelligent.
        Uses linguistic patterns rather than exact keyword matching.
        """
        msg_lower = message.lower().strip()
        words = set(msg_lower.split())

        # Detect response language preference (not translation)
        response_lang = self._detect_response_language(msg_lower)

        # Detect if this is a translation request (explicit language conversion)
        is_translation, target_lang, source_lang, content = self._detect_translation(
            message
        )
        if is_translation:
            return IntentAnalysis(
                primary_intent=Intent.TRANSLATION,
                confidence=0.9,
                target_language=target_lang,
                source_language=source_lang,
                content_to_process=content,
                needs_translation_model=True,
                reasoning="Explicit translation request detected",
            )

        # Detect audio/TTS request
        if self._is_audio_request(msg_lower):
            return IntentAnalysis(
                primary_intent=Intent.AUDIO,
                confidence=0.85,
                needs_tts=True,
                content_to_process=self._extract_content(message),
                reasoning="Audio/speech request detected",
            )

        # Detect calculation request
        if self._is_calculation(msg_lower):
            return IntentAnalysis(
                primary_intent=Intent.CALCULATION,
                confidence=0.9,
                needs_calculation=True,
                content_to_process=message,
                reasoning="Mathematical expression detected",
            )

        # Detect code-related request
        if self._is_code_request(msg_lower, words):
            return IntentAnalysis(
                primary_intent=Intent.CODE,
                confidence=0.8,
                needs_code_execution="run" in words or "execute" in words,
                reasoning="Code-related request detected",
            )

        # Detect simplification request
        if self._is_simplification(msg_lower):
            return IntentAnalysis(
                primary_intent=Intent.SIMPLIFICATION,
                confidence=0.85,
                content_to_process=self._extract_content(message),
                reasoning="Simplification request detected",
            )

        # Detect quiz request
        if self._is_quiz_request(msg_lower, words):
            return IntentAnalysis(
                primary_intent=Intent.QUIZ,
                confidence=0.85,
                reasoning="Quiz/test request detected",
            )

        # Detect summarization
        if self._is_summarization(msg_lower, words):
            return IntentAnalysis(
                primary_intent=Intent.SUMMARIZATION,
                confidence=0.8,
                content_to_process=self._extract_content(message),
                reasoning="Summarization request detected",
            )

        # Detect greeting/conversation
        if self._is_greeting(msg_lower, words):
            return IntentAnalysis(
                primary_intent=Intent.CONVERSATION,
                confidence=0.9,
                response_language=response_lang,
                reasoning="Greeting or casual conversation",
            )

        # Default: treat as question/explanation request
        return IntentAnalysis(
            primary_intent=Intent.QUESTION,
            confidence=0.7,
            response_language=response_lang,
            reasoning="General question or explanation request",
        )

    def _detect_response_language(self, msg: str) -> str | None:
        """Detect what language the user wants the response in."""
        # Language codes and their indicators
        lang_indicators = {
            "hi": [
                "in hindi",
                "हिंदी में",
                "hindi mein",
                "हिंदी",
                "respond in hindi",
                "answer in hindi",
            ],
            "te": ["in telugu", "తెలుగులో", "telugu lo", "respond in telugu"],
            "ta": ["in tamil", "தமிழில்", "respond in tamil"],
            "bn": ["in bengali", "বাংলায়", "bangla", "respond in bengali"],
            "mr": ["in marathi", "मराठीत", "respond in marathi"],
            "gu": ["in gujarati", "ગુજરાતીમાં", "respond in gujarati"],
            "kn": ["in kannada", "ಕನ್ನಡದಲ್ಲಿ", "respond in kannada"],
            "ml": ["in malayalam", "മലയാളത്തിൽ", "respond in malayalam"],
            "pa": ["in punjabi", "ਪੰਜਾਬੀ ਵਿੱਚ", "respond in punjabi"],
            "or": ["in odia", "ଓଡ଼ିଆରେ", "respond in odia"],
            "en": ["in english", "respond in english", "answer in english"],
        }

        for code, indicators in lang_indicators.items():
            if any(ind in msg for ind in indicators):
                return code

        return None

    def _detect_translation(
        self, message: str
    ) -> tuple[bool, str | None, str | None, str | None]:
        """
        Detect if this is a translation request.
        Returns: (is_translation, target_lang, source_lang, content)
        """
        msg_lower = message.lower()

        # Must have explicit translation keywords
        translation_words = ["translate", "convert", "change to", "बदलो", "अनुवाद"]
        if not any(word in msg_lower for word in translation_words):
            return (False, None, None, None)

        # Language detection
        lang_map = {
            "hindi": "hi",
            "हिंदी": "hi",
            "telugu": "te",
            "తెలుగు": "te",
            "tamil": "ta",
            "தமிழ்": "ta",
            "bengali": "bn",
            "বাংলা": "bn",
            "marathi": "mr",
            "मराठी": "mr",
            "gujarati": "gu",
            "ગુજરાતી": "gu",
            "kannada": "kn",
            "ಕನ್ನಡ": "kn",
            "malayalam": "ml",
            "മലയാളം": "ml",
            "punjabi": "pa",
            "ਪੰਜਾਬੀ": "pa",
            "odia": "or",
            "ଓଡ଼ିଆ": "or",
            "english": "en",
        }

        target_lang = None
        for name, code in lang_map.items():
            if (
                f"to {name}" in msg_lower
                or f"into {name}" in msg_lower
                or f"{name} में" in msg_lower
            ):
                target_lang = code
                break

        if not target_lang:
            return (False, None, None, None)

        # Extract content to translate
        content = self._extract_content(message)

        return (True, target_lang, None, content)

    def _is_audio_request(self, msg: str) -> bool:
        """Check if user wants audio output."""
        audio_patterns = [
            "read aloud",
            "read out",
            "speak",
            "pronounce",
            "audio",
            "text to speech",
            "tts",
            "listen",
            "say it",
            "voice",
            "पढ़ो",
            "बोलो",
            "सुनाओ",
        ]
        return any(p in msg for p in audio_patterns)

    def _is_calculation(self, msg: str) -> bool:
        """Check if message contains a calculation.

        Recognizes:
        - Simple arithmetic: '2+2', '10*5', '100/4'
        - Expressions with words: 'what is 2+2', 'calculate 5*3'
        - Word-based math: '2 plus 2', '10 times 5'
        """
        # Direct math expression pattern (highest priority)
        # Matches: "2+2", "10 * 5", "100/4-2", etc.
        direct_math_pattern = r"^\s*[\d\.]+\s*[\+\-\*\/\^\%]\s*[\d\.]+"
        if re.search(direct_math_pattern, msg):
            return True

        # General math expression anywhere in message
        math_pattern = r"[\d\.]+\s*[\+\-\*\/\^\%]\s*[\d\.]+"
        if re.search(math_pattern, msg):
            return True

        # Word-based arithmetic: "2 plus 2", "10 times 5"
        word_math_pattern = (
            r"[\d\.]+\s+(plus|minus|times|divided\s*by|multiplied\s*by)\s+[\d\.]+"
        )
        if re.search(word_math_pattern, msg.lower()):
            return True

        # Explicit calculation requests
        calc_words = [
            "calculate",
            "compute",
            "solve",
            "evaluate",
            "what is",
            "what's",
            "how much is",
        ]
        has_calc_word = any(w in msg.lower() for w in calc_words)
        has_numbers = bool(re.search(r"\d+", msg))

        return has_calc_word and has_numbers

    def _is_code_request(self, msg: str, words: set) -> bool:
        """Check if message is about code."""
        code_indicators = {
            "code",
            "program",
            "function",
            "debug",
            "error",
            "bug",
            "python",
            "javascript",
            "script",
        }
        return bool(words & code_indicators)

    def _is_simplification(self, msg: str) -> bool:
        """Check if user wants content simplified."""
        patterns = [
            "simplify",
            "make it simple",
            "simpler",
            "easy",
            "easier",
            "eli5",
            "explain like",
            "in simple words",
            "basic",
            "सरल",
            "आसान",
        ]
        return any(p in msg for p in patterns)

    def _is_quiz_request(self, msg: str, words: set) -> bool:
        """Check if user wants a quiz."""
        quiz_words = {"quiz", "test", "mcq", "questions", "practice", "exam"}
        return bool(words & quiz_words)

    def _is_summarization(self, msg: str, words: set) -> bool:
        """Check if user wants summarization."""
        sum_words = {"summarize", "summary", "tldr", "brief", "shorten", "condense"}
        return bool(words & sum_words)

    def _is_greeting(self, msg: str, words: set) -> bool:
        """Check if message is a greeting."""
        greetings = {
            "hi",
            "hello",
            "hey",
            "namaste",
            "नमस्ते",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
        }
        return bool(words & greetings) and len(words) < 5

    def _extract_content(self, message: str) -> str | None:
        """Extract the actual content to process from message."""
        # Try to extract content after colon
        if ":" in message:
            parts = message.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                return parts[1].strip()

        # Try to extract quoted content
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'|"([^"]+)"', message)
        if quoted:
            return quoted[0][0] or quoted[0][1] or quoted[0][2]

        return None


# Global classifier instance
_classifier: IntelligentIntentClassifier | None = None


def get_intent_classifier() -> IntelligentIntentClassifier:
    """Get or create the global intent classifier."""
    global _classifier
    if _classifier is None:
        _classifier = IntelligentIntentClassifier()
    return _classifier


async def analyze_intent(
    message: str, history: list[dict[str, str]] | None = None, use_llm: bool = True
) -> IntentAnalysis:
    """Convenience function to analyze intent."""
    classifier = get_intent_classifier()
    return await classifier.analyze(message, history, use_llm)
