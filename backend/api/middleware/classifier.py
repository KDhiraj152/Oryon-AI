"""
Request Classifier — Phase 1
=============================

Detects task complexity, routes to the appropriate model tier,
and avoids unnecessary large-model calls.

Design:
- Two-stage classification: fast heuristic → optional LLM refinement
- Complexity scoring produces a continuous 0.0-1.0 signal
- Routing map binds (intent, complexity) → model tier
- Embedding cache deduplication: identical/near-identical prompts reuse prior decisions
- All state is immutable after construction — thread-safe by design
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum, StrEnum
from typing import Any

logger = logging.getLogger(__name__)

# ── Enums ────────────────────────────────────────────────────────────────────

class ComplexityLevel(IntEnum):
    """Discrete complexity buckets with numeric ordering."""

    TRIVIAL = 0      # Greetings, yes/no, single-word answers
    SIMPLE = 1       # Direct definitions, lookups
    MODERATE = 2     # Explanations, comparisons
    COMPLEX = 3      # Multi-step reasoning, code generation
    DEEP = 4         # Research-grade, comprehensive analysis

class TaskIntent(StrEnum):
    """Canonical task intents recognised by the middleware."""

    CHAT = "chat"
    QUESTION = "question"
    TRANSLATION = "translation"
    SIMPLIFICATION = "simplification"
    SUMMARIZATION = "summarization"
    CODE = "code"
    CALCULATION = "calculation"
    AUDIO = "audio"
    QUIZ = "quiz"
    CREATIVE = "creative"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    STT = "stt"
    OCR = "ocr"
    CONVERSATION = "conversation"

class ModelTarget(StrEnum):
    """Which model tier to route to."""

    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    STRONG = "strong"
    SPECIALIZED = "specialized"
    SKIP_LLM = "skip_llm"          # Task handled without LLM (calc, lookup)

# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ClassifiedRequest:
    """Immutable classification result — the single contract between
    the classifier and everything downstream."""

    request_id: str
    intent: TaskIntent
    complexity: ComplexityLevel
    complexity_score: float          # 0.0-1.0 continuous
    model_target: ModelTarget
    estimated_tokens: int
    needs_rag: bool
    needs_translation: bool
    needs_tts: bool
    parallel_safe: bool              # Can stages run in parallel?
    source_language: str | None = None
    target_language: str | None = None
    confidence: float = 0.0          # Classifier confidence
    raw_extras: dict[str, Any] = field(default_factory=dict)
    classified_at: float = field(default_factory=time.time)

# ── Routing Table ────────────────────────────────────────────────────────────

# (intent, complexity_threshold) → model target
# Lower entries win — first match by intent, then by complexity ≥ threshold
_ROUTING_TABLE: list[tuple[TaskIntent | None, ComplexityLevel, ModelTarget]] = [
    # Specialised tasks always go to dedicated models regardless of complexity
    (TaskIntent.TRANSLATION, ComplexityLevel.TRIVIAL, ModelTarget.SPECIALIZED),
    (TaskIntent.EMBEDDING, ComplexityLevel.TRIVIAL, ModelTarget.SPECIALIZED),
    (TaskIntent.RERANKING, ComplexityLevel.TRIVIAL, ModelTarget.SPECIALIZED),
    (TaskIntent.STT, ComplexityLevel.TRIVIAL, ModelTarget.SPECIALIZED),
    (TaskIntent.OCR, ComplexityLevel.TRIVIAL, ModelTarget.SPECIALIZED),
    (TaskIntent.AUDIO, ComplexityLevel.TRIVIAL, ModelTarget.SPECIALIZED),

    # Calculations handled without LLM
    (TaskIntent.CALCULATION, ComplexityLevel.TRIVIAL, ModelTarget.SKIP_LLM),

    # Conversations — scale with complexity
    (TaskIntent.CONVERSATION, ComplexityLevel.TRIVIAL, ModelTarget.LIGHTWEIGHT),
    (TaskIntent.CONVERSATION, ComplexityLevel.SIMPLE, ModelTarget.LIGHTWEIGHT),
    (TaskIntent.CONVERSATION, ComplexityLevel.MODERATE, ModelTarget.STANDARD),
    (TaskIntent.CONVERSATION, ComplexityLevel.COMPLEX, ModelTarget.STRONG),
    (TaskIntent.CONVERSATION, ComplexityLevel.DEEP, ModelTarget.STRONG),

    # Questions — scale with complexity
    (TaskIntent.QUESTION, ComplexityLevel.TRIVIAL, ModelTarget.LIGHTWEIGHT),
    (TaskIntent.QUESTION, ComplexityLevel.SIMPLE, ModelTarget.LIGHTWEIGHT),
    (TaskIntent.QUESTION, ComplexityLevel.MODERATE, ModelTarget.STANDARD),
    (TaskIntent.QUESTION, ComplexityLevel.COMPLEX, ModelTarget.STRONG),
    (TaskIntent.QUESTION, ComplexityLevel.DEEP, ModelTarget.STRONG),

    # Code — always standard+
    (TaskIntent.CODE, ComplexityLevel.TRIVIAL, ModelTarget.STANDARD),
    (TaskIntent.CODE, ComplexityLevel.MODERATE, ModelTarget.STANDARD),
    (TaskIntent.CODE, ComplexityLevel.COMPLEX, ModelTarget.STRONG),
    (TaskIntent.CODE, ComplexityLevel.DEEP, ModelTarget.STRONG),

    # Creative writing — needs generative capability
    (TaskIntent.CREATIVE, ComplexityLevel.TRIVIAL, ModelTarget.STANDARD),
    (TaskIntent.CREATIVE, ComplexityLevel.COMPLEX, ModelTarget.STRONG),

    # Catch-all by complexity alone (intent=None means "any")
    (None, ComplexityLevel.TRIVIAL, ModelTarget.LIGHTWEIGHT),
    (None, ComplexityLevel.SIMPLE, ModelTarget.LIGHTWEIGHT),
    (None, ComplexityLevel.MODERATE, ModelTarget.STANDARD),
    (None, ComplexityLevel.COMPLEX, ModelTarget.STRONG),
    (None, ComplexityLevel.DEEP, ModelTarget.STRONG),
]

# ── Token Estimation ─────────────────────────────────────────────────────────

_TOKEN_ESTIMATES: dict[tuple[TaskIntent, ComplexityLevel], int] = {
    (TaskIntent.CONVERSATION, ComplexityLevel.TRIVIAL): 256,
    (TaskIntent.CONVERSATION, ComplexityLevel.SIMPLE): 512,
    (TaskIntent.QUESTION, ComplexityLevel.TRIVIAL): 512,
    (TaskIntent.QUESTION, ComplexityLevel.SIMPLE): 1024,
    (TaskIntent.QUESTION, ComplexityLevel.MODERATE): 2048,
    (TaskIntent.QUESTION, ComplexityLevel.COMPLEX): 4096,
    (TaskIntent.QUESTION, ComplexityLevel.DEEP): 6144,
    (TaskIntent.CODE, ComplexityLevel.TRIVIAL): 2048,
    (TaskIntent.CODE, ComplexityLevel.SIMPLE): 2048,
    (TaskIntent.CODE, ComplexityLevel.MODERATE): 4096,
    (TaskIntent.CODE, ComplexityLevel.COMPLEX): 6144,
    (TaskIntent.CODE, ComplexityLevel.DEEP): 8192,
    (TaskIntent.CREATIVE, ComplexityLevel.TRIVIAL): 1024,
    (TaskIntent.CREATIVE, ComplexityLevel.MODERATE): 4096,
    (TaskIntent.CREATIVE, ComplexityLevel.COMPLEX): 6144,
    (TaskIntent.SUMMARIZATION, ComplexityLevel.SIMPLE): 1024,
    (TaskIntent.SUMMARIZATION, ComplexityLevel.MODERATE): 2048,
    (TaskIntent.SIMPLIFICATION, ComplexityLevel.SIMPLE): 1024,
    (TaskIntent.SIMPLIFICATION, ComplexityLevel.MODERATE): 2048,
    (TaskIntent.TRANSLATION, ComplexityLevel.SIMPLE): 512,
    (TaskIntent.TRANSLATION, ComplexityLevel.MODERATE): 1024,
}

_DEFAULT_TOKENS = 2048

# ── Complexity Scoring ───────────────────────────────────────────────────────

# keyword sets for scoring (weighted)
_HIGH_COMPLEXITY_WORDS = frozenset({
    "analyze", "compare", "implement", "design", "architect", "optimise",
    "optimize", "debug", "refactor", "prove", "derive", "comprehensive",
    "step by step", "in detail", "explain in detail", "research",
    "detailed analysis", "advanced",
})

_LOW_COMPLEXITY_WORDS = frozenset({
    "what is", "define", "who is", "when", "translate", "hello",
    "hi", "thanks", "yes", "no", "ok",
})

_CODE_WORDS = frozenset({
    "code", "program", "function", "debug", "error", "bug",
    "python", "javascript", "script", "implement", "algorithm", "class",
})

_MATH_PATTERN_CHARS = frozenset("+-*/^%=")

def _score_complexity(text: str) -> tuple[ComplexityLevel, float]:
    """Score text complexity on a 0.0-1.0 scale and discretize.

    Pure function — no side effects — safe for concurrent use.
    """
    lower = text.lower()
    words_set = set(lower.split())
    length = len(text)

    score = 0.0

    # High-complexity keyword matches
    high_hits = sum(1 for kw in _HIGH_COMPLEXITY_WORDS if kw in lower)
    score += min(high_hits * 0.15, 0.45)

    # Low-complexity keyword matches (negative weight)
    low_hits = sum(1 for kw in _LOW_COMPLEXITY_WORDS if kw in lower)
    score -= min(low_hits * 0.10, 0.30)

    # Length contribution (longer → probably more complex)
    if length > 800:
        score += 0.25
    elif length > 400:
        score += 0.15
    elif length > 150:
        score += 0.05
    elif length < 30:
        score -= 0.10

    # Multi-sentence penalty/boost
    sentence_count = lower.count(".") + lower.count("?") + lower.count("!")
    if sentence_count > 4:
        score += 0.10
    elif sentence_count <= 1 and length < 60:
        score -= 0.05

    # Code indicators
    code_hits = len(words_set & _CODE_WORDS)
    if code_hits >= 2:
        score += 0.15

    # Clamp
    score = max(0.0, min(1.0, score))

    # Discretize
    if score < 0.15:
        level = ComplexityLevel.TRIVIAL
    elif score < 0.30:
        level = ComplexityLevel.SIMPLE
    elif score < 0.55:
        level = ComplexityLevel.MODERATE
    elif score < 0.75:
        level = ComplexityLevel.COMPLEX
    else:
        level = ComplexityLevel.DEEP

    return level, round(score, 4)

def _detect_intent(text: str) -> tuple[TaskIntent, float]:
    """Fast heuristic intent detection.

    Returns (intent, confidence).
    """
    lower = text.lower().strip()
    words = set(lower.split())

    # Translation
    translate_kw = {"translate", "convert", "बदलो", "अनुवाद"}
    lang_targets = {
        "hindi", "telugu", "tamil", "bengali", "marathi",
        "gujarati", "kannada", "malayalam", "punjabi", "odia", "english",
    }
    if words & translate_kw and any(f"to {lang}" in lower or f"into {lang}" in lower for lang in lang_targets):
        return TaskIntent.TRANSLATION, 0.92

    # Audio/TTS
    audio_kw = {"read aloud", "read this aloud", "speak", "audio", "tts", "voice", "pronounce", "पढ़ो", "बोलो"}
    if any(kw in lower for kw in audio_kw):
        return TaskIntent.AUDIO, 0.88

    # Calculation — presence of math operators with digits
    if any(ch in text for ch in _MATH_PATTERN_CHARS):
        import re
        if re.search(r"\d+\s*[\+\-\*\/\^\%]\s*\d+", text):
            return TaskIntent.CALCULATION, 0.93

    calc_words = {"calculate", "compute", "solve", "evaluate"}
    if words & calc_words and any(c.isdigit() for c in text):
        return TaskIntent.CALCULATION, 0.85

    # Code
    if len(words & _CODE_WORDS) >= 2:
        return TaskIntent.CODE, 0.82

    # Simplification
    simple_kw = {"simplify", "simpler", "easier", "eli5", "in simple words", "सरल", "आसान"}
    if any(kw in lower for kw in simple_kw):
        return TaskIntent.SIMPLIFICATION, 0.87

    # Quiz
    quiz_kw = {"quiz", "test", "mcq", "exam", "practice"}
    if words & quiz_kw:
        return TaskIntent.QUIZ, 0.86

    # Summarization
    sum_kw = {"summarize", "summary", "tldr", "condense", "shorten"}
    if words & sum_kw:
        return TaskIntent.SUMMARIZATION, 0.84

    # Greeting/conversation
    greetings = {"hi", "hello", "hey", "namaste", "नमस्ते", "good morning", "thanks", "thank you"}
    if (words & greetings) and len(words) < 6:
        return TaskIntent.CONVERSATION, 0.91

    # Creative writing
    creative_kw = {"write a story", "poem", "creative", "fiction", "imagine"}
    if any(kw in lower for kw in creative_kw):
        return TaskIntent.CREATIVE, 0.80

    # Default → question
    return TaskIntent.QUESTION, 0.65

def _resolve_model_target(intent: TaskIntent, complexity: ComplexityLevel) -> ModelTarget:
    """Walk the routing table to find the best model target.

    Picks the rule with the **highest** complexity threshold that is
    still <= the request's complexity.  Intent-specific rules take
    priority; catch-all (intent=None) rules are used only when no
    intent-specific rules exist for the given intent.
    """
    best: ModelTarget | None = None
    best_threshold: int = -1

    # Pass 1: intent-specific rules
    for rule_intent, threshold, target in _ROUTING_TABLE:
        if rule_intent != intent:
            continue
        if threshold <= complexity and threshold > best_threshold:
            best = target
            best_threshold = threshold

    if best is not None:
        return best

    # Pass 2: catch-all rules (intent=None)
    best_threshold = -1
    for rule_intent, threshold, target in _ROUTING_TABLE:
        if rule_intent is not None:
            continue
        if threshold <= complexity and threshold > best_threshold:
            best = target
            best_threshold = threshold

    return best if best is not None else ModelTarget.STANDARD

def _detect_languages(text: str) -> tuple[str | None, str | None]:
    """Detect source/target language from text."""
    lower = text.lower()
    lang_map = {
        "hindi": "hi", "telugu": "te", "tamil": "ta", "bengali": "bn",
        "marathi": "mr", "gujarati": "gu", "kannada": "kn",
        "malayalam": "ml", "punjabi": "pa", "odia": "or", "english": "en",
    }
    target = None
    for name, code in lang_map.items():
        if f"to {name}" in lower or f"into {name}" in lower or f"{name} में" in lower:
            target = code
            break

    source = None
    for name, code in lang_map.items():
        if f"from {name}" in lower:
            source = code
            break

    return source, target

# ── Classifier ───────────────────────────────────────────────────────────────

class RequestClassifier:
    """
    Stateless request classifier.

    Two-stage:
      1. Fast heuristic (< 1 ms) — always runs
      2. Optional LLM refinement — only for ambiguous cases (confidence < threshold)

    Thread-safe: all instance state is read-only after __init__.
    Uses a bounded dedup cache for identical prompts.
    """

    def __init__(
        self,
        llm_refinement_threshold: float = 0.60,
        cache_max_size: int = 2048,
    ):
        self._llm_threshold = llm_refinement_threshold
        self._cache_max = cache_max_size
        self._cache: dict[str, ClassifiedRequest] = {}
        self._stats = _ClassifierStats()

    # ── Public API ───────────────────────────────────────────────────

    def classify(self, text: str, request_id: str = "") -> ClassifiedRequest:
        """Synchronous fast-path classification (heuristic only)."""
        cache_key = self._cache_key(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._stats.cache_hits += 1
            return cached

        self._stats.cache_misses += 1
        result = self._heuristic_classify(text, request_id)
        self._put_cache(cache_key, result)
        return result

    async def classify_async(
        self,
        text: str,
        request_id: str = "",
        llm_client: Any = None,
    ) -> ClassifiedRequest:
        """Async classification with optional LLM refinement for low-confidence."""
        result = self.classify(text, request_id)
        if result.confidence >= self._llm_threshold or llm_client is None:
            return result

        # LLM refinement path (rare)
        try:
            refined = await self._llm_refine(text, result, llm_client)
            self._put_cache(self._cache_key(text), refined)
            self._stats.llm_refinements += 1
            return refined
        except Exception as e:  # middleware handler
            logger.warning("LLM refinement failed, using heuristic: %s", e)
            return result

    def get_stats(self) -> dict[str, Any]:
        return {
            "cache_hits": self._stats.cache_hits,
            "cache_misses": self._stats.cache_misses,
            "llm_refinements": self._stats.llm_refinements,
            "cache_size": len(self._cache),
        }

    # ── Internals ────────────────────────────────────────────────────

    def _heuristic_classify(self, text: str, request_id: str) -> ClassifiedRequest:
        """Pure-function heuristic classifier."""
        intent, intent_confidence = _detect_intent(text)
        complexity, complexity_score = _score_complexity(text)
        model_target = _resolve_model_target(intent, complexity)
        source_lang, target_lang = _detect_languages(text)

        estimated_tokens = _TOKEN_ESTIMATES.get(
            (intent, complexity), _DEFAULT_TOKENS
        )

        needs_rag = intent in {
            TaskIntent.QUESTION, TaskIntent.CHAT,
            TaskIntent.SUMMARIZATION, TaskIntent.SIMPLIFICATION,
        } and complexity >= ComplexityLevel.MODERATE

        needs_translation = intent == TaskIntent.TRANSLATION or target_lang is not None
        needs_tts = intent == TaskIntent.AUDIO

        # Parallel-safe if the task doesn't feed into a sequential pipeline
        parallel_safe = intent not in {
            TaskIntent.SIMPLIFICATION,  # simplify → translate → TTS is sequential
        }

        return ClassifiedRequest(
            request_id=request_id,
            intent=intent,
            complexity=complexity,
            complexity_score=complexity_score,
            model_target=model_target,
            estimated_tokens=estimated_tokens,
            needs_rag=needs_rag,
            needs_translation=needs_translation,
            needs_tts=needs_tts,
            parallel_safe=parallel_safe,
            source_language=source_lang,
            target_language=target_lang,
            confidence=intent_confidence,
        )

    async def _llm_refine(
        self,
        text: str,
        heuristic: ClassifiedRequest,
        llm_client: Any,
    ) -> ClassifiedRequest:
        """Use LLM to refine a low-confidence classification."""
        prompt = (
            f"Classify the user message intent. Respond with ONLY one word from: "
            f"question, translation, simplification, summarization, code, "
            f"calculation, audio, quiz, creative, conversation.\n\n"
            f"Message: {text[:400]}\n\nIntent:"
        )
        raw = await llm_client.generate(prompt=prompt, max_tokens=10, temperature=0.0)
        raw_lower = raw.strip().lower().split()[0] if raw else ""

        try:
            refined_intent = TaskIntent(raw_lower)
        except ValueError:
            refined_intent = heuristic.intent

        complexity = heuristic.complexity
        model_target = _resolve_model_target(refined_intent, complexity)

        return ClassifiedRequest(
            request_id=heuristic.request_id,
            intent=refined_intent,
            complexity=complexity,
            complexity_score=heuristic.complexity_score,
            model_target=model_target,
            estimated_tokens=_TOKEN_ESTIMATES.get(
                (refined_intent, complexity), _DEFAULT_TOKENS
            ),
            needs_rag=heuristic.needs_rag,
            needs_translation=refined_intent == TaskIntent.TRANSLATION,
            needs_tts=refined_intent == TaskIntent.AUDIO,
            parallel_safe=heuristic.parallel_safe,
            source_language=heuristic.source_language,
            target_language=heuristic.target_language,
            confidence=0.85,  # LLM refinement boosts confidence
        )

    def _cache_key(self, text: str) -> str:
        normalized = text.strip().lower()[:300]
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _put_cache(self, key: str, value: ClassifiedRequest) -> None:
        if len(self._cache) >= self._cache_max:
            # Evict oldest 25%
            evict_count = self._cache_max // 4
            keys = list(self._cache.keys())[:evict_count]
            for k in keys:
                del self._cache[k]
        self._cache[key] = value

@dataclass
class _ClassifierStats:
    cache_hits: int = 0
    cache_misses: int = 0
    llm_refinements: int = 0
