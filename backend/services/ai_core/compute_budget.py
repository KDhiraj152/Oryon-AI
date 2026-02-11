"""
Compute Budget — Intent-Based Compute Allocation
==================================================

Classifies every query into one of four compute classes before any model
invocation, then returns an immutable budget that caps all downstream
resource consumption.

    ComputeClass.TRIVIAL   → greetings, acks, ultra-short
    ComputeClass.STANDARD  → general Q&A, simple tasks
    ComputeClass.COMPLEX   → code gen, multi-step reasoning, comparisons
    ComputeClass.CRITICAL  → safety-critical, factual-precision domains

Over-computation is a failure.  Under-computation is a failure.
The budget enforces optimal computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .formatter import Intent
from .router import ModelTier

logger = logging.getLogger(__name__)
class ComputeClass(str, Enum):
    """Four-tier compute classification."""

    TRIVIAL = "trivial"
    STANDARD = "standard"
    COMPLEX = "complex"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ComputeBudget:
    """Immutable per-request compute budget.

    Every numeric field is a *ceiling* — downstream components must not
    exceed these bounds.
    """

    compute_class: ComputeClass

    # Maximum chain-of-thought / model invocation depth
    max_reasoning_depth: int

    # Maximum number of distinct models that may be invoked
    model_count_ceiling: int

    # Target latency ceiling (ms) — advisory, not enforced by hard timeout
    max_latency_ms: float

    # 0.0 = exact answer required, 1.0 = rough approximation acceptable
    acceptable_approximation_error: float

    # Hard ceiling on generated output tokens
    max_tokens: int

    # RAG: False ⇒ skip, None ⇒ auto-detect, True ⇒ force
    use_rag: Optional[bool]

    # Whether to run the 3-pass safety pipeline
    run_safety: bool

    # Highest model tier allowed (hard ceiling, router must respect)
    model_tier_ceiling: ModelTier


# ---------------------------------------------------------------------------
# Pre-computed budget singletons (zero per-request allocation)
# ---------------------------------------------------------------------------

COMPUTE_BUDGETS: dict[ComputeClass, ComputeBudget] = {
    ComputeClass.TRIVIAL: ComputeBudget(
        compute_class=ComputeClass.TRIVIAL,
        max_reasoning_depth=1,
        model_count_ceiling=1,
        max_latency_ms=500.0,
        acceptable_approximation_error=0.5,
        max_tokens=256,
        use_rag=False,
        run_safety=True,  # Safety is never optional — even LLMs for greetings can emit PII/toxicity
        model_tier_ceiling=ModelTier.LIGHTWEIGHT,
    ),
    ComputeClass.STANDARD: ComputeBudget(
        compute_class=ComputeClass.STANDARD,
        max_reasoning_depth=2,
        model_count_ceiling=2,
        max_latency_ms=2000.0,
        acceptable_approximation_error=0.2,
        max_tokens=512,
        use_rag=None,  # auto-detect
        run_safety=True,
        model_tier_ceiling=ModelTier.STANDARD,
    ),
    ComputeClass.COMPLEX: ComputeBudget(
        compute_class=ComputeClass.COMPLEX,
        max_reasoning_depth=4,
        model_count_ceiling=3,
        max_latency_ms=5000.0,
        acceptable_approximation_error=0.05,
        max_tokens=2048,
        use_rag=None,  # auto-detect — code gen may not need RAG
        run_safety=True,
        model_tier_ceiling=ModelTier.STRONG,
    ),
    ComputeClass.CRITICAL: ComputeBudget(
        compute_class=ComputeClass.CRITICAL,
        max_reasoning_depth=6,
        model_count_ceiling=4,
        max_latency_ms=10000.0,
        acceptable_approximation_error=0.01,
        max_tokens=2048,
        use_rag=True,  # always try verified sources for critical queries
        run_safety=True,
        model_tier_ceiling=ModelTier.STRONG,
    ),
}


def get_budget(cls: ComputeClass) -> ComputeBudget:
    """Return the singleton budget for a compute class."""
    return COMPUTE_BUDGETS[cls]


# ---------------------------------------------------------------------------
# Unified classifier — deterministic, O(n) in message length, zero I/O
# ---------------------------------------------------------------------------

# --- CRITICAL-tier keyword groups (education safety-critical domains) ---
_CRITICAL_PATTERNS: list[str] = [
    # Health / medical
    "medicine", "medication", "dosage", "symptom", "disease", "diagnosis",
    "treatment", "poison", "toxic", "allergic", "allergy", "first aid",
    "emergency", "unconscious", "bleeding", "choking",
    # Chemistry / physics hazards
    "chemical equation", "exothermic", "endothermic", "concentrated acid",
    "concentrated base", "electrolysis", "radioactive", "radiation",
    "flammable", "combustion", "explosive",
    # Electrical safety
    "high voltage", "electric shock", "short circuit",
    # Precision-required educational content
    "exam answer", "correct answer", "mark scheme", "board exam",
    "entrance exam", "competitive exam",
]

_PRECISION_SIGNALS: list[str] = [
    "exactly", "precisely", "accurate", "verify", "fact check",
    "is it true", "correct that", "make sure", "how many exactly",
]

# --- COMPLEX-tier keyword groups ---
_COMPLEX_KEYWORDS: list[str] = [
    "explain in detail", "step by step", "step-by-step",
    "analyze", "analyse", "compare and contrast", "evaluate",
    "derive", "prove", "implement", "design", "architecture",
    "trade-off", "tradeoff", "pros and cons",
    "advantages and disadvantages", "algorithm", "optimize",
    "refactor", "write a program", "write code", "debug",
    "comprehensive", "in-depth", "thorough",
]

# --- TRIVIAL-tier greeting prefixes ---
_TRIVIAL_GREETINGS: list[str] = [
    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
    "ok", "okay", "good morning", "good afternoon", "good evening",
    "good night", "see you", "cool", "nice", "great", "awesome",
    "got it", "understood", "sure", "yes", "no", "yep", "nope",
    "namaste", "dhanyavad", "shukriya",
]


def _is_critical_compute(msg_lower: str) -> bool:
    """Check if message requires CRITICAL compute tier."""
    if any(p in msg_lower for p in _CRITICAL_PATTERNS):
        return True
    precision_hits = sum(1 for s in _PRECISION_SIGNALS if s in msg_lower)
    return precision_hits >= 2


def _is_trivial_compute(message: str, msg_lower: str, msg_len: int, intent: Intent) -> bool:
    """Check if message qualifies for TRIVIAL compute tier."""
    # Ultra-short non-questions
    if msg_len < 20 and "?" not in message:
        return True

    # Short greetings / acknowledgments
    if msg_len < 50 and "?" not in message and len(msg_lower.split()) <= 4:
        if any(msg_lower.startswith(g) or msg_lower == g for g in _TRIVIAL_GREETINGS):
            return True

    # Small talk intent — only if genuinely just a greeting
    if intent == Intent.SMALL_TALK:
        if msg_len < 50 and "?" not in message and len(msg_lower.split()) <= 4:
            return True

    return False


def _is_complex_compute(msg_lower: str, msg_len: int, intent: Intent) -> bool:
    """Check if message requires COMPLEX compute tier."""
    if intent in (Intent.CODE_REQUEST, Intent.COMPARISON):
        return True
    if msg_len > 300:
        return True
    if any(k in msg_lower for k in _COMPLEX_KEYWORDS):
        return True
    if msg_lower.count("?") >= 2:
        return True
    return False


def classify_compute(message: str, intent: Intent) -> ComputeClass:
    """Classify a user message into a compute class.

    This is the single point of truth for compute allocation.  It absorbs
    the signals that were previously scattered across:
      - ``_is_simple_query()``       (binary fast-path gate)
      - ``_assess_complexity()``     (router keyword matching)
      - ``detect_intent()``          (regex intent enum)

    Returns:
        The compute class that governs all downstream resource ceilings.
    """
    msg_lower = message.lower().strip()
    msg_len = len(message)

    if _is_critical_compute(msg_lower):
        return ComputeClass.CRITICAL

    if _is_trivial_compute(message, msg_lower, msg_len, intent):
        return ComputeClass.TRIVIAL

    if _is_complex_compute(msg_lower, msg_len, intent):
        return ComputeClass.COMPLEX

    return ComputeClass.STANDARD
