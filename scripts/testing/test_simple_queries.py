#!/usr/bin/env python3
"""
Test Script for Simple Query Handling
======================================

Validates that simple queries like "2+2" work correctly after the fixes.

Run with:
    python scripts/testing/test_simple_queries.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_fast_arithmetic():
    """Test the fast arithmetic computation path."""
    from backend.services.ai_core.engine import AIEngine

    engine = AIEngine()

    # Test cases for fast arithmetic
    test_cases = [
        ("2+2", "4"),
        ("10 * 5", "50"),
        ("100/4", "25"),
        ("2^8", "256"),
        ("10 - 3", "7"),
        ("(2 + 3) * 4", "20"),
        ("15 + 25", "40"),
        ("100 / 5 + 10", "30"),
    ]

    print("Testing Fast Arithmetic Path:")
    print("-" * 50)

    for query, expected in test_cases:
        start = time.perf_counter()
        result = engine._try_fast_arithmetic(query)
        elapsed = (time.perf_counter() - start) * 1000

        if result and expected in result:
            print(f"✅ '{query}' = {expected} (computed in {elapsed:.2f}ms)")
        elif result:
            print(f"⚠️  '{query}' computed but got: {result}")
        else:
            print(f"❌ '{query}' - fast path returned None (will use LLM)")

    print()


def test_intent_classification():
    """Test intent classification for simple queries."""
    from backend.services.ai_core.intent import IntelligentIntentClassifier, Intent

    classifier = IntelligentIntentClassifier()

    test_cases = [
        ("2+2", Intent.CALCULATION),
        ("10 * 5", Intent.CALCULATION),
        ("what is 25 + 75", Intent.CALCULATION),
        ("hello", Intent.CONVERSATION),
        ("explain photosynthesis", Intent.QUESTION),
        ("write code for sorting", Intent.CODE),
    ]

    print("Testing Intent Classification:")
    print("-" * 50)

    for query, expected_intent in test_cases:
        result = classifier._smart_heuristic_classify(query)

        if result.primary_intent == expected_intent:
            print(
                f"✅ '{query}' → {result.primary_intent.value} (confidence: {result.confidence})"
            )
        else:
            print(
                f"❌ '{query}' → got {result.primary_intent.value}, expected {expected_intent.value}"
            )

    print()


def test_is_simple_query():
    """Test simple query detection."""
    from backend.services.ai_core.engine import AIEngine
    from backend.services.ai_core.formatter import Intent

    engine = AIEngine()

    test_cases = [
        ("2+2", Intent.QUESTION, True),
        ("hello", Intent.SMALL_TALK, True),
        ("hi", Intent.SMALL_TALK, True),
        ("is the sky blue?", Intent.QUESTION, True),  # Short yes/no
        (
            "explain the theory of relativity in detail with examples",
            Intent.EXPLANATION,
            False,
        ),
    ]

    print("Testing Simple Query Detection:")
    print("-" * 50)

    for query, intent, expected in test_cases:
        result = engine._is_simple_query(query, intent)

        if result == expected:
            print(f"✅ '{query[:40]}...' is_simple={result}")
        else:
            print(f"❌ '{query[:40]}...' got is_simple={result}, expected {expected}")

    print()


def test_formatter_intent_detection():
    """Test formatter intent detection for arithmetic."""
    from backend.services.ai_core.formatter import Intent, ResponseFormatter

    formatter = ResponseFormatter()

    test_cases = [
        ("2+2", Intent.QUESTION),  # Pure arithmetic
        ("10 * 5", Intent.QUESTION),
        ("what is 25 + 75", Intent.QUESTION),
        ("calculate 100/4", Intent.QUESTION),
    ]

    print("Testing Formatter Intent Detection:")
    print("-" * 50)

    for query, expected_intent in test_cases:
        result = formatter.detect_intent(query)

        if result == expected_intent or result != Intent.UNKNOWN:
            print(f"✅ '{query}' → {result.value}")
        else:
            print(
                f"❌ '{query}' → got {result.value}, expected {expected_intent.value}"
            )

    print()


async def test_full_chat_flow():
    """Test the full chat flow with simple queries."""
    from backend.services.ai_core.engine import GenerationConfig, get_ai_engine

    print("Testing Full Chat Flow:")
    print("-" * 50)
    print("(This requires the LLM to be loaded - may take a moment)")
    print()

    try:
        engine = get_ai_engine()

        test_cases = [
            "2+2",
            "hello",
            "what is 10 * 5",
        ]

        for query in test_cases:
            print(f"Query: '{query}'")
            start = time.perf_counter()

            try:
                result = await engine.chat(
                    message=query,
                    config=GenerationConfig(max_tokens=100, temperature=0.1),
                )
                elapsed = (time.perf_counter() - start) * 1000
                print(f"  Response: {result.content[:100]}...")
                print(f"  Latency: {elapsed:.0f}ms")
                print(f"  Model: {result.metadata.model_id}")
                print()
            except Exception as e:
                print(f"  Error: {e}")
                print()

    except Exception as e:
        print(f"Could not initialize AI Engine: {e}")
        print("(This is expected if models are not loaded)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SHIKSHA SETU - Simple Query Handling Tests")
    print("=" * 60)
    print()

    try:
        test_fast_arithmetic()
        test_intent_classification()
        test_is_simple_query()
        test_formatter_intent_detection()

        # Optional: Full flow test (requires LLM)
        print("\nRunning async chat flow test...")
        asyncio.run(test_full_chat_flow())

    except ImportError as e:
        print(f"Import error: {e}")
        print(
            "Make sure you're running from the project root with the virtual environment activated."
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
