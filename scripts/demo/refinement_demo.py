#!/usr/bin/env python3
"""
Demo script for the Semantic Refinement Pipeline.

This script demonstrates:
1. Single-pass simplification (old behavior)
2. Refinement-enabled simplification (new behavior with 8.2+ target)
3. Performance comparison

Usage:
    python scripts/demo/refinement_demo.py
"""

import asyncio
import sys
import time

sys.path.insert(0, ".")

from backend.services.simplifier import REFINEMENT_AVAILABLE, TextSimplifier

SAMPLE_CONTENT = """
Photosynthesis is a sophisticated biochemical process by which photoautotrophic
organisms, primarily chlorophyll-containing plants and cyanobacteria, convert
electromagnetic radiation from the sun into chemical potential energy stored
in glucose molecules (C6H12O6). This endergonic reaction occurs within
specialized organelles called chloroplasts and comprises two interconnected
stages: the light-dependent reactions occurring in the thylakoid membranes
and the light-independent reactions (Calvin cycle) taking place in the stroma.
The net equation can be represented as: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.
"""


async def demo_single_pass():
    """Demonstrate single-pass simplification."""
    print("\n" + "=" * 60)
    print("SINGLE-PASS SIMPLIFICATION (No refinement)")
    print("=" * 60)

    simplifier = TextSimplifier(enable_refinement=False)

    start = time.perf_counter()
    result = await simplifier.simplify_text(
        content=SAMPLE_CONTENT, grade_level=7, subject="Science"
    )
    elapsed = (time.perf_counter() - start) * 1000

    print(
        f"\nOriginal complexity: {result.metadata.get('original_complexity', 'N/A'):.2f}"
    )
    print(f"Final complexity: {result.complexity_score:.2f}")
    print(f"Time: {elapsed:.0f}ms")
    print(f"\nSimplified text:\n{result.text[:500]}...")

    return result


async def demo_with_refinement():
    """Demonstrate refinement-enabled simplification."""
    print("\n" + "=" * 60)
    print("REFINEMENT-ENABLED SIMPLIFICATION (Target: 9.0+)")
    print("=" * 60)

    if not REFINEMENT_AVAILABLE:
        print("⚠️  Refinement pipeline not available")
        return None

    simplifier = TextSimplifier(
        enable_refinement=True,
        target_semantic_score=9.0,  # M4-optimized target
    )

    start = time.perf_counter()
    result = await simplifier.simplify_text(
        content=SAMPLE_CONTENT, grade_level=7, subject="Science"
    )
    elapsed = (time.perf_counter() - start) * 1000

    print(
        f"\nOriginal complexity: {result.metadata.get('original_complexity', 'N/A'):.2f}"
    )
    print(f"Final complexity: {result.complexity_score:.2f}")

    if result.semantic_score is not None:
        print(f"\n✅ Semantic Score: {result.semantic_score:.2f}")
        print(f"   Iterations used: {result.refinement_iterations}")
        print(f"   Target reached: {result.metadata.get('target_reached', 'Unknown')}")

        if result.dimension_scores:
            print("\n   Dimension Scores:")
            for dim, score in result.dimension_scores.items():
                emoji = "✓" if score >= 8.0 else "✗"
                print(f"     {emoji} {dim}: {score:.2f}")
    else:
        print("\n⚠️  Refinement was skipped or failed")
        if "refinement_error" in result.metadata:
            print(f"   Error: {result.metadata['refinement_error']}")

    print(f"\nTime: {elapsed:.0f}ms")
    print(f"\nSimplified text:\n{result.text[:500]}...")

    return result


TARGET_SCORE = 9.0  # M4-optimized target (achievable with RAG + validation)


async def main():
    """Run the demo."""
    print("\n" + "=" * 60)
    print("SEMANTIC ACCURACY REFINEMENT PIPELINE DEMO")
    print("=" * 60)
    print(f"\nRefinement Available: {REFINEMENT_AVAILABLE}")
    print(f"Target Semantic Score: {TARGET_SCORE}")

    # Single-pass demo
    result1 = await demo_single_pass()

    # Refinement demo
    result2 = await demo_with_refinement()

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if result1 and result2:
        print(f"\nSingle-pass complexity: {result1.complexity_score:.2f}")
        print(f"Refined complexity: {result2.complexity_score:.2f}")

        if result2.semantic_score is not None:
            print(f"\nRefined semantic score: {result2.semantic_score:.2f}")
            if result2.semantic_score >= TARGET_SCORE:
                print("✅ Target semantic accuracy achieved!")
            else:
                print(
                    f"⚠️  Score below target (need {TARGET_SCORE - result2.semantic_score:.2f} more)"
                )

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
