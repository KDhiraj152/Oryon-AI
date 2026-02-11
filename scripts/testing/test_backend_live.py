#!/usr/bin/env python3
"""
Live Backend Test Script
========================
Tests the ShikshaSetu API with real prompts and verifies responses.

Usage:
    ./bin/test-backend-live [--port 9000] [--verbose]
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

try:
    import httpx
except ImportError:
    print("Installing httpx...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

# ============================================================================
# Configuration
# ============================================================================

CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
DIM = '\033[2m'
NC = '\033[0m'

@dataclass
class TestResult:
    name: str
    passed: bool
    response_time_ms: float
    status_code: int = 0
    response_data: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class TestSuite:
    name: str
    results: List[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)


# ============================================================================
# Test Cases
# ============================================================================

class BackendTester:
    def __init__(self, base_url: str, verbose: bool = False):
        self.base_url = base_url
        self.verbose = verbose
        self.suites: List[TestSuite] = []
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=120.0)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    def log(self, msg: str, level: str = "info"):
        if level == "info":
            print(f"   {msg}")
        elif level == "success":
            print(f"   {GREEN}✓{NC} {msg}")
        elif level == "error":
            print(f"   {RED}✗{NC} {msg}")
        elif level == "warn":
            print(f"   {YELLOW}○{NC} {msg}")
        elif level == "debug" and self.verbose:
            print(f"   {DIM}{msg}{NC}")

    async def request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        timeout: float = 60.0
    ) -> TestResult:
        """Make a request and return result."""
        url = f"{self.base_url}{endpoint}"
        start = time.perf_counter()

        try:
            if method == "GET":
                response = await self.client.get(url, timeout=timeout)
            else:
                response = await self.client.post(url, json=json_data, timeout=timeout)

            elapsed = (time.perf_counter() - start) * 1000

            try:
                data = response.json()
            except:
                data = {"raw": response.text[:500]}

            return TestResult(
                name=endpoint,
                passed=response.status_code in [200, 201],
                response_time_ms=elapsed,
                status_code=response.status_code,
                response_data=data
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult(
                name=endpoint,
                passed=False,
                response_time_ms=elapsed,
                error=str(e)
            )

    # ==========================================================================
    # Test Suites
    # ==========================================================================

    async def test_health(self) -> TestSuite:
        """Test health endpoints."""
        suite = TestSuite(name="Health Checks")

        print(f"\n{CYAN}▸ Testing Health Endpoints{NC}")
        print(f"{CYAN}{'─'*60}{NC}")

        # Basic health
        result = await self.request("GET", "/health")
        suite.results.append(result)
        if result.passed:
            self.log(f"/health: {result.status_code} ({result.response_time_ms:.0f}ms)", "success")
            if self.verbose and result.response_data:
                self.log(f"  Status: {result.response_data.get('status')}", "debug")
        else:
            self.log(f"/health: {result.error or result.status_code}", "error")

        # V2 health
        result = await self.request("GET", "/api/v2/health")
        suite.results.append(result)
        if result.passed:
            self.log(f"/api/v2/health: {result.status_code} ({result.response_time_ms:.0f}ms)", "success")
        else:
            self.log(f"/api/v2/health: {result.error or result.status_code}", "error")

        # Detailed health
        result = await self.request("GET", "/api/v2/health/detailed")
        suite.results.append(result)
        if result.passed:
            self.log(f"/api/v2/health/detailed: {result.status_code} ({result.response_time_ms:.0f}ms)", "success")
            if self.verbose and result.response_data:
                data = result.response_data
                self.log(f"  Memory: {data.get('memory', {}).get('available_gb', 'N/A')}GB available", "debug")
                self.log(f"  Models: {data.get('models_loaded', [])}", "debug")
        else:
            self.log(f"/api/v2/health/detailed: {result.error or result.status_code}", "error")

        self.suites.append(suite)
        return suite

    async def test_content_simplify(self) -> TestSuite:
        """Test text simplification."""
        suite = TestSuite(name="Content Simplification")

        print(f"\n{CYAN}▸ Testing Content Simplification{NC}")
        print(f"{CYAN}{'─'*60}{NC}")

        test_texts = [
            {
                "name": "Scientific Text",
                "text": "Photosynthesis is the process by which plants convert light energy into chemical energy, storing it in glucose molecules synthesized from carbon dioxide and water.",
                "complexity_level": 5
            },
            {
                "name": "Math Concept",
                "text": "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
                "complexity_level": 6
            },
            {
                "name": "Hindi Educational",
                "text": "भारत का संविधान विश्व का सबसे लंबा लिखित संविधान है जो 26 जनवरी 1950 को लागू हुआ था।",
                "complexity_level": 8
            }
        ]

        for test in test_texts:
            result = await self.request("POST", "/api/v2/content/simplify", {
                "text": test["text"],
                "complexity_level": test["complexity_level"],
                "subject": "Education"
            })
            result.name = test["name"]
            suite.results.append(result)

            if result.passed:
                self.log(f"{test['name']}: {result.status_code} ({result.response_time_ms:.0f}ms)", "success")
                if result.response_data:
                    simplified = result.response_data.get("simplified_text", "")[:100]
                    self.log(f"  Input: {test['text'][:60]}...", "debug")
                    self.log(f"  Output: {simplified}...", "debug")
            else:
                self.log(f"{test['name']}: {result.error or result.status_code}", "error")

        self.suites.append(suite)
        return suite

    async def test_content_translate(self) -> TestSuite:
        """Test translation."""
        suite = TestSuite(name="Translation")

        print(f"\n{CYAN}▸ Testing Translation{NC}")
        print(f"{CYAN}{'─'*60}{NC}")

        test_cases = [
            {
                "name": "English to Hindi",
                "text": "Water is essential for life. Plants and animals need water to survive.",
                "source": "en",
                "target": "hi"
            },
            {
                "name": "Hindi to English",
                "text": "पानी जीवन के लिए आवश्यक है। पौधों और जानवरों को जीवित रहने के लिए पानी की जरूरत होती है।",
                "source": "hi",
                "target": "en"
            }
        ]

        for test in test_cases:
            result = await self.request("POST", "/api/v2/content/translate", {
                "text": test["text"],
                "source_language": test["source"],
                "target_language": test["target"]
            })
            result.name = test["name"]
            suite.results.append(result)

            if result.passed:
                self.log(f"{test['name']}: {result.status_code} ({result.response_time_ms:.0f}ms)", "success")
                if result.response_data:
                    translated = result.response_data.get("translated_text", "")[:100]
                    self.log(f"  Input: {test['text'][:50]}...", "debug")
                    self.log(f"  Output: {translated}...", "debug")
            else:
                self.log(f"{test['name']}: {result.error or result.status_code}", "error")

        self.suites.append(suite)
        return suite

    async def test_content_process(self) -> TestSuite:
        """Test full content processing pipeline."""
        suite = TestSuite(name="Full Pipeline")

        print(f"\n{CYAN}▸ Testing Full Content Pipeline{NC}")
        print(f"{CYAN}{'─'*60}{NC}")

        # Test full pipeline: simplify + translate
        result = await self.request("POST", "/api/v2/content/process", {
            "text": "The mitochondria is the powerhouse of the cell, generating ATP through oxidative phosphorylation in the electron transport chain.",
            "simplify": True,
            "translate": True,
            "generate_audio": False,
            "validate_content": True,
            "complexity_level": 6,
            "target_language": "Hindi",
            "subject": "Biology",
            "quality_mode": "balanced",
            "enable_collaboration": True
        })
        result.name = "Simplify + Translate Pipeline"
        suite.results.append(result)

        if result.passed:
            self.log(f"Full Pipeline: {result.status_code} ({result.response_time_ms:.0f}ms)", "success")
            if result.response_data:
                data = result.response_data
                self.log(f"  Simplified: {str(data.get('simplified_text', ''))[:60]}...", "debug")
                self.log(f"  Translated: {str(data.get('translated_text', ''))[:60]}...", "debug")
                self.log(f"  Validation Score: {data.get('validation_score', 'N/A')}", "debug")
                self.log(f"  Models Used: {data.get('models_used', [])}", "debug")
        else:
            self.log(f"Full Pipeline: {result.error or result.status_code}", "error")
            if result.response_data:
                self.log(f"  Error details: {result.response_data}", "debug")

        self.suites.append(suite)
        return suite

    async def test_batch_processing(self) -> TestSuite:
        """Test batch processing."""
        suite = TestSuite(name="Batch Processing")

        print(f"\n{CYAN}▸ Testing Batch Processing{NC}")
        print(f"{CYAN}{'─'*60}{NC}")

        batch_items = [
            {"id": "1", "text": "The sun is a star at the center of our solar system."},
            {"id": "2", "text": "Water freezes at zero degrees Celsius."},
            {"id": "3", "text": "Plants make their own food through photosynthesis."},
        ]

        result = await self.request("POST", "/api/v2/batch/process", {
            "items": batch_items,
            "operations": ["simplify"],
            "complexity_level": 5,
            "subject": "Science",
            "enable_collaboration": False,
            "max_concurrency": 4
        })
        result.name = "Batch Simplification"
        suite.results.append(result)

        if result.passed:
            data = result.response_data or {}
            self.log(f"Batch Processing: {result.status_code} ({result.response_time_ms:.0f}ms)", "success")
            self.log(f"  Total: {data.get('total_items', 0)}, Success: {data.get('successful', 0)}", "debug")
            self.log(f"  Throughput: {data.get('throughput_items_per_sec', 0):.1f} items/sec", "debug")
        else:
            self.log(f"Batch Processing: {result.error or result.status_code}", "error")

        self.suites.append(suite)
        return suite

    async def test_embeddings(self) -> TestSuite:
        """Test embedding generation."""
        suite = TestSuite(name="Embeddings")

        print(f"\n{CYAN}▸ Testing Embeddings{NC}")
        print(f"{CYAN}{'─'*60}{NC}")

        result = await self.request("POST", "/api/v2/batch/embed", {
            "texts": [
                "What is photosynthesis?",
                "How do plants make food?",
                "Explain the water cycle."
            ],
            "normalize": True
        })
        result.name = "Embedding Generation"
        suite.results.append(result)

        if result.passed:
            data = result.response_data or {}
            self.log(f"Embeddings: {result.status_code} ({result.response_time_ms:.0f}ms)", "success")
            embeddings = data.get("embeddings", [])
            if embeddings:
                self.log(f"  Dimension: {len(embeddings[0]) if embeddings else 'N/A'}", "debug")
                self.log(f"  Count: {len(embeddings)}", "debug")
                self.log(f"  Throughput: {data.get('throughput_texts_per_sec', 0):.1f} texts/sec", "debug")
        else:
            self.log(f"Embeddings: {result.error or result.status_code}", "error")

        self.suites.append(suite)
        return suite

    # ==========================================================================
    # Run All Tests
    # ==========================================================================

    async def run_all(self):
        """Run all test suites."""
        print(f"\n{CYAN}{'═'*60}")
        print(f"  {BOLD}🧪 SHIKSHA SETU LIVE BACKEND TEST{NC}")
        print(f"{CYAN}{'═'*60}{NC}")
        print(f"\n   Base URL: {self.base_url}")
        print(f"   Verbose: {self.verbose}")

        # Wait for server to be ready
        print(f"\n   Checking server availability...")
        for i in range(5):
            result = await self.request("GET", "/health", timeout=5.0)
            if result.passed:
                print(f"   {GREEN}✓ Server is ready{NC}")
                break
            await asyncio.sleep(2)
        else:
            print(f"   {RED}✗ Server not responding at {self.base_url}{NC}")
            return

        # Run test suites
        await self.test_health()
        await self.test_content_simplify()
        await self.test_content_translate()
        await self.test_content_process()
        await self.test_batch_processing()
        await self.test_embeddings()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print(f"\n{CYAN}{'═'*60}")
        print(f"  {BOLD}📊 TEST SUMMARY{NC}")
        print(f"{CYAN}{'═'*60}{NC}\n")

        total_passed = 0
        total_failed = 0
        total_time = 0.0

        for suite in self.suites:
            passed = suite.passed
            failed = suite.failed
            total_passed += passed
            total_failed += failed

            suite_time = sum(r.response_time_ms for r in suite.results)
            total_time += suite_time

            status = f"{GREEN}PASS{NC}" if failed == 0 else f"{RED}FAIL{NC}"
            print(f"   {suite.name}: {passed}/{passed+failed} {status} ({suite_time:.0f}ms)")

        print(f"\n   {'─'*50}")
        print(f"   {BOLD}Total:{NC} {total_passed} passed, {total_failed} failed")
        print(f"   {BOLD}Time:{NC} {total_time:.0f}ms ({total_time/1000:.1f}s)")

        if total_failed == 0:
            print(f"\n   {GREEN}{BOLD}✓ All tests passed!{NC}")
        else:
            print(f"\n   {YELLOW}! {total_failed} test(s) failed{NC}")


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Live Backend Tester")
    parser.add_argument("--port", type=int, default=9000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    async with BackendTester(base_url, verbose=args.verbose) as tester:
        await tester.run_all()


if __name__ == "__main__":
    asyncio.run(main())
