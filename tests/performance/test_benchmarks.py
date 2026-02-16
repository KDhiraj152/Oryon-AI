"""
Performance Benchmarks for Oryon
======================================

Automated performance regression detection for critical paths.
Run with: pytest tests/performance/ -v --benchmark-only

These tests establish baselines and detect regressions in:
- API response times
- Database query performance
- Cache hit rates
- Memory usage
- Model inference latency
"""

import asyncio
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List

import pytest

# Try to import benchmark plugin
try:
    import pytest_benchmark

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    std_dev_ms: float
    min_ms: float
    max_ms: float
    ops_per_sec: float

    def __str__(self):
        return (
            f"{self.name}: {self.mean_ms:.2f}ms avg "
            f"(±{self.std_dev_ms:.2f}ms, {self.ops_per_sec:.1f} ops/sec)"
        )


class PerformanceBenchmark:
    """Simple benchmark utility for when pytest-benchmark isn't available."""

    def __init__(self, warmup: int = 3, iterations: int = 10):
        self.warmup = warmup
        self.iterations = iterations
        self.results: list[BenchmarkResult] = []

    def run(self, func: Callable, name: str | None = None) -> BenchmarkResult:
        """Run a synchronous benchmark."""
        name = name or func.__name__

        # Warmup
        for _ in range(self.warmup):
            func()

        # Benchmark
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        result = BenchmarkResult(
            name=name,
            iterations=self.iterations,
            mean_ms=statistics.mean(times),
            median_ms=statistics.median(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            min_ms=min(times),
            max_ms=max(times),
            ops_per_sec=1000 / statistics.mean(times)
            if statistics.mean(times) > 0
            else 0,
        )
        self.results.append(result)
        return result

    async def run_async(
        self, func: Callable, name: str | None = None
    ) -> BenchmarkResult:
        """Run an async benchmark."""
        name = name or func.__name__

        # Warmup
        for _ in range(self.warmup):
            await func()

        # Benchmark
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            await func()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        result = BenchmarkResult(
            name=name,
            iterations=self.iterations,
            mean_ms=statistics.mean(times),
            median_ms=statistics.median(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            min_ms=min(times),
            max_ms=max(times),
            ops_per_sec=1000 / statistics.mean(times)
            if statistics.mean(times) > 0
            else 0,
        )
        self.results.append(result)
        return result

    def report(self) -> str:
        """Generate benchmark report."""
        lines = ["=" * 60, "BENCHMARK RESULTS", "=" * 60]
        for result in sorted(self.results, key=lambda r: r.mean_ms):
            lines.append(str(result))
        lines.append("=" * 60)
        return "\n".join(lines)


# Performance thresholds (in milliseconds)
THRESHOLDS = {
    "health_check": 50,  # Health endpoint should be < 50ms
    "auth_validation": 10,  # Token validation < 10ms
    "cache_get": 5,  # Cache retrieval < 5ms
    "embedding_single": 100,  # Single text embedding < 100ms
    "embedding_batch_10": 200,  # 10 text embeddings < 200ms
    "simple_query": 500,  # Simple AI query < 500ms
    "complex_query": 2000,  # Complex AI query < 2000ms
    "translation_short": 300,  # Short text translation < 300ms
    "simplification": 400,  # Text simplification < 400ms
}


class TestAPIPerformance:
    """Test API endpoint performance."""

    @pytest.fixture
    def benchmark_util(self):
        return PerformanceBenchmark(warmup=2, iterations=5)

    @pytest.fixture
    def client(self):
        """Get test client."""
        from fastapi.testclient import TestClient

        from backend.api.main import app

        return TestClient(app)

    def test_health_endpoint_performance(self, client, benchmark_util):
        """Health check should respond quickly."""

        def health_check():
            response = client.get("/api/v2/health")
            assert response.status_code == 200

        result = benchmark_util.run(health_check, "health_check")
        assert (
            result.mean_ms < THRESHOLDS["health_check"]
        ), f"Health check too slow: {result.mean_ms:.2f}ms > {THRESHOLDS['health_check']}ms"

    def test_hardware_status_performance(self, client, benchmark_util):
        """Hardware status endpoint performance."""

        def hardware_status():
            response = client.get("/api/v2/hardware/status")
            assert response.status_code == 200

        result = benchmark_util.run(hardware_status, "hardware_status")
        # Should be quick since it's just reading system info
        assert result.mean_ms < 100, f"Hardware status too slow: {result.mean_ms:.2f}ms"

    def test_models_status_performance(self, client, benchmark_util):
        """Models status endpoint should be cached."""

        def models_status():
            response = client.get("/api/v2/models/status")
            assert response.status_code == 200

        result = benchmark_util.run(models_status, "models_status")
        print(f"\nModels status: {result}")


class TestCachePerformance:
    """Test caching layer performance."""

    @pytest.fixture
    def benchmark_util(self):
        return PerformanceBenchmark(warmup=5, iterations=20)

    def test_fast_hash_performance(self, benchmark_util):
        """Fast hash should be very quick."""
        from backend.utils.hashing import fast_hash

        test_data = "Test content for hashing" * 100  # ~2.4KB

        def hash_operation():
            fast_hash(test_data)

        result = benchmark_util.run(hash_operation, "fast_hash")
        assert result.mean_ms < 1, f"Hash too slow: {result.mean_ms:.2f}ms"
        print(f"\nFast hash: {result}")

    def test_serialization_performance(self, benchmark_util):
        """Test JSON serialization speed."""
        import orjson

        test_data = {
            "text": "Sample content " * 100,
            "embeddings": [0.1] * 768,
            "metadata": {"key": "value"},
        }

        def serialize():
            orjson.dumps(test_data)

        def deserialize():
            data = orjson.dumps(test_data)
            orjson.loads(data)

        result = benchmark_util.run(serialize, "orjson_serialize")
        assert result.mean_ms < 1, f"Serialization too slow: {result.mean_ms:.2f}ms"

        result = benchmark_util.run(deserialize, "orjson_round_trip")
        print(f"\nOrjson round trip: {result}")


class TestEmbeddingPerformance:
    """Test embedding model performance."""

    @pytest.fixture
    def benchmark_util(self):
        return PerformanceBenchmark(warmup=1, iterations=3)

    @pytest.mark.skip(reason="Requires model to be loaded")
    def test_single_embedding_performance(self, benchmark_util):
        """Single text embedding should be fast."""
        from backend.services.chat.rag import get_embedder

        embedder = get_embedder()
        test_text = "This is a sample text for embedding."

        def embed_single():
            embedder.encode([test_text])

        result = benchmark_util.run(embed_single, "embedding_single")
        assert (
            result.mean_ms < THRESHOLDS["embedding_single"]
        ), f"Single embedding too slow: {result.mean_ms:.2f}ms"

    @pytest.mark.skip(reason="Requires model to be loaded")
    def test_batch_embedding_performance(self, benchmark_util):
        """Batch embeddings should be efficient."""
        from backend.services.chat.rag import get_embedder

        embedder = get_embedder()
        test_texts = [f"Sample text number {i}" for i in range(10)]

        def embed_batch():
            embedder.encode(test_texts)

        result = benchmark_util.run(embed_batch, "embedding_batch_10")
        assert (
            result.mean_ms < THRESHOLDS["embedding_batch_10"]
        ), f"Batch embedding too slow: {result.mean_ms:.2f}ms"


class TestMemoryUsage:
    """Test memory efficiency."""

    def test_memory_baseline(self):
        """Establish memory baseline for key components."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        baseline = process.memory_info().rss / 1024 / 1024  # MB

        # Import key modules
        from backend.api.main import app
        from backend.core.config import settings

        after_import = process.memory_info().rss / 1024 / 1024
        import_overhead = after_import - baseline

        print(f"\nMemory baseline: {baseline:.1f}MB")
        print(f"After imports: {after_import:.1f}MB")
        print(f"Import overhead: {import_overhead:.1f}MB")

        # Should not exceed 500MB just for imports
        assert (
            import_overhead < 500
        ), f"Import overhead too high: {import_overhead:.1f}MB"


class TestConcurrencyPerformance:
    """Test concurrent request handling."""

    @pytest.fixture
    def benchmark_util(self):
        return PerformanceBenchmark(warmup=1, iterations=3)

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, benchmark_util):
        """Test concurrent request handling."""
        import httpx

        async def concurrent_requests():
            async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
                tasks = [client.get("/api/v2/health") for _ in range(10)]
                await asyncio.gather(*tasks, return_exceptions=True)

        # Skip if server not running
        try:
            result = await benchmark_util.run_async(
                concurrent_requests, "concurrent_health_10"
            )
            print(f"\nConcurrent health checks (10): {result}")
        except Exception:
            pytest.skip("Backend server not running")


def run_all_benchmarks():
    """Run all benchmarks and generate report."""
    benchmark = PerformanceBenchmark(warmup=3, iterations=10)

    # Import tests
    print("\n" + "=" * 60)
    print("Running Oryon Performance Benchmarks")
    print("=" * 60)

    # Test imports
    def test_imports():
        from backend.api.main import app
        from backend.core.config import settings

    result = benchmark.run(test_imports, "module_imports")
    print(f"Module imports: {result}")

    # Test hashing
    from backend.utils.hashing import fast_hash

    data = "x" * 10000
    result = benchmark.run(lambda: fast_hash(data), "fast_hash_10kb")
    print(f"Fast hash 10KB: {result}")

    # Test JSON
    import orjson

    obj = {"data": list(range(1000))}
    result = benchmark.run(lambda: orjson.dumps(obj), "orjson_serialize")
    print(f"Orjson serialize: {result}")

    print("\n" + benchmark.report())


if __name__ == "__main__":
    run_all_benchmarks()
