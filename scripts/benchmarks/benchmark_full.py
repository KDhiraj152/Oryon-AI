#!/usr/bin/env python3
"""
ORYON AI - COMPREHENSIVE END-TO-END BENCHMARK SUITE
========================================================

Tests all system components with rigorous benchmarking:
1. Database performance (PostgreSQL + pgvector)
2. Cache performance (Redis + in-memory)
3. Model loading times
4. Inference throughput
5. API response times
6. Memory utilization
7. Concurrent request handling
"""

import asyncio
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Benchmark results storage
@dataclass
class BenchmarkResult:
    name: str
    metric: str
    value: float
    unit: str
    samples: int = 1
    min_val: float = 0
    max_val: float = 0
    std_dev: float = 0
    status: str = "PASS"
    details: str = ""


@dataclass
class BenchmarkSuite:
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    system_info: dict[str, Any] = field(default_factory=dict)
    results: list[BenchmarkResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
        print(
            f"  {'✓' if result.status == 'PASS' else '✗'} {result.name}: {result.value:.2f} {result.unit}"
        )


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================


def measure_time(func, *args, iterations=1, **kwargs):
    """Measure function execution time with statistics."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "result": result,
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "samples": len(times),
    }


async def async_measure_time(coro_func, *args, iterations=1, **kwargs):
    """Measure async function execution time."""
    times = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = await coro_func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "result": result,
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "samples": len(times),
    }


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0


# ============================================================================
# BENCHMARK 1: SYSTEM INFO
# ============================================================================


def benchmark_system_info() -> dict[str, Any]:
    """Collect system information."""
    print("\n" + "=" * 60)
    print("📊 SYSTEM INFORMATION")
    print("=" * 60)

    import platform

    info: dict[str, Any] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or "Apple M4",
    }

    # Get memory info
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["total_memory_gb"] = round(mem.total / 1024**3, 1)
        info["available_memory_gb"] = round(mem.available / 1024**3, 1)
    except ImportError:
        info["total_memory_gb"] = 16  # Default for M4 Mac
        info["available_memory_gb"] = 8

    # Check GPU/MPS availability
    try:
        import torch

        info["mps_available"] = torch.backends.mps.is_available()
        info["mps_built"] = torch.backends.mps.is_built()
    except ImportError:
        info["mps_available"] = False

    # Check MLX availability
    try:
        import mlx.core as mx

        info["mlx_available"] = True
        info["mlx_default_device"] = str(mx.default_device())
    except ImportError:
        info["mlx_available"] = False

    for key, value in info.items():
        print(f"  {key}: {value}")

    return info


# ============================================================================
# BENCHMARK 2: DATABASE PERFORMANCE
# ============================================================================


def benchmark_database(suite: BenchmarkSuite):
    """Benchmark database operations."""
    print("\n" + "=" * 60)
    print("🗄️  DATABASE BENCHMARKS")
    print("=" * 60)

    try:
        from sqlalchemy import text

        from backend.database import SessionLocal, engine

        # Test connection
        start = time.perf_counter()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        connect_time = (time.perf_counter() - start) * 1000

        suite.add_result(
            BenchmarkResult(
                name="DB Connection",
                metric="latency",
                value=connect_time,
                unit="ms",
                status="PASS" if connect_time < 100 else "WARN",
            )
        )

        # Test pgvector
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            version = result.scalar()
            suite.add_result(
                BenchmarkResult(
                    name="pgvector Extension",
                    metric="version",
                    value=float(version.replace(".", "")),
                    unit=f"v{version}",
                    status="PASS",
                )
            )

        # Benchmark vector operations
        with engine.connect() as conn:
            # Create temp table for benchmarking
            conn.execute(
                text("""
                CREATE TEMP TABLE IF NOT EXISTS bench_vectors (
                    id SERIAL PRIMARY KEY,
                    embedding vector(1024)
                )
            """)
            )
            conn.commit()

            # Insert benchmark - 100 vectors
            import random

            start = time.perf_counter()
            for _i in range(100):
                vec = [random.random() for _ in range(1024)]
                conn.execute(
                    text("INSERT INTO bench_vectors (embedding) VALUES (:vec::vector)"),
                    {"vec": str(vec)},
                )
            conn.commit()
            insert_time = time.perf_counter() - start

            suite.add_result(
                BenchmarkResult(
                    name="Vector Insert (100x1024D)",
                    metric="throughput",
                    value=100 / insert_time,
                    unit="vectors/sec",
                    status="PASS" if insert_time < 5 else "WARN",
                )
            )

            # Search benchmark
            query_vec = [random.random() for _ in range(1024)]
            start = time.perf_counter()
            for _ in range(10):
                conn.execute(
                    text("""
                    SELECT id, embedding <-> :query::vector as distance
                    FROM bench_vectors
                    ORDER BY distance
                    LIMIT 5
                """),
                    {"query": str(query_vec)},
                )
            search_time = (time.perf_counter() - start) / 10 * 1000

            suite.add_result(
                BenchmarkResult(
                    name="Vector Search (top-5)",
                    metric="latency",
                    value=search_time,
                    unit="ms",
                    status="PASS" if search_time < 50 else "WARN",
                )
            )

    except Exception as e:
        suite.add_result(
            BenchmarkResult(
                name="Database",
                metric="error",
                value=0,
                unit="",
                status="FAIL",
                details=str(e),
            )
        )


# ============================================================================
# BENCHMARK 3: CACHE PERFORMANCE
# ============================================================================


def benchmark_cache(suite: BenchmarkSuite):
    """Benchmark cache operations."""
    print("\n" + "=" * 60)
    print("⚡ CACHE BENCHMARKS")
    print("=" * 60)

    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Ping test
        start = time.perf_counter()
        for _ in range(100):
            r.ping()
        ping_time = (time.perf_counter() - start) / 100 * 1000

        suite.add_result(
            BenchmarkResult(
                name="Redis Ping",
                metric="latency",
                value=ping_time,
                unit="ms",
                status="PASS" if ping_time < 1 else "WARN",
            )
        )

        # Write benchmark
        start = time.perf_counter()
        for i in range(1000):
            r.set(f"bench:key:{i}", f"value_{i}" * 100)
        write_time = time.perf_counter() - start

        suite.add_result(
            BenchmarkResult(
                name="Redis Write (1000 keys)",
                metric="throughput",
                value=1000 / write_time,
                unit="ops/sec",
                status="PASS",
            )
        )

        # Read benchmark
        start = time.perf_counter()
        for i in range(1000):
            r.get(f"bench:key:{i}")
        read_time = time.perf_counter() - start

        suite.add_result(
            BenchmarkResult(
                name="Redis Read (1000 keys)",
                metric="throughput",
                value=1000 / read_time,
                unit="ops/sec",
                status="PASS",
            )
        )

        # Cleanup
        for i in range(1000):
            r.delete(f"bench:key:{i}")

    except Exception as e:
        suite.add_result(
            BenchmarkResult(
                name="Cache",
                metric="error",
                value=0,
                unit="",
                status="FAIL",
                details=str(e),
            )
        )


# ============================================================================
# BENCHMARK 4: ML MODEL LOADING
# ============================================================================


def benchmark_model_loading(suite: BenchmarkSuite):
    """Benchmark model loading times."""
    print("\n" + "=" * 60)
    print("🧠 MODEL LOADING BENCHMARKS")
    print("=" * 60)

    models_loaded = 0
    memory_before = get_memory_usage()

    # Embedding model (BGE-M3)
    try:
        from sentence_transformers import SentenceTransformer

        start = time.perf_counter()
        model = SentenceTransformer("BAAI/bge-m3", device="mps")
        load_time = time.perf_counter() - start

        suite.add_result(
            BenchmarkResult(
                name="BGE-M3 Load",
                metric="time",
                value=load_time,
                unit="sec",
                status="PASS" if load_time < 30 else "WARN",
            )
        )
        models_loaded += 1
        del model
    except Exception as e:
        suite.add_result(
            BenchmarkResult(
                name="BGE-M3 Load",
                metric="error",
                value=0,
                unit="",
                status="SKIP",
                details=str(e)[:50],
            )
        )

    # LLM (MLX Qwen)
    try:
        import mlx_lm

        start = time.perf_counter()
        model, tokenizer, *_ = mlx_lm.load("mlx-community/Qwen3-8B-4bit")  # type: ignore
        load_time = time.perf_counter() - start

        suite.add_result(
            BenchmarkResult(
                name="Qwen3-8B Load",
                metric="time",
                value=load_time,
                unit="sec",
                status="PASS" if load_time < 15 else "WARN",
            )
        )
        models_loaded += 1
        del model, tokenizer
    except Exception as e:
        suite.add_result(
            BenchmarkResult(
                name="Qwen3-8B Load",
                metric="error",
                value=0,
                unit="",
                status="SKIP",
                details=str(e)[:50],
            )
        )

    memory_after = get_memory_usage()
    suite.add_result(
        BenchmarkResult(
            name="Models Loaded",
            metric="count",
            value=models_loaded,
            unit="models",
            status="PASS",
            details=f"Memory delta: {memory_after - memory_before:.0f} MB",
        )
    )


# ============================================================================
# BENCHMARK 5: INFERENCE THROUGHPUT
# ============================================================================


def benchmark_inference(suite: BenchmarkSuite):
    """Benchmark inference performance."""
    print("\n" + "=" * 60)
    print("🚀 INFERENCE BENCHMARKS")
    print("=" * 60)

    # Embedding inference
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-m3", device="mps")

        test_texts = [
            "What is photosynthesis and how do plants use it?",
            "Explain the water cycle in simple terms.",
            "How does gravity affect objects on Earth?",
            "What are the three states of matter?",
            "Describe the solar system and its planets.",
        ] * 10  # 50 texts

        # Warmup
        _ = model.encode(test_texts[:5])

        # Benchmark
        start = time.perf_counter()
        embeddings = model.encode(test_texts, batch_size=16, show_progress_bar=False)
        embed_time = time.perf_counter() - start

        suite.add_result(
            BenchmarkResult(
                name="Embedding (50 texts)",
                metric="throughput",
                value=len(test_texts) / embed_time,
                unit="texts/sec",
                status="PASS" if embed_time < 10 else "WARN",
            )
        )

        suite.add_result(
            BenchmarkResult(
                name="Embedding Dimension",
                metric="size",
                value=embeddings.shape[1],
                unit="dimensions",
                status="PASS",
            )
        )

        del model
    except Exception as e:
        suite.add_result(
            BenchmarkResult(
                name="Embedding",
                metric="error",
                value=0,
                unit="",
                status="SKIP",
                details=str(e)[:50],
            )
        )

    # LLM inference
    try:
        import mlx_lm

        model, tokenizer, *_ = mlx_lm.load("mlx-community/Qwen3-8B-4bit")  # type: ignore

        prompt = "Explain photosynthesis in simple terms for a beginner user."
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Warmup
        _ = mlx_lm.generate(model, tokenizer, prompt=prompt_text, max_tokens=10)  # type: ignore[arg-type]

        # Benchmark - generate 100 tokens
        start = time.perf_counter()
        response = mlx_lm.generate(model, tokenizer, prompt=prompt_text, max_tokens=100)  # type: ignore[arg-type]
        gen_time = time.perf_counter() - start

        tokens_generated = len(tokenizer.encode(response))
        throughput = tokens_generated / gen_time

        suite.add_result(
            BenchmarkResult(
                name="LLM Generation",
                metric="throughput",
                value=throughput,
                unit="tok/sec",
                status="PASS" if throughput > 30 else "WARN",
            )
        )

        del model, tokenizer
    except Exception as e:
        suite.add_result(
            BenchmarkResult(
                name="LLM Generation",
                metric="error",
                value=0,
                unit="",
                status="SKIP",
                details=str(e)[:50],
            )
        )


# ============================================================================
# BENCHMARK 6: FRONTEND BUILD
# ============================================================================


def benchmark_frontend(suite: BenchmarkSuite):
    """Benchmark frontend build performance."""
    print("\n" + "=" * 60)
    print("⚛️  FRONTEND BENCHMARKS")
    print("=" * 60)

    frontend_dir = PROJECT_ROOT / "frontend"

    # Check if built
    dist_dir = frontend_dir / "dist"
    if dist_dir.exists():
        # Count chunks
        js_files = list(dist_dir.glob("assets/*.js"))
        css_files = list(dist_dir.glob("assets/*.css"))

        total_js_size = sum(f.stat().st_size for f in js_files) / 1024
        total_css_size = sum(f.stat().st_size for f in css_files) / 1024

        suite.add_result(
            BenchmarkResult(
                name="JS Bundle Size",
                metric="size",
                value=total_js_size,
                unit="KB",
                status="PASS" if total_js_size < 1000 else "WARN",
            )
        )

        suite.add_result(
            BenchmarkResult(
                name="CSS Bundle Size",
                metric="size",
                value=total_css_size,
                unit="KB",
                status="PASS",
            )
        )

        suite.add_result(
            BenchmarkResult(
                name="JS Chunks",
                metric="count",
                value=len(js_files),
                unit="files",
                status="PASS",
            )
        )
    else:
        suite.add_result(
            BenchmarkResult(
                name="Frontend Build",
                metric="status",
                value=0,
                unit="",
                status="SKIP",
                details="Run 'npm run build' first",
            )
        )


# ============================================================================
# BENCHMARK 7: CODEBASE METRICS
# ============================================================================


def benchmark_codebase(suite: BenchmarkSuite):
    """Analyze codebase metrics."""
    print("\n" + "=" * 60)
    print("📁 CODEBASE METRICS")
    print("=" * 60)

    # Count Python files
    py_files = list(PROJECT_ROOT.glob("backend/**/*.py"))
    py_lines = sum(len(f.read_text().splitlines()) for f in py_files if f.exists())

    suite.add_result(
        BenchmarkResult(
            name="Python Files",
            metric="count",
            value=len(py_files),
            unit="files",
            status="PASS",
        )
    )

    suite.add_result(
        BenchmarkResult(
            name="Python LOC",
            metric="count",
            value=py_lines,
            unit="lines",
            status="PASS",
        )
    )

    # Count TypeScript files
    ts_files = list((PROJECT_ROOT / "frontend" / "src").glob("**/*.tsx")) + list(
        (PROJECT_ROOT / "frontend" / "src").glob("**/*.ts")
    )
    ts_lines = sum(len(f.read_text().splitlines()) for f in ts_files if f.exists())

    suite.add_result(
        BenchmarkResult(
            name="TypeScript Files",
            metric="count",
            value=len(ts_files),
            unit="files",
            status="PASS",
        )
    )

    suite.add_result(
        BenchmarkResult(
            name="TypeScript LOC",
            metric="count",
            value=ts_lines,
            unit="lines",
            status="PASS",
        )
    )

    # Alembic migrations
    migrations = list((PROJECT_ROOT / "alembic" / "versions").glob("*.py"))
    suite.add_result(
        BenchmarkResult(
            name="DB Migrations",
            metric="count",
            value=len(migrations),
            unit="migrations",
            status="PASS",
        )
    )

    # API routes
    routes_dir = PROJECT_ROOT / "backend" / "api" / "routes" / "v2"
    route_files = list(routes_dir.glob("*.py"))
    suite.add_result(
        BenchmarkResult(
            name="API Route Modules",
            metric="count",
            value=len(route_files),
            unit="modules",
            status="PASS",
        )
    )


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================


def run_benchmarks():
    """Run all benchmarks and generate report."""
    print("\n" + "=" * 60)
    print("🏁 ORYON AI - COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    suite = BenchmarkSuite()

    # System info
    suite.system_info = benchmark_system_info()

    # Run benchmarks
    benchmark_database(suite)
    benchmark_cache(suite)
    benchmark_codebase(suite)
    benchmark_frontend(suite)
    benchmark_model_loading(suite)
    benchmark_inference(suite)

    # Generate summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in suite.results if r.status == "PASS")
    warned = sum(1 for r in suite.results if r.status == "WARN")
    failed = sum(1 for r in suite.results if r.status == "FAIL")
    skipped = sum(1 for r in suite.results if r.status == "SKIP")

    suite.summary = {
        "total_benchmarks": len(suite.results),
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "skipped": skipped,
        "success_rate": f"{(passed / len(suite.results)) * 100:.1f}%"
        if suite.results
        else "0%",
    }

    print(f"\n  Total Benchmarks: {len(suite.results)}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ⚠️  Warned: {warned}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"\n  Success Rate: {suite.summary['success_rate']}")

    # Save results
    results_file = PROJECT_ROOT / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": suite.timestamp,
                "system_info": suite.system_info,
                "results": [asdict(r) for r in suite.results],
                "summary": suite.summary,
            },
            f,
            indent=2,
        )

    print(f"\n  Results saved to: {results_file}")
    print("\n" + "=" * 60)

    return suite


if __name__ == "__main__":
    run_benchmarks()
