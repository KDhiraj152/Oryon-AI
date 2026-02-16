#!/usr/bin/env python3
"""
Hardware Performance Benchmarks
================================

Validates optimizations against baseline:
- SIMD operations throughput
- GPU distance computation
- Zero-copy memory operations
- HNSW search performance
- Embedding cache efficiency

Run with: python -m backend.core.optimized.benchmark
"""

import gc
import os
import sys
import time
from collections.abc import Callable

import numpy as np

# Add parent to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

def benchmark(name: str, func: Callable, iterations: int = 100) -> tuple[float, float]:
    """
    Benchmark a function.

    Returns (mean_ms, std_ms)
    """
    # Warmup
    for _ in range(5):
        func()

    # GC before benchmark
    gc.collect()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    mean = np.mean(times)
    std = np.std(times)
    return float(mean), float(std)

def print_result(
    name: str, mean: float, std: float, baseline_mean: float | None = None
):
    """Print benchmark result."""
    speedup = ""
    if baseline_mean is not None and baseline_mean > 0:
        ratio = baseline_mean / mean
        speedup = f" ({ratio:.2f}x vs baseline)"
    print(f"  {name}: {mean:.3f} ± {std:.3f} ms{speedup}")

def benchmark_simd():
    """Benchmark SIMD operations."""
    print("\n=== SIMD Operations ===")

    from backend.core.optimized.simd_ops import (
        cosine_similarity_batch,
        get_simd_capabilities,
        l2_distance_batch,
        normalize_vectors,
        top_k_indices,
    )

    caps = get_simd_capabilities()
    print(f"SIMD capabilities: {caps}")

    # Test data
    n_queries = 100
    n_docs = 10000
    dim = 1024

    queries = np.random.randn(n_queries, dim).astype(np.float32)
    docs = np.random.randn(n_docs, dim).astype(np.float32)
    query = queries[0]
    scores = np.random.randn(n_docs).astype(np.float32)

    # Baseline: naive numpy
    def baseline_cosine():
        q_norm = np.linalg.norm(queries, axis=1, keepdims=True)
        d_norm = np.linalg.norm(docs, axis=1, keepdims=True)
        return np.dot(queries / q_norm, (docs / d_norm).T)

    def optimized_cosine():
        return cosine_similarity_batch(queries, docs)

    # Benchmark cosine similarity
    base_mean, base_std = benchmark("Baseline cosine", baseline_cosine, 20)
    opt_mean, opt_std = benchmark("Optimized cosine", optimized_cosine, 20)
    print_result("Baseline cosine", base_mean, base_std)
    print_result("Optimized cosine", opt_mean, opt_std, base_mean)

    # L2 distance
    def baseline_l2():
        diff = docs - query
        return np.linalg.norm(diff, axis=1)

    def optimized_l2():
        return l2_distance_batch(query, docs)

    base_mean, base_std = benchmark("Baseline L2", baseline_l2, 50)
    opt_mean, opt_std = benchmark("Optimized L2", optimized_l2, 50)
    print_result("Baseline L2", base_mean, base_std)
    print_result("Optimized L2", opt_mean, opt_std, base_mean)

    # Top-k selection
    k = 100

    def baseline_topk():
        return np.argsort(scores)[:k]

    def optimized_topk():
        return top_k_indices(scores, k, largest=False)

    base_mean, base_std = benchmark("Baseline top-k", baseline_topk, 100)
    opt_mean, opt_std = benchmark("Optimized top-k", optimized_topk, 100)
    print_result("Baseline top-k", base_mean, base_std)
    print_result("Optimized top-k", opt_mean, opt_std, base_mean)

    # Normalization
    def baseline_norm():
        norms = np.linalg.norm(docs, axis=1, keepdims=True)
        return docs / norms

    def optimized_norm():
        return normalize_vectors(docs)

    base_mean, base_std = benchmark("Baseline normalize", baseline_norm, 50)
    opt_mean, opt_std = benchmark("Optimized normalize", optimized_norm, 50)
    print_result("Baseline normalize", base_mean, base_std)
    print_result("Optimized normalize", opt_mean, opt_std, base_mean)

def benchmark_hnsw():
    """Benchmark HNSW search."""
    print("\n=== HNSW Search ===")

    from backend.core.optimized.hnsw_accel import (
        HNSWConfig,
        OptimizedHNSWSearcher,
    )

    # Test data
    n_vectors = 50000
    dim = 1024
    k = 10

    print(f"Building index: {n_vectors} vectors, dim={dim}")
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)

    config = HNSWConfig(
        metric="cosine",
        use_gpu=True,
        ef_search=40,
    )

    searcher = OptimizedHNSWSearcher(vectors, config)
    print(f"Device: {searcher.distance_computer.device_type}")

    # Single query benchmark
    query = np.random.randn(dim).astype(np.float32)

    def search_single():
        return searcher.search(query, k=k)

    mean, std = benchmark("Single search", search_single, 50)
    print_result("Single query (k=10)", mean, std)

    # Batch query benchmark
    n_queries = 100
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    def search_batch():
        return searcher.batch_search(queries, k=k)

    mean, std = benchmark("Batch search", search_batch, 10)
    print_result(f"Batch ({n_queries} queries, k={k})", mean, std)
    qps = n_queries / (mean / 1000)
    print(f"  Throughput: {qps:.0f} queries/sec")

    stats = searcher.get_stats()
    print(f"  GPU ratio: {stats['gpu_ratio']:.1%}")

def benchmark_embedding_cache():
    """Benchmark embedding cache operations."""
    print("\n=== Embedding Cache ===")

    # Skip if database not available
    try:
        from backend.infra.cache.embedding_cache import EmbeddingCache
    except (ImportError, RuntimeError) as e:
        print(f"  Skipped: {e}")
        return

    import tempfile

    # Create temporary cache
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        cache = EmbeddingCache(db_path=db_path, memory_cache_size=10000)

        # Generate test data
        n_items = 1000
        dim = 1024
        texts = [f"test text {i} for embedding" for i in range(n_items)]
        embeddings = [np.random.randn(dim).astype(np.float32) for _ in range(n_items)]

        # Serialize benchmark
        def serialize():
            for emb in embeddings[:100]:
                cache._serialize_embedding(emb)

        def deserialize():
            data = cache._serialize_embedding(embeddings[0])
            for _ in range(100):
                cache._deserialize_embedding(data, dim)

        mean, std = benchmark("Serialize 100 embeddings", serialize, 10)
        print_result("Serialize", mean, std)

        mean, std = benchmark("Deserialize 100 embeddings", deserialize, 10)
        print_result("Deserialize", mean, std)

        # Memory cache benchmark
        def add_to_cache():
            for i, text in enumerate(texts[:100]):
                key = cache._hash_text(text)
                cache._add_to_memory_cache(key, embeddings[i])

        mean, std = benchmark("Add 100 to memory cache", add_to_cache, 10)
        print_result("Memory cache add", mean, std)

    finally:
        os.unlink(db_path)

def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("Hardware Performance Benchmarks")
    print("=" * 60)

    import platform

    print(f"Platform: {platform.machine()}")
    print(f"Python: {platform.python_version()}")

    try:
        import torch

        if torch.backends.mps.is_available():
            print("GPU: Apple Metal (MPS)")
        elif torch.cuda.is_available():
            print(f"GPU: CUDA ({torch.cuda.get_device_name()})")
        else:
            print("GPU: None (CPU only)")
    except ImportError:
        print("GPU: Unknown (torch not available)")

    # Run benchmarks
    benchmark_simd()

    try:
        benchmark_hnsw()
    except (RuntimeError, OSError, ValueError) as e:
        print(f"\n=== HNSW Search ===\n  Skipped: {e}")

    try:
        benchmark_embedding_cache()
    except (RuntimeError, OSError, ValueError) as e:
        print(f"\n=== Embedding Cache ===\n  Skipped: {e}")

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
