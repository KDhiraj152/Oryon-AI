#!/usr/bin/env python3
"""
Hardware Optimization Benchmark Suite
Tests SIMD, GPU, and memory optimization performance
"""

import platform
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


# Hardware detection
def detect_hardware() -> dict[str, Any]:
    """Detect available hardware acceleration."""
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gpu_available": False,
        "gpu_type": None,
        "mps_available": False,
        "cuda_available": False,
        "mlx_available": False,
    }

    # Check PyTorch MPS (Apple Silicon)
    try:
        import torch

        info["torch"] = torch.__version__
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
            info["gpu_available"] = True
            info["gpu_type"] = "Apple Metal (MPS)"
        elif torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_available"] = True
            info["gpu_type"] = f"CUDA ({torch.cuda.get_device_name(0)})"
    except ImportError:
        pass

    # Check MLX (Apple Silicon)
    try:
        import mlx.core as mx

        info["mlx_available"] = True
        info["mlx"] = "available"
    except ImportError:
        pass

    return info


def benchmark_numpy_simd(
    dim: int = 1024, n_vectors: int = 10000, iterations: int = 10
) -> dict[str, float]:
    """Benchmark NumPy SIMD-optimized operations."""
    print(f"\n{'='*60}")
    print("NUMPY SIMD BENCHMARK")
    print(f"Dimensions: {dim}, Vectors: {n_vectors}, Iterations: {iterations}")
    print(f"{'='*60}")

    # Generate test data
    np.random.seed(42)
    query = np.random.randn(dim).astype(np.float32)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)

    # Normalize for cosine similarity
    query = query / np.linalg.norm(query)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_normalized = vectors / norms

    results = {}

    # Test 1: Dot product (basic)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        scores = np.dot(vectors, query)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])  # Skip warmup
    throughput = n_vectors / avg_time / 1e6
    results["dot_product_ms"] = avg_time * 1000
    results["dot_product_throughput_M"] = throughput
    print(f"Dot Product: {avg_time*1000:.3f}ms, {throughput:.2f}M vectors/sec")

    # Test 2: Cosine similarity (normalized dot product)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        scores = np.dot(vectors_normalized, query)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = n_vectors / avg_time / 1e6
    results["cosine_sim_ms"] = avg_time * 1000
    results["cosine_sim_throughput_M"] = throughput
    print(f"Cosine Similarity: {avg_time*1000:.3f}ms, {throughput:.2f}M vectors/sec")

    # Test 3: L2 distance
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        diff = vectors - query
        distances = np.sqrt(np.sum(diff * diff, axis=1))
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = n_vectors / avg_time / 1e6
    results["l2_distance_ms"] = avg_time * 1000
    results["l2_distance_throughput_M"] = throughput
    print(f"L2 Distance: {avg_time*1000:.3f}ms, {throughput:.2f}M vectors/sec")

    # Test 4: Top-K selection
    times = []
    k = 10
    for _ in range(iterations):
        start = time.perf_counter()
        scores = np.dot(vectors_normalized, query)
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = n_vectors / avg_time / 1e6
    results["top_k_ms"] = avg_time * 1000
    results["top_k_throughput_M"] = throughput
    print(f"Top-{k} Selection: {avg_time*1000:.3f}ms, {throughput:.2f}M vectors/sec")

    # Test 5: Batch matrix multiply
    batch_size = 32
    queries = np.random.randn(batch_size, dim).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        scores = np.dot(vectors_normalized, queries.T)  # (n_vectors, batch_size)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = (n_vectors * batch_size) / avg_time / 1e6
    results["batch_matmul_ms"] = avg_time * 1000
    results["batch_matmul_throughput_M"] = throughput
    print(
        f"Batch MatMul ({batch_size}x): {avg_time*1000:.3f}ms, {throughput:.2f}M ops/sec"
    )

    return results


def benchmark_mps_gpu(
    dim: int = 1024, n_vectors: int = 10000, iterations: int = 10
) -> dict[str, float] | None:
    """Benchmark Apple MPS GPU operations."""
    try:
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("\nMPS not available, skipping GPU benchmark")
            return None
    except ImportError:
        print("\nPyTorch not installed, skipping GPU benchmark")
        return None

    print(f"\n{'='*60}")
    print("APPLE MPS GPU BENCHMARK")
    print(f"Dimensions: {dim}, Vectors: {n_vectors}, Iterations: {iterations}")
    print(f"{'='*60}")

    device = torch.device("mps")

    # Generate test data
    torch.manual_seed(42)
    query = torch.randn(dim, device=device, dtype=torch.float32)
    vectors = torch.randn(n_vectors, dim, device=device, dtype=torch.float32)

    # Normalize
    query = query / torch.norm(query)
    vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)

    # Warmup
    _ = torch.matmul(vectors, query)
    torch.mps.synchronize()

    results = {}

    # Test 1: Matrix-vector multiply
    times = []
    for _ in range(iterations):
        torch.mps.synchronize()
        start = time.perf_counter()
        scores = torch.matmul(vectors, query)
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = n_vectors / avg_time / 1e6
    results["gpu_matvec_ms"] = avg_time * 1000
    results["gpu_matvec_throughput_M"] = throughput
    print(f"GPU MatVec: {avg_time*1000:.3f}ms, {throughput:.2f}M vectors/sec")

    # Test 2: Top-K on GPU
    times = []
    k = 10
    for _ in range(iterations):
        torch.mps.synchronize()
        start = time.perf_counter()
        scores = torch.matmul(vectors, query)
        top_k_values, top_k_idx = torch.topk(scores, k)
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = n_vectors / avg_time / 1e6
    results["gpu_top_k_ms"] = avg_time * 1000
    results["gpu_top_k_throughput_M"] = throughput
    print(f"GPU Top-{k}: {avg_time*1000:.3f}ms, {throughput:.2f}M vectors/sec")

    # Test 3: Batch queries on GPU
    batch_size = 32
    queries = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
    queries = queries / torch.norm(queries, dim=1, keepdim=True)

    # Warmup
    _ = torch.matmul(vectors, queries.T)
    torch.mps.synchronize()

    times = []
    for _ in range(iterations):
        torch.mps.synchronize()
        start = time.perf_counter()
        scores = torch.matmul(vectors, queries.T)
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = (n_vectors * batch_size) / avg_time / 1e6
    results["gpu_batch_ms"] = avg_time * 1000
    results["gpu_batch_throughput_M"] = throughput
    print(
        f"GPU Batch ({batch_size}x): {avg_time*1000:.3f}ms, {throughput:.2f}M ops/sec"
    )

    # Test 4: Float16 acceleration
    query_f16 = query.half()
    vectors_f16 = vectors.half()

    # Warmup
    _ = torch.matmul(vectors_f16, query_f16)
    torch.mps.synchronize()

    times = []
    for _ in range(iterations):
        torch.mps.synchronize()
        start = time.perf_counter()
        scores = torch.matmul(vectors_f16, query_f16)
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = n_vectors / avg_time / 1e6
    results["gpu_f16_ms"] = avg_time * 1000
    results["gpu_f16_throughput_M"] = throughput
    print(f"GPU Float16: {avg_time*1000:.3f}ms, {throughput:.2f}M vectors/sec")

    return results


def benchmark_mlx_ane(
    dim: int = 1024, n_vectors: int = 10000, iterations: int = 10
) -> dict[str, float] | None:
    """Benchmark MLX (Apple Neural Engine potential)."""
    try:
        import mlx.core as mx
    except ImportError:
        print("\nMLX not installed, skipping ANE benchmark")
        return None

    print(f"\n{'='*60}")
    print("MLX BENCHMARK (Metal/ANE)")
    print(f"Dimensions: {dim}, Vectors: {n_vectors}, Iterations: {iterations}")
    print(f"{'='*60}")

    # Generate test data
    mx.random.seed(42)
    query = mx.random.normal((dim,))
    vectors = mx.random.normal((n_vectors, dim))

    # Normalize
    query = query / mx.sqrt(mx.sum(query * query))
    norms = mx.sqrt(mx.sum(vectors * vectors, axis=1, keepdims=True))
    vectors = vectors / norms

    # Force evaluation
    mx.eval(query, vectors)

    results = {}

    # Test 1: Matrix-vector multiply
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        scores = mx.matmul(vectors, query)
        mx.eval(scores)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = n_vectors / avg_time / 1e6
    results["mlx_matvec_ms"] = avg_time * 1000
    results["mlx_matvec_throughput_M"] = throughput
    print(f"MLX MatVec: {avg_time*1000:.3f}ms, {throughput:.2f}M vectors/sec")

    # Test 2: Top-K
    times = []
    k = 10
    for _ in range(iterations):
        start = time.perf_counter()
        scores = mx.matmul(vectors, query)
        top_k_idx = mx.argpartition(scores, -k)[-k:]
        mx.eval(top_k_idx)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = n_vectors / avg_time / 1e6
    results["mlx_top_k_ms"] = avg_time * 1000
    results["mlx_top_k_throughput_M"] = throughput
    print(f"MLX Top-{k}: {avg_time*1000:.3f}ms, {throughput:.2f}M vectors/sec")

    # Test 3: Batch queries
    batch_size = 32
    queries = mx.random.normal((batch_size, dim))
    queries = queries / mx.sqrt(mx.sum(queries * queries, axis=1, keepdims=True))
    mx.eval(queries)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        scores = mx.matmul(vectors, queries.T)
        mx.eval(scores)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    throughput = (n_vectors * batch_size) / avg_time / 1e6
    results["mlx_batch_ms"] = avg_time * 1000
    results["mlx_batch_throughput_M"] = throughput
    print(
        f"MLX Batch ({batch_size}x): {avg_time*1000:.3f}ms, {throughput:.2f}M ops/sec"
    )

    return results


def benchmark_memory_operations(
    dim: int = 1024, n_vectors: int = 10000
) -> dict[str, float]:
    """Benchmark memory operations and zero-copy patterns."""
    print(f"\n{'='*60}")
    print("MEMORY OPERATIONS BENCHMARK")
    print(f"{'='*60}")

    results = {}

    # Test 1: Memory allocation speed
    times = []
    for _ in range(10):
        start = time.perf_counter()
        arr = np.empty((n_vectors, dim), dtype=np.float32)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    results["alloc_ms"] = avg_time * 1000
    print(f"Allocation ({n_vectors}x{dim} float32): {avg_time*1000:.3f}ms")

    # Test 2: Memory copy speed
    src = np.random.randn(n_vectors, dim).astype(np.float32)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        dst = np.copy(src)
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times[1:])
    mb_size = src.nbytes / 1024 / 1024
    bandwidth = mb_size / avg_time / 1024  # GB/s
    results["copy_ms"] = avg_time * 1000
    results["copy_bandwidth_GBs"] = bandwidth
    print(f"Memory Copy ({mb_size:.1f}MB): {avg_time*1000:.3f}ms, {bandwidth:.1f} GB/s")

    # Test 3: View (zero-copy) vs copy
    times_view = []
    times_copy = []
    for _ in range(100):
        # View (zero-copy)
        start = time.perf_counter()
        view = src[: n_vectors // 2]
        times_view.append(time.perf_counter() - start)

        # Copy
        start = time.perf_counter()
        copied = src[: n_vectors // 2].copy()
        times_copy.append(time.perf_counter() - start)

    avg_view = np.mean(times_view[10:]) * 1e6  # microseconds
    avg_copy = np.mean(times_copy[10:]) * 1e6
    speedup = avg_copy / avg_view
    results["view_us"] = avg_view
    results["copy_us"] = avg_copy
    results["zero_copy_speedup"] = speedup
    print(
        f"Zero-Copy View: {avg_view:.3f}μs vs Copy: {avg_copy:.1f}μs ({speedup:.0f}x faster)"
    )

    # Test 4: Contiguous vs strided access
    contiguous = np.ascontiguousarray(src)
    strided = src[::2]  # Every other row (strided)

    times_contig = []
    times_strided = []
    for _ in range(10):
        # Contiguous sum
        start = time.perf_counter()
        _ = np.sum(contiguous)
        times_contig.append(time.perf_counter() - start)

        # Strided sum (half the data)
        start = time.perf_counter()
        _ = np.sum(strided)
        times_strided.append(time.perf_counter() - start)

    avg_contig = np.mean(times_contig[1:])
    avg_strided = np.mean(times_strided[1:])
    # Normalize by data size
    normalized_strided = avg_strided * 2  # strided has half the data
    ratio = normalized_strided / avg_contig
    results["contiguous_ms"] = avg_contig * 1000
    results["strided_penalty"] = ratio
    print(f"Contiguous: {avg_contig*1000:.3f}ms, Strided penalty: {ratio:.2f}x slower")

    return results


def benchmark_quantization(dim: int = 1024, n_vectors: int = 10000) -> dict[str, float]:
    """Benchmark quantized operations."""
    print(f"\n{'='*60}")
    print("QUANTIZATION BENCHMARK")
    print(f"{'='*60}")

    results = {}

    # Generate float32 data
    vectors_f32 = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors_f32 = vectors_f32 / np.linalg.norm(vectors_f32, axis=1, keepdims=True)
    query_f32 = np.random.randn(dim).astype(np.float32)
    query_f32 = query_f32 / np.linalg.norm(query_f32)

    # Float16
    vectors_f16 = vectors_f32.astype(np.float16)
    query_f16 = query_f32.astype(np.float16)

    # Test float32 similarity
    times = []
    for _ in range(10):
        start = time.perf_counter()
        scores = np.dot(vectors_f32, query_f32)
        times.append(time.perf_counter() - start)
    avg_f32 = np.mean(times[1:])
    results["f32_ms"] = avg_f32 * 1000
    print(f"Float32: {avg_f32*1000:.3f}ms")

    # Test float16 similarity
    times = []
    for _ in range(10):
        start = time.perf_counter()
        scores = np.dot(vectors_f16.astype(np.float32), query_f16.astype(np.float32))
        times.append(time.perf_counter() - start)
    avg_f16 = np.mean(times[1:])
    results["f16_ms"] = avg_f16 * 1000
    print(f"Float16: {avg_f16*1000:.3f}ms")

    # Memory savings
    mem_f32 = vectors_f32.nbytes / 1024 / 1024
    mem_f16 = vectors_f16.nbytes / 1024 / 1024
    results["mem_savings_pct"] = (1 - mem_f16 / mem_f32) * 100
    print(
        f"Memory: {mem_f32:.1f}MB (f32) vs {mem_f16:.1f}MB (f16) = {results['mem_savings_pct']:.0f}% savings"
    )

    # Quality check
    scores_f32 = np.dot(vectors_f32, query_f32)
    scores_f16 = np.dot(vectors_f16.astype(np.float32), query_f16.astype(np.float32))
    top_10_f32 = np.argsort(scores_f32)[-10:][::-1]
    top_10_f16 = np.argsort(scores_f16)[-10:][::-1]
    overlap = len(set(top_10_f32) & set(top_10_f16))
    results["top10_overlap"] = overlap
    print(f"Top-10 overlap (quality): {overlap}/10")

    return results


def run_all_benchmarks():
    """Run complete benchmark suite."""
    print("\n" + "=" * 60)
    print(" HARDWARE OPTIMIZATION BENCHMARK SUITE")
    print("=" * 60)

    # Detect hardware
    hw_info = detect_hardware()
    print(f"\nPlatform: {hw_info['platform']} {hw_info['machine']}")
    print(f"Processor: {hw_info['processor']}")
    print(f"Python: {hw_info['python']}")
    print(f"NumPy: {hw_info['numpy']}")
    if hw_info.get("torch"):
        print(f"PyTorch: {hw_info['torch']}")
    if hw_info["gpu_available"]:
        print(f"GPU: {hw_info['gpu_type']}")
    if hw_info["mlx_available"]:
        print("MLX: Available")

    all_results = {"hardware": hw_info}

    # Run benchmarks
    all_results["numpy_simd"] = benchmark_numpy_simd()

    mps_results = benchmark_mps_gpu()
    if mps_results:
        all_results["mps_gpu"] = mps_results

    mlx_results = benchmark_mlx_ane()
    if mlx_results:
        all_results["mlx"] = mlx_results

    all_results["memory"] = benchmark_memory_operations()
    all_results["quantization"] = benchmark_quantization()

    # Summary
    print("\n" + "=" * 60)
    print(" BENCHMARK SUMMARY")
    print("=" * 60)

    print("\n[VECTOR SEARCH THROUGHPUT (Million vectors/sec)]")
    if "numpy_simd" in all_results:
        print(
            f"  NumPy SIMD Cosine: {all_results['numpy_simd']['cosine_sim_throughput_M']:.2f}M/s"
        )
    if "mps_gpu" in all_results:
        print(
            f"  MPS GPU MatVec:    {all_results['mps_gpu']['gpu_matvec_throughput_M']:.2f}M/s"
        )
        print(
            f"  MPS GPU Float16:   {all_results['mps_gpu']['gpu_f16_throughput_M']:.2f}M/s"
        )
    if "mlx" in all_results:
        print(
            f"  MLX MatVec:        {all_results['mlx']['mlx_matvec_throughput_M']:.2f}M/s"
        )

    print("\n[MEMORY PERFORMANCE]")
    if "memory" in all_results:
        print(
            f"  Copy Bandwidth:    {all_results['memory']['copy_bandwidth_GBs']:.1f} GB/s"
        )
        print(f"  Zero-Copy Speedup: {all_results['memory']['zero_copy_speedup']:.0f}x")

    print("\n[OPTIMIZATION RECOMMENDATIONS]")
    if hw_info["mps_available"]:
        print("  ✓ Use PyTorch MPS for batch similarity search")
    if hw_info["mlx_available"]:
        print("  ✓ Use MLX for ML inference (LLM, embeddings)")
    print("  ✓ Use NumPy with contiguous arrays for CPU operations")
    print("  ✓ Use memory views instead of copies where possible")
    print("  ✓ Consider float16 for memory-bound operations")

    return all_results


if __name__ == "__main__":
    results = run_all_benchmarks()
    print("\nBenchmark complete.")
