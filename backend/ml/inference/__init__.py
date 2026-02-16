"""
Inference Backends Module
==========================

Provides optimized inference backends:
- MLX (Apple Silicon native)
- CoreML (Neural Engine)
- MPS (Metal Performance Shaders)
- CUDA (NVIDIA)
- ONNX (CPU optimized)

Also provides:
- GPU coordination (locks, semaphores, executor) via gpu_coordination
- Model warm-up service for reduced first-inference latency
"""

# GPU coordination primitives (extracted to gpu_coordination.py)
from .coreml_backend import CoreMLEmbeddingEngine, get_coreml_embeddings
from .gpu_coordination import (
    MLX_SEMAPHORE_LIMIT,
    MPS_SEMAPHORE_LIMIT,
    cleanup_gpu_resources,
    get_gpu_executor,
    get_gpu_lock,
    get_gpu_semaphore,
    get_mlx_lock,
    get_mlx_semaphore,
    run_on_gpu_async,
    run_on_gpu_sync,
)
from .mlx_backend import MLXInferenceEngine, get_mlx_engine
from .unified_engine import (
    GenerationConfig,
    UnifiedInferenceEngine,
    get_inference_engine,
)
from .warmup import (
    ModelPriority,
    ModelSpec,
    ModelWarmupService,
    get_warmup_service,
    warmup_model,
)

# Aliases for backwards compatibility
CoreMLInferenceEngine = CoreMLEmbeddingEngine
WarmupService = ModelWarmupService

__all__ = [
    "CoreMLEmbeddingEngine",
    "CoreMLInferenceEngine",  # Alias
    "GenerationConfig",
    # Inference engines
    "MLXInferenceEngine",
    "ModelPriority",
    "ModelSpec",
    # Warm-up
    "ModelWarmupService",
    "UnifiedInferenceEngine",
    "WarmupService",  # Alias
    # GPU coordination
    "cleanup_gpu_resources",
    "get_coreml_embeddings",
    "get_gpu_executor",
    "get_gpu_lock",
    "get_gpu_semaphore",
    "get_inference_engine",
    "get_mlx_engine",
    "get_mlx_lock",
    "get_mlx_semaphore",
    "get_warmup_service",
    "run_on_gpu_async",
    "run_on_gpu_sync",
    "warmup_model",
]
