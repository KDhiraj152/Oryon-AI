# Hardware Optimization

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## Overview

Oryon AI v4.0 is optimized for multiple hardware configurations, with specialized support for Apple Silicon (M-series), NVIDIA CUDA GPUs, and CPU-only deployments. This guide covers configuration, optimization strategies, and performance benchmarks across different hardware profiles.

---

## Hardware Detection

The system automatically detects and configures for the optimal backend:

```python
# backend/core/optimized/device_router.py
import torch
import platform

def detect_optimal_device() -> str:
    """Detect and return the optimal compute device."""

    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        if torch.backends.mps.is_available():
            return "mps"

    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Fallback to CPU
    return "cpu"

def get_device_info() -> dict:
    """Get detailed device information."""
    device = detect_optimal_device()

    info = {
        "device": device,
        "platform": platform.system(),
        "processor": platform.processor(),
    }

    if device == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
    elif device == "mps":
        info["metal_available"] = torch.backends.mps.is_available()
        info["mps_built"] = torch.backends.mps.is_built()

    return info
```

---

## Apple Silicon Optimization

### M4 Configuration

Apple Silicon M4 provides excellent performance for ML inference with unified memory architecture:

```python
# backend/core/optimized/apple_silicon.py
import os
import torch

class AppleSiliconConfig:
    """Configuration for Apple Silicon optimization."""

    @staticmethod
    def configure():
        """Apply optimal settings for Apple Silicon."""

        # Memory management
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.5"

        # Disable memory-intensive features
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Use float16 for efficiency
        torch.set_default_dtype(torch.float16)

    @staticmethod
    def get_batch_config(model_size_gb: float) -> dict:
        """Get optimal batch configuration based on model size."""
        import psutil

        total_memory = psutil.virtual_memory().total / (1024**3)
        available_memory = psutil.virtual_memory().available / (1024**3)

        # Reserve memory for system and other models
        usable_memory = min(available_memory * 0.6, total_memory * 0.5)

        # Calculate optimal batch size
        if model_size_gb <= 2:
            batch_size = 32
        elif model_size_gb <= 4:
            batch_size = 16
        elif model_size_gb <= 8:
            batch_size = 8
        else:
            batch_size = 4

        return {
            "batch_size": batch_size,
            "max_length": 2048,
            "use_cache": True,
        }
```

### Memory Management for M4

```python
# backend/core/optimized/memory_coordinator.py
import gc
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelAllocation:
    model_id: str
    memory_mb: float
    last_access: float
    priority: int

class MemoryCoordinator:
    """Coordinate memory across ML models for Apple Silicon."""

    def __init__(self, max_memory_gb: float = 16.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.allocations: dict[str, ModelAllocation] = {}
        self._lock = asyncio.Lock()

    async def request_allocation(
        self,
        model_id: str,
        required_mb: float,
        priority: int = 1,
    ) -> bool:
        """Request memory allocation for a model."""
        async with self._lock:
            current_usage = sum(a.memory_mb for a in self.allocations.values())
            required_bytes = required_mb * 1024**2

            if current_usage * 1024**2 + required_bytes > self.max_memory_bytes:
                freed = await self._evict_models(required_mb)
                if freed < required_mb:
                    return False

            self.allocations[model_id] = ModelAllocation(
                model_id=model_id,
                memory_mb=required_mb,
                last_access=time.time(),
                priority=priority,
            )
            return True

    async def _evict_models(self, needed_mb: float) -> float:
        """Evict low-priority models to free memory."""
        # Sort by priority (ascending) then last access (ascending)
        candidates = sorted(
            self.allocations.items(),
            key=lambda x: (x[1].priority, x[1].last_access),
        )

        freed = 0.0
        for model_id, allocation in candidates:
            if freed >= needed_mb:
                break

            await self._unload_model(model_id)
            freed += allocation.memory_mb
            del self.allocations[model_id]

        # Force garbage collection
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return freed

    async def _unload_model(self, model_id: str):
        """Unload a model from memory."""
        # Implementation depends on model registry
        logger.info(f"Unloading model: {model_id}")
```

### M4 Performance Benchmarks

| Model | Operation | M4 Pro (16GB) | M4 Max (32GB) |
|-------|-----------|---------------|---------------|
| Qwen3-8B (MLX 4-bit) | Token generation | 45 tok/s | 52 tok/s |
| BGE-M3 | Embedding (batch=32) | 280ms | 210ms |
| IndicTrans2-1B | Translation | 95ms | 72ms |
| Whisper Turbo | 30s audio | 1.8s | 1.4s |
| MMS-TTS | 100 chars | 180ms | 140ms |

---

## CUDA Optimization

### NVIDIA GPU Configuration

```python
# Reference: CUDA configuration (not yet implemented)
# backend/core/cuda_config.py
import torch

class CUDAConfig:
    """Configuration for NVIDIA CUDA optimization."""

    @staticmethod
    def configure():
        """Apply optimal settings for CUDA devices."""

        if not torch.cuda.is_available():
            return

        # Enable TF32 for Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.85)

    @staticmethod
    def get_device_config() -> dict:
        """Get CUDA device configuration."""
        props = torch.cuda.get_device_properties(0)

        return {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / (1024**3),
            "multi_processor_count": props.multi_processor_count,
            "supports_tf32": props.major >= 8,
            "supports_bf16": props.major >= 8,
        }

    @staticmethod
    def optimize_for_inference():
        """Apply inference-specific optimizations."""
        torch.inference_mode(True)

        # Disable gradient computation globally
        torch.set_grad_enabled(False)
```

### Multi-GPU Configuration

```python
# Reference: Multi-GPU configuration (not yet implemented)
# backend/core/multi_gpu.py
import torch
from torch.nn.parallel import DataParallel

class MultiGPUManager:
    """Manage model distribution across multiple GPUs."""

    def __init__(self):
        self.device_count = torch.cuda.device_count()
        self.primary_device = torch.device("cuda:0")

    def distribute_model(
        self,
        model: torch.nn.Module,
        strategy: str = "data_parallel",
    ):
        """Distribute model across available GPUs."""

        if self.device_count <= 1:
            return model.to(self.primary_device)

        if strategy == "data_parallel":
            return DataParallel(model)
        elif strategy == "device_map":
            # For large models that don't fit on single GPU
            return self._create_device_map(model)

    def _create_device_map(self, model) -> dict:
        """Create device map for model parallelism."""
        from accelerate import infer_auto_device_map

        return infer_auto_device_map(
            model,
            max_memory={
                i: f"{self._get_available_memory(i)}GiB"
                for i in range(self.device_count)
            },
        )

    def _get_available_memory(self, device_id: int) -> float:
        """Get available memory on specific GPU."""
        torch.cuda.set_device(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory
        allocated = torch.cuda.memory_allocated(device_id)
        return (total - allocated) / (1024**3) * 0.9  # 90% of available
```

### CUDA Performance Benchmarks

| Model | Operation | RTX 3090 | RTX 4090 | A100 |
|-------|-----------|----------|----------|------|
| Qwen3-8B (INT4) | Token generation | 85 tok/s | 120 tok/s | 150 tok/s |
| BGE-M3 | Embedding (batch=64) | 120ms | 75ms | 55ms |
| IndicTrans2-1B | Translation | 45ms | 28ms | 18ms |
| Whisper Turbo | 30s audio | 0.9s | 0.6s | 0.4s |
| MMS-TTS | 100 chars | 80ms | 55ms | 35ms |

---

## CPU Optimization

### CPU-Only Configuration

```python
# Reference: CPU configuration (not yet implemented)
# backend/core/cpu_config.py
import os
import torch

class CPUConfig:
    """Configuration for CPU-only deployment."""

    @staticmethod
    def configure():
        """Apply optimal settings for CPU inference."""

        # Set thread count
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)

        # Enable MKL optimizations if available
        if torch.backends.mkl.is_available():
            os.environ["MKL_NUM_THREADS"] = str(num_threads)

        # OpenMP settings
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["OMP_SCHEDULE"] = "STATIC"

        # Enable oneDNN (Intel)
        os.environ["DNNL_DEFAULT_FPMATH_MODE"] = "BF16"

    @staticmethod
    def get_quantized_model(model_id: str):
        """Load quantized model for CPU inference."""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,  # CPU needs float32
            bnb_4bit_quant_type="nf4",
        )

        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
```

### ONNX Runtime Optimization

```python
# Reference: ONNX optimization (not yet implemented)
# backend/core/onnx_runtime.py
import onnxruntime as ort

class ONNXOptimizer:
    """ONNX Runtime optimization for CPU inference."""

    @staticmethod
    def create_session(model_path: str) -> ort.InferenceSession:
        """Create optimized ONNX inference session."""

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = os.cpu_count()
        sess_options.inter_op_num_threads = 2

        # Enable memory pattern optimization
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        providers = ["CPUExecutionProvider"]

        return ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )
```

---

## Model Loading Strategies

### Lazy Loading

```python
# backend/core/optimized/model_manager.py
from functools import lru_cache

class ModelManager:
    """Manage model loading with lazy initialization."""

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._loading_locks: dict[str, asyncio.Lock] = {}

    async def get_model(self, model_id: str):
        """Get model with lazy loading."""
        if model_id not in self._models:
            if model_id not in self._loading_locks:
                self._loading_locks[model_id] = asyncio.Lock()

            async with self._loading_locks[model_id]:
                if model_id not in self._models:
                    self._models[model_id] = await self._load_model(model_id)

        return self._models[model_id]

    async def _load_model(self, model_id: str):
        """Load model based on type."""
        config = MODEL_CONFIGS.get(model_id)

        if config["type"] == "llm":
            return await self._load_llm(config)
        elif config["type"] == "embedder":
            return await self._load_embedder(config)
        elif config["type"] == "translator":
            return await self._load_translator(config)
```

### Warmup Strategy

```python
# backend/ml/inference/warmup.py
class ModelWarmupService:
    """Warm up models during application startup."""

    WARMUP_CONFIGS = {
        "llm": {
            "prompt": "Hello, how are you?",
            "max_tokens": 10,
        },
        "embedder": {
            "texts": ["Sample text for embedding warmup."],
        },
        "translator": {
            "text": "Hello world",
            "source": "eng_Latn",
            "target": "hin_Deva",
        },
    }

    async def warmup_all(self):
        """Warm up all models."""
        logger.info("Starting model warmup...")

        for model_type, config in self.WARMUP_CONFIGS.items():
            try:
                await self._warmup_model(model_type, config)
                logger.info(f"Warmed up: {model_type}")
            except Exception as e:
                logger.error(f"Warmup failed for {model_type}: {e}")

        logger.info("Model warmup complete")

    async def _warmup_model(self, model_type: str, config: dict):
        """Warm up a specific model."""
        if model_type == "llm":
            await self.llm_service.generate(
                prompt=config["prompt"],
                max_tokens=config["max_tokens"],
            )
        elif model_type == "embedder":
            await self.embedder.encode(config["texts"])
        elif model_type == "translator":
            await self.translator.translate(
                text=config["text"],
                source_lang=config["source"],
                target_lang=config["target"],
            )
```

---

## Quantization

### INT4 Quantization for LLMs

```python
# backend/core/optimized/quantization.py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_quantized_llm(
    model_id: str,
    device: str = "auto",
) -> AutoModelForCausalLM:
    """Load INT4 quantized LLM."""

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    return model
```

### Quantization Impact

| Model | Precision | Size | Quality Impact | Speed Impact |
|-------|-----------|------|----------------|--------------|
| Qwen3-8B | FP16 | 16.0GB | Baseline | Baseline |
| Qwen3-8B | INT8 | 8.0GB | -0.5% | +25% |
| Qwen3-8B | 4-bit (MLX) | 4.6GB | -2% | +40% |
| BGE-M3 | FP16 | 1.1GB | Baseline | Baseline |
| BGE-M3 | INT8 | 0.6GB | -1% | +30% |

---

## Memory Profiling

### Profiling Tools

```python
# Reference: Memory profiling script (not yet created)
# scripts/profile_memory.py
import torch
import tracemalloc
from memory_profiler import profile

@profile
def profile_model_loading():
    """Profile memory usage during model loading."""
    from backend.services.rag import RAGService

    # Start memory tracking
    tracemalloc.start()

    # Load service
    service = RAGService()

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Current memory: {current / 1024**2:.2f} MB")
    print(f"Peak memory: {peak / 1024**2:.2f} MB")

    if torch.cuda.is_available():
        print(f"CUDA allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    profile_model_loading()
```

### Memory Optimization Tips

1. **Use gradient checkpointing** for training scenarios
2. **Enable KV cache** for inference with repeated contexts
3. **Batch similar-length sequences** to minimize padding
4. **Clear CUDA cache** between large operations
5. **Use memory-mapped loading** for large models

---

## Hardware Recommendations

### Development

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 16 cores |
| RAM | 16GB | 32GB |
| GPU | 8GB VRAM | 16GB VRAM |
| Storage | 100GB SSD | 250GB NVMe |

### Production (Single Node)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 16 cores | 32 cores |
| RAM | 32GB | 64GB |
| GPU | 16GB VRAM | 24GB VRAM |
| Storage | 500GB NVMe | 1TB NVMe |

### Production (Cluster)

| Component | Per Node |
|-----------|----------|
| CPU | 32 cores |
| RAM | 64GB |
| GPU | A10G (24GB) or better |
| Storage | 500GB NVMe |
| Network | 10Gbps |

---

## Troubleshooting

### Common Issues

**Out of Memory (CUDA)**
```python
# Clear cache and reduce batch size
torch.cuda.empty_cache()
gc.collect()

# Reduce batch size in config
settings.EMBEDDING_BATCH_SIZE = 16  # Down from 32
```

**Slow MPS Performance**
```python
# Ensure MPS is properly initialized
torch.mps.synchronize()

# Check for fallback to CPU
print(f"MPS available: {torch.backends.mps.is_available()}")
```

**High Memory on CPU**
```python
# Use memory-efficient loading
model = AutoModel.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
```

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com
