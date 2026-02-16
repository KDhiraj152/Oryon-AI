"""
Quantization Strategy - Platform-Aware Model Optimization
==========================================================

Provides optimal quantization config based on hardware:
- Apple Silicon (M4): FP16 with MLX (unified memory)
- CUDA: INT4/AWQ for memory efficiency
- CPU: INT8/ONNX for speed

Key insight: M4's unified memory makes FP16 practical,
while CUDA benefits from aggressive quantization.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

class QuantizationType(StrEnum):
    """Available quantization types."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    AWQ = "awq"  # Activation-aware Weight Quantization
    GPTQ = "gptq"  # GPT Quantization
    GGUF = "gguf"  # llama.cpp format

class QuantizationBackend(StrEnum):
    """Quantization backends."""

    NATIVE = "native"  # PyTorch native
    MLX = "mlx"  # Apple MLX
    BITSANDBYTES = "bitsandbytes"  # For CUDA
    ONNX = "onnx"  # ONNX Runtime
    COREML = "coreml"  # Apple Core ML
    LLAMA_CPP = "llama_cpp"  # llama.cpp

@dataclass
class QuantConfig:
    """Quantization configuration."""

    dtype: QuantizationType = QuantizationType.FP16
    backend: QuantizationBackend = QuantizationBackend.NATIVE

    # Model loading options
    use_kv_cache: bool = True
    kv_cache_dtype: QuantizationType | None = None
    max_kv_size: int = 8192  # M4 can handle large context

    # Memory optimization
    use_flash_attention: bool = False
    use_sdpa: bool = True  # Scaled Dot Product Attention
    gradient_checkpointing: bool = False

    # Generation optimization
    use_speculative_decoding: bool = False
    speculative_draft_model: str | None = None

    # CUDA specific
    use_tensor_cores: bool = True
    allow_tf32: bool = True

    # Apple Silicon specific
    use_metal_performance_shaders: bool = True
    use_neural_engine: bool = False
    use_fast_math: bool = True  # MLX fast math
    use_channels_last: bool = True  # Memory layout optimization
    num_threads: int = 4  # Performance cores

    # ONNX specific
    use_dynamic_quantization: bool = False
    optimize_for_inference: bool = True

    # Memory limits
    max_memory_gb: float | None = None
    offload_to_cpu: bool = False

    # M4 optimal settings
    compute_units: str = "ALL"  # ALL, CPU_ONLY, CPU_AND_NE, CPU_AND_GPU

    # Metadata
    model_id: str | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)

    def to_transformers_kwargs(self) -> dict[str, Any]:
        """Convert to HuggingFace transformers kwargs."""
        import torch

        kwargs = {
            "use_cache": self.use_kv_cache,
        }

        # Dtype mapping
        dtype_map = {
            QuantizationType.FP32: torch.float32,
            QuantizationType.FP16: torch.float16,
            QuantizationType.BF16: torch.bfloat16,
        }

        if self.dtype in dtype_map:
            kwargs["torch_dtype"] = dtype_map[self.dtype]

        # Attention implementation
        if self.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        elif self.use_sdpa:
            kwargs["attn_implementation"] = "sdpa"

        # Low memory options
        if self.offload_to_cpu:
            kwargs["device_map"] = "auto"
            kwargs["offload_folder"] = "offload"

        # M4 optimization: set threads for performance cores
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)

        return kwargs

    def to_mlx_kwargs(self) -> dict[str, Any]:
        """Convert to MLX load kwargs."""
        return {
            "tokenizer_config": {},
            "model_config": {
                **self.extra_config,
                "use_fast_math": self.use_fast_math,
                "max_kv_size": self.max_kv_size,
            },
        }

    def to_coreml_kwargs(self) -> dict[str, Any]:
        """Convert to CoreML settings."""
        return {
            "compute_units": self.compute_units,
            "precision": "float16"
            if self.dtype == QuantizationType.FP16
            else "float32",
        }

class QuantizationStrategy:
    """
    Platform-aware quantization strategy selector.

    Selects optimal quantization based on:
    - Available hardware (Apple Silicon, CUDA, CPU)
    - Model size
    - Memory constraints
    - Task requirements (accuracy vs speed)
    """

    # Model size thresholds (in billions of parameters)
    SMALL_MODEL_THRESHOLD = 3.0  # <3B can run FP16 on most devices
    MEDIUM_MODEL_THRESHOLD = 7.0  # 3-7B may need quantization
    LARGE_MODEL_THRESHOLD = 14.0  # >7B needs aggressive quantization

    # Memory requirements per billion parameters (GB)
    MEMORY_PER_BILLION: ClassVar[dict] = {
        QuantizationType.FP32: 4.0,
        QuantizationType.FP16: 2.0,
        QuantizationType.BF16: 2.0,
        QuantizationType.INT8: 1.0,
        QuantizationType.INT4: 0.5,
        QuantizationType.AWQ: 0.5,
        QuantizationType.GPTQ: 0.5,
    }

    @classmethod
    def get_optimal_config(
        cls,
        model_size_b: float,
        device_router,  # DeviceRouter instance
        accuracy_priority: bool = True,
        max_memory_gb: float | None = None,
    ) -> QuantConfig:
        """
        Get optimal quantization config for given constraints.

        Args:
            model_size_b: Model size in billions of parameters
            device_router: DeviceRouter instance for capability detection
            accuracy_priority: Prioritize accuracy over speed
            max_memory_gb: Maximum memory to use (None = auto)

        Returns:
            QuantConfig with optimal settings
        """
        caps = device_router.capabilities

        # Auto-detect memory limit
        if max_memory_gb is None:
            max_memory_gb = caps.memory_gb * 0.8  # Leave 20% headroom

        # Apple Silicon (M1/M2/M3/M4)
        if caps.is_apple_silicon:
            return cls._get_apple_silicon_config(
                model_size_b, caps, accuracy_priority, max_memory_gb
            )

        # NVIDIA CUDA
        if caps.has_cuda:
            return cls._get_cuda_config(model_size_b, accuracy_priority, max_memory_gb)

        # CPU fallback
        return cls._get_cpu_config(model_size_b, accuracy_priority)

    @classmethod
    def _get_apple_silicon_config(
        cls,
        model_size_b: float,
        caps,
        accuracy_priority: bool,
        max_memory_gb: float,
    ) -> QuantConfig:
        """Get optimal config for Apple Silicon."""

        # M4 with 16GB+ can handle 7B models in FP16
        memory_for_fp16 = model_size_b * cls.MEMORY_PER_BILLION[QuantizationType.FP16]

        # M4-specific settings - use 4 performance cores
        num_threads = 4  # Optimal for both M4 and other Apple Silicon
        use_fast_math = caps.is_m4  # MLX fast math on M4
        max_kv = 8192 if caps.memory_gb >= 16 else 4096

        # Use MLX if available (fastest on Apple Silicon)
        if caps.mlx_available:
            if memory_for_fp16 <= max_memory_gb:
                # FP16 with MLX is fastest and most accurate
                return QuantConfig(
                    dtype=QuantizationType.FP16,
                    backend=QuantizationBackend.MLX,
                    use_kv_cache=True,
                    max_kv_size=max_kv,
                    use_speculative_decoding=model_size_b >= cls.MEDIUM_MODEL_THRESHOLD,
                    use_metal_performance_shaders=True,
                    use_fast_math=use_fast_math,
                    num_threads=num_threads,
                    max_memory_gb=max_memory_gb,
                )
            else:
                # INT4 with MLX for larger models
                return QuantConfig(
                    dtype=QuantizationType.INT4,
                    backend=QuantizationBackend.MLX,
                    use_kv_cache=True,
                    max_kv_size=max_kv,
                    use_metal_performance_shaders=True,
                    use_fast_math=use_fast_math,
                    num_threads=num_threads,
                    max_memory_gb=max_memory_gb,
                )

        # Fallback to MPS (PyTorch)
        if memory_for_fp16 <= max_memory_gb:
            return QuantConfig(
                dtype=QuantizationType.FP16,
                backend=QuantizationBackend.NATIVE,
                use_kv_cache=True,
                use_sdpa=True,
                use_metal_performance_shaders=True,
                use_channels_last=True,
                num_threads=num_threads,
                max_memory_gb=max_memory_gb,
            )

        # Large models on limited memory - use GGUF
        return QuantConfig(
            dtype=QuantizationType.INT4,
            backend=QuantizationBackend.LLAMA_CPP,
            use_kv_cache=True,
            max_memory_gb=max_memory_gb,
        )

    @classmethod
    def _get_cuda_config(
        cls,
        model_size_b: float,
        accuracy_priority: bool,
        max_memory_gb: float,
    ) -> QuantConfig:
        """Get optimal config for CUDA."""

        memory_for_fp16 = model_size_b * cls.MEMORY_PER_BILLION[QuantizationType.FP16]

        if memory_for_fp16 <= max_memory_gb and accuracy_priority:
            # FP16 with Flash Attention
            return QuantConfig(
                dtype=QuantizationType.FP16,
                backend=QuantizationBackend.NATIVE,
                use_kv_cache=True,
                use_flash_attention=True,
                use_tensor_cores=True,
                allow_tf32=True,
                max_memory_gb=max_memory_gb,
            )

        # AWQ is better than GPTQ for accuracy
        if accuracy_priority:
            return QuantConfig(
                dtype=QuantizationType.AWQ,
                backend=QuantizationBackend.BITSANDBYTES,
                use_kv_cache=True,
                use_flash_attention=True,
                use_tensor_cores=True,
                max_memory_gb=max_memory_gb,
            )

        # INT4 for maximum speed
        return QuantConfig(
            dtype=QuantizationType.INT4,
            backend=QuantizationBackend.BITSANDBYTES,
            use_kv_cache=True,
            use_flash_attention=True,
            max_memory_gb=max_memory_gb,
        )

    @classmethod
    def _get_cpu_config(
        cls,
        model_size_b: float,
        accuracy_priority: bool,
    ) -> QuantConfig:
        """Get optimal config for CPU."""

        if model_size_b < cls.SMALL_MODEL_THRESHOLD and accuracy_priority:
            # Small models can run FP32 on CPU
            return QuantConfig(
                dtype=QuantizationType.FP32,
                backend=QuantizationBackend.ONNX,
                use_kv_cache=True,
                optimize_for_inference=True,
            )

        # INT8 ONNX for larger models
        return QuantConfig(
            dtype=QuantizationType.INT8,
            backend=QuantizationBackend.ONNX,
            use_kv_cache=True,
            use_dynamic_quantization=True,
            optimize_for_inference=True,
        )

    @classmethod
    def estimate_memory(
        cls,
        model_size_b: float,
        config: QuantConfig,
    ) -> dict[str, float]:
        """Estimate memory usage for a configuration."""
        base_memory = model_size_b * cls.MEMORY_PER_BILLION.get(config.dtype, 2.0)

        # KV cache overhead (~10-20% depending on context length)
        kv_overhead = 0.15 if config.use_kv_cache else 0.0

        # Activation memory during inference
        activation_memory = base_memory * 0.1

        total = base_memory * (1 + kv_overhead) + activation_memory

        return {
            "model_memory_gb": base_memory,
            "kv_cache_gb": base_memory * kv_overhead,
            "activation_gb": activation_memory,
            "total_gb": total,
            "recommended_system_gb": total * 1.3,  # 30% headroom
        }
