"""
Inference Backend Protocol — Hardware-Agnostic Model Interface
================================================================

Defines the contract that all inference backends must implement.
Backends handle the actual model loading, execution, and unloading
for a specific hardware target (MLX, CoreML, PyTorch/CUDA, ONNX, CPU).

Any new hardware support is added by implementing this protocol.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BackendType(StrEnum):
    """Supported inference backend types."""

    MLX = "mlx"
    COREML = "coreml"
    TORCH_MPS = "torch_mps"
    TORCH_CUDA = "torch_cuda"
    TORCH_CPU = "torch_cpu"
    ONNX = "onnx"
    VLLM = "vllm"

@dataclass
class BackendCapabilities:
    """What a backend can do."""

    backend_type: BackendType
    supports_streaming: bool = False
    supports_batching: bool = False
    supports_quantization: bool = False
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_int8: bool = False
    supports_int4: bool = False
    max_batch_size: int = 1
    max_sequence_length: int = 8192
    supported_model_types: set[str] = field(default_factory=set)
    device: str = "cpu"
    memory_budget_bytes: int = 0

@dataclass
class InferenceRequest:
    """Standardized inference request."""

    task: str  # generate, embed, translate, tts, stt, ocr, rerank
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    timeout_s: float = 30.0
    batch_id: str | None = None

@dataclass
class InferenceResult:
    """Standardized inference result."""

    output: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    device: str = ""
    batch_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

class InferenceBackend(ABC):
    """
    Abstract inference backend.

    Implementations handle model loading and inference for
    a specific hardware/framework combination.
    """

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return backend capabilities."""
        ...

    @abstractmethod
    async def load_model(self, model_name: str, **kwargs: Any) -> None:
        """Load a model into memory."""
        ...

    @abstractmethod
    async def unload_model(self, model_name: str) -> None:
        """Unload a model from memory."""
        ...

    @abstractmethod
    async def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        ...

    @abstractmethod
    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference on a single request."""
        ...

    async def infer_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """Run batched inference. Default: concurrent execution via asyncio.gather."""
        return list(await asyncio.gather(*(self.infer(req) for req in requests)))

    async def infer_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """Stream inference results. Default: yield full result."""
        result = await self.infer(request)
        yield str(result.output)

    async def warmup(self, model_name: str) -> None:  # noqa: B027
        """Warm up a model with a dummy inference pass."""
        pass

    async def health_check(self) -> bool:
        """Check if the backend is healthy."""
        return True

    async def shutdown(self) -> None:  # noqa: B027
        """Clean shutdown of the backend."""
        pass
