"""
Precision Manager — Mixed Precision Inference Control
=========================================================

Manages dtype selection for model inference:
  - Auto-detection based on hardware capabilities
  - Per-model precision overrides
  - Dynamic precision switching under memory pressure
  - Quantization support (int8, int4)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from backend.infra.telemetry import get_logger

logger = get_logger(__name__)

class PrecisionMode(StrEnum):
    """Supported precision modes."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"  # FP16 compute, FP32 accumulation
    AUTO = "auto"    # Auto-select based on hardware

@dataclass
class PrecisionConfig:
    """Per-model precision configuration."""

    model_name: str
    compute_dtype: PrecisionMode = PrecisionMode.AUTO
    storage_dtype: PrecisionMode = PrecisionMode.AUTO
    accumulation_dtype: PrecisionMode = PrecisionMode.FP32
    quantization_bits: int | None = None

class PrecisionManager:
    """
    Manages precision modes across all models.

    Automatically selects optimal precision based on:
      - Device capabilities (fp16, bf16, int8 support)
      - Model requirements
      - Available memory
    """

    def __init__(self) -> None:
        self._configs: dict[str, PrecisionConfig] = {}
        self._device_fp16: bool = False
        self._device_bf16: bool = False
        self._device_int8: bool = False
        self._device_int4: bool = False
        self._lock = threading.Lock()

    def configure_device(
        self,
        *,
        supports_fp16: bool = False,
        supports_bf16: bool = False,
        supports_int8: bool = False,
        supports_int4: bool = False,
    ) -> None:
        """Configure device precision capabilities."""
        self._device_fp16 = supports_fp16
        self._device_bf16 = supports_bf16
        self._device_int8 = supports_int8
        self._device_int4 = supports_int4
        logger.info(
            "precision_configured",
            fp16=supports_fp16,
            bf16=supports_bf16,
            int8=supports_int8,
            int4=supports_int4,
        )

    def set_model_precision(self, config: PrecisionConfig) -> None:
        """Set precision configuration for a specific model."""
        with self._lock:
            self._configs[config.model_name] = config

    def get_compute_dtype(self, model_name: str) -> PrecisionMode:
        """Get the compute dtype for a model."""
        config = self._configs.get(model_name)
        if config and config.compute_dtype != PrecisionMode.AUTO:
            return config.compute_dtype
        return self._auto_select_compute()

    def get_storage_dtype(self, model_name: str) -> PrecisionMode:
        """Get the storage dtype for a model (weights)."""
        config = self._configs.get(model_name)
        if config and config.storage_dtype != PrecisionMode.AUTO:
            return config.storage_dtype
        return self._auto_select_storage()

    def get_torch_dtype(self, model_name: str) -> Any:
        """Get the torch dtype for model loading."""
        try:
            import torch
        except ImportError:
            return None

        mode = self.get_compute_dtype(model_name)
        dtype_map = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.MIXED: torch.float16,
        }
        return dtype_map.get(mode, torch.float32)

    def should_quantize(self, model_name: str) -> tuple[bool, int]:
        """Check if a model should be quantized, and to how many bits."""
        config = self._configs.get(model_name)
        if config and config.quantization_bits:
            return True, config.quantization_bits
        return False, 0

    def downgrade_precision(self, model_name: str) -> PrecisionMode:
        """
        Downgrade precision under memory pressure.

        fp32 → fp16 → int8 → int4
        """
        current = self.get_compute_dtype(model_name)
        downgrade_chain = {
            PrecisionMode.FP32: PrecisionMode.FP16,
            PrecisionMode.BF16: PrecisionMode.FP16,
            PrecisionMode.FP16: PrecisionMode.INT8,
            PrecisionMode.INT8: PrecisionMode.INT4,
        }

        new_mode = downgrade_chain.get(current, current)
        if new_mode != current:
            config = self._configs.get(model_name) or PrecisionConfig(model_name=model_name)
            config.compute_dtype = new_mode
            self._configs[model_name] = config
            logger.info(
                "precision_downgraded",
                model=model_name,
                from_mode=current.value,
                to_mode=new_mode.value,
            )
        return new_mode

    def _auto_select_compute(self) -> PrecisionMode:
        """Auto-select compute precision."""
        if self._device_bf16:
            return PrecisionMode.BF16
        if self._device_fp16:
            return PrecisionMode.FP16
        return PrecisionMode.FP32

    def _auto_select_storage(self) -> PrecisionMode:
        """Auto-select storage precision."""
        if self._device_int8:
            return PrecisionMode.INT8
        if self._device_fp16:
            return PrecisionMode.FP16
        return PrecisionMode.FP32

    def get_summary(self) -> dict[str, Any]:
        return {
            "device_capabilities": {
                "fp16": self._device_fp16,
                "bf16": self._device_bf16,
                "int8": self._device_int8,
                "int4": self._device_int4,
            },
            "models": {
                name: {
                    "compute": cfg.compute_dtype.value,
                    "storage": cfg.storage_dtype.value,
                    "quantization_bits": cfg.quantization_bits,
                }
                for name, cfg in self._configs.items()
            },
        }

# ── Singleton ──────────────────────────────────────────────────────

_manager: PrecisionManager | None = None

def get_precision_manager() -> PrecisionManager:
    # Lock-free benign-race singleton.
    # Avoids threading.Lock which blocks the event loop in async context.
    global _manager
    if _manager is not None:
        return _manager
    _manager = PrecisionManager()
    return _manager