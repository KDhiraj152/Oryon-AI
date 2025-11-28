"""
Dynamic Quantization Manager - Adaptive Model Quantization

Automatically adjusts quantization levels based on:
- Available system memory
- Current memory usage
- Node load (number of concurrent requests)
- Device capabilities (MPS/CUDA/CPU)

Supports variable quantization: FP16, INT8, INT4, INT2 (GGUF)
"""
import logging
import psutil
import torch
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


class QuantizationLevel(str, Enum):
    """Quantization levels from highest to lowest precision."""
    FP16 = "fp16"        # Half precision (best quality, 2x compression)
    INT8 = "int8"        # 8-bit quantization (4x compression)
    INT4 = "int4"        # 4-bit quantization (8x compression)
    INT2 = "int2"        # 2-bit quantization (16x compression, GGUF only)
    DYNAMIC = "dynamic"  # Adaptive based on context


@dataclass
class MemoryMetrics:
    """System memory metrics."""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    process_memory_gb: float


@dataclass
class QuantizationConfig:
    """Configuration for dynamic quantization."""
    level: QuantizationLevel
    precision: str  # "fp16", "int8", "int4", "int2"
    compression_ratio: float
    estimated_memory_gb: float
    config: Dict[str, Any]


class DynamicQuantizationManager:
    """
    Manages adaptive quantization based on system resources.
    
    Strategy:
    1. Monitor system memory and load
    2. Calculate optimal quantization level
    3. Adjust on-demand as load changes
    4. Prefer higher precision when resources available
    5. Degrade gracefully under pressure
    """
    
    # Memory thresholds (percentage of available memory)
    MEMORY_COMFORTABLE = 0.4   # <40% used: FP16 ok
    MEMORY_MODERATE = 0.6      # 40-60% used: INT8 recommended
    MEMORY_TIGHT = 0.75        # 60-75% used: INT4 required
    MEMORY_CRITICAL = 0.85     # >85% used: INT2 or offload
    
    # Model size estimates (7B parameter model)
    MODEL_SIZES_GB = {
        QuantizationLevel.FP16: 14.0,   # 7B * 2 bytes
        QuantizationLevel.INT8: 7.0,    # 7B * 1 byte
        QuantizationLevel.INT4: 3.5,    # 7B * 0.5 bytes
        QuantizationLevel.INT2: 1.75,   # 7B * 0.25 bytes
    }
    
    # Load thresholds (concurrent requests)
    LOAD_LOW = 2      # <2 requests: higher precision ok
    LOAD_MODERATE = 5 # 2-5 requests: balanced quantization
    LOAD_HIGH = 10    # >10 requests: aggressive quantization
    
    def __init__(self, device: str = "auto"):
        """
        Initialize dynamic quantization manager.
        
        Args:
            device: Target device ("mps", "cuda", "cpu", "auto")
        """
        self.device = self._detect_device(device)
        self.active_requests = 0
        self.request_lock = threading.Lock()
        
        # Device capabilities
        self.supports_fp16 = self._check_fp16_support()
        self.supports_int8 = self._check_int8_support()
        self.supports_int4 = self._check_int4_support()
        
        logger.info(
            f"DynamicQuantizationManager initialized: device={self.device}, "
            f"FP16={self.supports_fp16}, INT8={self.supports_int8}, INT4={self.supports_int4}"
        )
    
    def _detect_device(self, device: str) -> str:
        """Detect optimal device."""
        if device != "auto":
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _check_fp16_support(self) -> bool:
        """Check if device supports FP16."""
        if self.device == "cuda":
            return torch.cuda.is_available()
        elif self.device == "mps":
            return True  # MPS supports FP16
        else:
            return False  # CPU FP16 is slow
    
    def _check_int8_support(self) -> bool:
        """Check if device supports INT8 quantization."""
        try:
            import bitsandbytes
            return self.device == "cuda"  # BitsAndBytes requires CUDA
        except ImportError:
            return False
    
    def _check_int4_support(self) -> bool:
        """Check if device supports INT4 quantization."""
        try:
            import llama_cpp
            return True  # llama-cpp supports all devices
        except ImportError:
            try:
                import bitsandbytes
                return self.device == "cuda"
            except ImportError:
                return False
    
    def get_memory_metrics(self) -> MemoryMetrics:
        """Get current system memory metrics."""
        vm = psutil.virtual_memory()
        process = psutil.Process()
        process_mem = process.memory_info().rss / (1024**3)
        
        return MemoryMetrics(
            total_gb=vm.total / (1024**3),
            available_gb=vm.available / (1024**3),
            used_gb=vm.used / (1024**3),
            percent_used=vm.percent / 100.0,
            process_memory_gb=process_mem
        )
    
    def calculate_optimal_quantization(
        self,
        model_size_params: float,
        task_priority: str = "balanced"
    ) -> QuantizationConfig:
        """
        Calculate optimal quantization level based on current resources.
        
        Args:
            model_size_params: Model size in billions of parameters
            task_priority: "quality" (prefer precision), "balanced", "speed" (aggressive compression)
        
        Returns:
            QuantizationConfig with optimal settings
        """
        metrics = self.get_memory_metrics()
        
        # Get current load
        current_load = self.active_requests
        
        logger.debug(
            f"Calculating quantization: memory={metrics.percent_used:.1%}, "
            f"available={metrics.available_gb:.1f}GB, load={current_load} requests"
        )
        
        # Decision matrix based on memory pressure and load
        if metrics.percent_used < self.MEMORY_COMFORTABLE and current_load < self.LOAD_LOW:
            # Plenty of resources - use best quality
            if self.supports_fp16 and task_priority in ["quality", "balanced"]:
                level = QuantizationLevel.FP16
            elif self.supports_int8:
                level = QuantizationLevel.INT8
            else:
                level = QuantizationLevel.INT4
        
        elif metrics.percent_used < self.MEMORY_MODERATE and current_load < self.LOAD_MODERATE:
            # Moderate resources - balance quality and memory
            if self.supports_int8 and task_priority == "quality":
                level = QuantizationLevel.INT8
            else:
                level = QuantizationLevel.INT4
        
        elif metrics.percent_used < self.MEMORY_TIGHT:
            # Tight resources - prioritize memory
            level = QuantizationLevel.INT4
        
        else:
            # Critical resources - maximum compression
            if self.supports_int4:
                level = QuantizationLevel.INT2 if current_load > self.LOAD_HIGH else QuantizationLevel.INT4
            else:
                level = QuantizationLevel.INT4
        
        # Calculate estimated memory
        base_size_gb = self.MODEL_SIZES_GB.get(level, 3.5) * (model_size_params / 7.0)
        
        # Check if we have enough memory
        if base_size_gb > metrics.available_gb * 0.8:
            # Not enough memory - degrade to next level
            logger.warning(
                f"Insufficient memory for {level.value} ({base_size_gb:.1f}GB needed, "
                f"{metrics.available_gb:.1f}GB available). Degrading quantization."
            )
            level = self._degrade_quantization(level)
            base_size_gb = self.MODEL_SIZES_GB.get(level, 3.5) * (model_size_params / 7.0)
        
        # Build configuration
        config = self._build_quantization_config(level, model_size_params)
        
        return QuantizationConfig(
            level=level,
            precision=level.value,
            compression_ratio=14.0 / base_size_gb,
            estimated_memory_gb=base_size_gb,
            config=config
        )
    
    def _degrade_quantization(self, current_level: QuantizationLevel) -> QuantizationLevel:
        """Degrade to next lower quantization level."""
        degradation_order = [
            QuantizationLevel.FP16,
            QuantizationLevel.INT8,
            QuantizationLevel.INT4,
            QuantizationLevel.INT2
        ]
        
        try:
            current_idx = degradation_order.index(current_level)
            if current_idx < len(degradation_order) - 1:
                return degradation_order[current_idx + 1]
        except ValueError:
            pass
        
        return QuantizationLevel.INT4  # Default fallback
    
    def _build_quantization_config(
        self,
        level: QuantizationLevel,
        model_size_params: float
    ) -> Dict[str, Any]:
        """Build quantization configuration for transformers."""
        config = {
            "quantization_level": level.value,
            "model_size_params": model_size_params,
            "device": self.device,
        }
        
        if level == QuantizationLevel.FP16:
            config.update({
                "torch_dtype": "float16",
                "load_in_8bit": False,
                "load_in_4bit": False,
            })
        
        elif level == QuantizationLevel.INT8:
            config.update({
                "load_in_8bit": True,
                "load_in_4bit": False,
                "llm_int8_threshold": 6.0,
            })
        
        elif level == QuantizationLevel.INT4:
            config.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            })
        
        elif level == QuantizationLevel.INT2:
            config.update({
                "use_gguf": True,
                "quantization": "Q2_K",  # 2-bit GGUF
                "n_gpu_layers": -1 if self.device in ["cuda", "mps"] else 0,
            })
        
        return config
    
    def register_request_start(self):
        """Register that a request has started (for load tracking)."""
        with self.request_lock:
            self.active_requests += 1
            logger.debug(f"Request started. Active requests: {self.active_requests}")
    
    def register_request_end(self):
        """Register that a request has ended."""
        with self.request_lock:
            self.active_requests = max(0, self.active_requests - 1)
            logger.debug(f"Request ended. Active requests: {self.active_requests}")
    
    def should_reoptimize(self, current_config: QuantizationConfig) -> bool:
        """
        Check if quantization should be re-optimized based on changed conditions.
        
        Args:
            current_config: Current quantization configuration
        
        Returns:
            True if re-optimization recommended
        """
        metrics = self.get_memory_metrics()
        
        # Reoptimize if memory pressure changed significantly
        if current_config.level == QuantizationLevel.FP16:
            if metrics.percent_used > self.MEMORY_MODERATE:
                logger.info("Memory pressure increased - recommending re-optimization")
                return True
        
        elif current_config.level == QuantizationLevel.INT8:
            if metrics.percent_used > self.MEMORY_TIGHT:
                logger.info("Memory pressure critical - recommending re-optimization")
                return True
            elif metrics.percent_used < self.MEMORY_COMFORTABLE and self.active_requests < self.LOAD_LOW:
                logger.info("Memory freed up - recommending upgrade to FP16")
                return True
        
        elif current_config.level == QuantizationLevel.INT4:
            if metrics.percent_used < self.MEMORY_MODERATE and self.active_requests < self.LOAD_LOW:
                logger.info("Resources available - recommending upgrade to INT8")
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current quantization manager status."""
        metrics = self.get_memory_metrics()
        
        return {
            "device": self.device,
            "active_requests": self.active_requests,
            "memory": {
                "total_gb": metrics.total_gb,
                "available_gb": metrics.available_gb,
                "used_percent": metrics.percent_used,
                "process_gb": metrics.process_memory_gb,
            },
            "capabilities": {
                "fp16": self.supports_fp16,
                "int8": self.supports_int8,
                "int4": self.supports_int4,
            },
            "recommended_level": self.calculate_optimal_quantization(7.0).level.value,
        }


# Singleton instance
_quantization_manager: Optional[DynamicQuantizationManager] = None
_manager_lock = threading.Lock()


def get_quantization_manager() -> DynamicQuantizationManager:
    """Get singleton quantization manager instance."""
    global _quantization_manager
    
    if _quantization_manager is None:
        with _manager_lock:
            if _quantization_manager is None:
                _quantization_manager = DynamicQuantizationManager()
    
    return _quantization_manager


import threading
