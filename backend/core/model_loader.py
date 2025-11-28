"""
Lazy Model Loader for Optimized Models.

This module provides lazy loading of optimized models to minimize memory usage
and improve startup time. Supports quantization (4-bit/8-bit) and MPS optimization.
"""
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
import time
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model loading status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    EVICTED = "evicted"


@dataclass
class LoadedModel:
    """Wrapper for a loaded model."""
    name: str
    model: Any
    status: ModelStatus
    size_mb: float
    load_time_ms: float
    last_accessed: float
    access_count: int
    metadata: Dict[str, Any]


class ModelCache:
    """
    LRU cache for loaded models with automatic eviction.
    """
    
    def __init__(self, max_cache_size_mb: int = 500):
        """
        Initialize model cache.
        
        Args:
            max_cache_size_mb: Maximum cache size in MB
        """
        self.max_cache_size_mb = max_cache_size_mb
        self.cache: Dict[str, LoadedModel] = {}
        self.lock = threading.Lock()
        logger.info(f"ModelCache initialized with {max_cache_size_mb}MB limit")
    
    def get(self, model_name: str) -> Optional[LoadedModel]:
        """Get model from cache."""
        with self.lock:
            if model_name in self.cache:
                loaded_model = self.cache[model_name]
                loaded_model.last_accessed = time.time()
                loaded_model.access_count += 1
                logger.debug(f"Cache hit for {model_name} (accesses: {loaded_model.access_count})")
                return loaded_model
            
            logger.debug(f"Cache miss for {model_name}")
            return None
    
    def put(self, model_name: str, loaded_model: LoadedModel):
        """Put model in cache with automatic eviction."""
        with self.lock:
            # Check if we need to evict
            current_size = self._get_current_size_mb()
            
            if current_size + loaded_model.size_mb > self.max_cache_size_mb:
                logger.info(
                    f"Cache full ({current_size:.1f}MB), evicting LRU models to fit "
                    f"{loaded_model.name} ({loaded_model.size_mb:.1f}MB)"
                )
                self._evict_lru(loaded_model.size_mb)
            
            self.cache[model_name] = loaded_model
            logger.info(
                f"Cached {model_name} ({loaded_model.size_mb:.1f}MB). "
                f"Cache: {self._get_current_size_mb():.1f}/{self.max_cache_size_mb}MB"
            )
    
    def remove(self, model_name: str):
        """Remove model from cache."""
        with self.lock:
            if model_name in self.cache:
                del self.cache[model_name]
                logger.info(f"Removed {model_name} from cache")
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def _get_current_size_mb(self) -> float:
        """Get current cache size in MB."""
        return sum(model.size_mb for model in self.cache.values())
    
    def _evict_lru(self, required_mb: float):
        """Evict least recently used models."""
        # Sort by last accessed time
        sorted_models = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        freed_mb = 0
        
        for model_name, loaded_model in sorted_models:
            if freed_mb >= required_mb:
                break
            
            logger.info(
                f"Evicting {model_name} ({loaded_model.size_mb:.1f}MB, "
                f"accessed {loaded_model.access_count} times)"
            )
            
            freed_mb += loaded_model.size_mb
            del self.cache[model_name]
        
        logger.info(f"Freed {freed_mb:.1f}MB by evicting {len(sorted_models)} models")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'current_size_mb': self._get_current_size_mb(),
                'max_size_mb': self.max_cache_size_mb,
                'models_cached': len(self.cache),
                'models': [
                    {
                        'name': name,
                        'size_mb': model.size_mb,
                        'access_count': model.access_count,
                        'status': model.status.value,
                    }
                    for name, model in self.cache.items()
                ]
            }


class LazyModelLoader:
    """
    Lazy loader for optimized models.
    
    Features:
    - Load models only when needed
    - Cache frequently used models
    - Automatic eviction of unused models
    - Thread-safe loading
    - Progress tracking
    """
    
    def __init__(
        self,
        models_dir: str = "data/models/optimized",
        max_cache_size_mb: int = 500,
        preload_models: Optional[list] = None
    ):
        """
        Initialize lazy model loader.
        
        Args:
            models_dir: Directory containing optimized models
            max_cache_size_mb: Maximum cache size in MB
            preload_models: List of model names to preload
        """
        self.models_dir = Path(models_dir)
        self.cache = ModelCache(max_cache_size_mb)
        self.loading_locks: Dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()
        
        # Model loaders (functions to load specific model types)
        self.model_loaders: Dict[str, Callable] = {}
        
        logger.info(f"LazyModelLoader initialized: {models_dir}")
        
        # Preload if specified
        if preload_models:
            self._preload_models(preload_models)
    
    def register_loader(
        self,
        model_name: str,
        loader_func: Callable,
        size_mb: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a model loader function.
        
        Args:
            model_name: Name of the model
            loader_func: Function to load the model
            size_mb: Model size in MB
            metadata: Additional metadata
        """
        self.model_loaders[model_name] = {
            'func': loader_func,
            'size_mb': size_mb,
            'metadata': metadata or {}
        }
        logger.info(f"Registered loader for {model_name} ({size_mb:.1f}MB)")
    
    def load_model(self, model_name: str) -> LoadedModel:
        """
        Load a model (lazy loading with caching).
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            LoadedModel instance
        """
        # Check cache first
        cached = self.cache.get(model_name)
        if cached and cached.status == ModelStatus.LOADED:
            return cached
        
        # Get or create lock for this model
        with self.global_lock:
            if model_name not in self.loading_locks:
                self.loading_locks[model_name] = threading.Lock()
        
        model_lock = self.loading_locks[model_name]
        
        # Load with lock (prevents duplicate loading)
        with model_lock:
            # Double-check cache (another thread may have loaded it)
            cached = self.cache.get(model_name)
            if cached and cached.status == ModelStatus.LOADED:
                return cached
            
            logger.info(f"Loading model: {model_name}")
            
            # Check if loader is registered
            if model_name not in self.model_loaders:
                raise ValueError(f"No loader registered for {model_name}")
            
            loader_info = self.model_loaders[model_name]
            
            # Create loading placeholder
            loading_model = LoadedModel(
                name=model_name,
                model=None,
                status=ModelStatus.LOADING,
                size_mb=loader_info['size_mb'],
                load_time_ms=0,
                last_accessed=time.time(),
                access_count=0,
                metadata=loader_info['metadata']
            )
            
            try:
                # Load the model
                start_time = time.time()
                model = loader_info['func']()
                load_time_ms = (time.time() - start_time) * 1000
                
                # Update loaded model
                loading_model.model = model
                loading_model.status = ModelStatus.LOADED
                loading_model.load_time_ms = load_time_ms
                
                # Cache it
                self.cache.put(model_name, loading_model)
                
                logger.info(
                    f"Loaded {model_name} in {load_time_ms:.1f}ms "
                    f"({loader_info['size_mb']:.1f}MB)"
                )
                
                return loading_model
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                loading_model.status = ModelStatus.ERROR
                loading_model.metadata['error'] = str(e)
                raise
    
    def unload_model(self, model_name: str):
        """
        Unload a model from cache.
        
        Args:
            model_name: Name of the model to unload
        """
        self.cache.remove(model_name)
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        logger.info(f"Unloaded model: {model_name}")
    
    def load_quantized_model(
        self,
        model_id: str,
        quantization: str = "4bit",
        device: str = "auto"
    ) -> Any:
        """
        Load a quantized transformer model.
        
        Args:
            model_id: HuggingFace model ID
            quantization: Quantization type (4bit, 8bit, none)
            device: Device to load on (auto, cuda, mps, cpu)
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading {model_id} with {quantization} quantization on {device}")
        
        # Detect device if auto
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Configure quantization
        if quantization == "4bit" and device == "cuda":
            # BitsAndBytes 4-bit (CUDA only)
            try:
                import bitsandbytes as bnb
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                logger.info(f"Loaded {model_id} in 4-bit (BitsAndBytes)")
            except ImportError:
                logger.warning("bitsandbytes not available, loading in FP16")
                quantization = "none"
        
        elif quantization == "8bit" and device == "cuda":
            # BitsAndBytes 8-bit (CUDA only)
            try:
                import bitsandbytes as bnb
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    load_in_8bit=True,
                    device_map="auto"
                )
                logger.info(f"Loaded {model_id} in 8-bit (BitsAndBytes)")
            except ImportError:
                logger.warning("bitsandbytes not available, loading in FP16")
                quantization = "none"
        
        if quantization in ["none", "4bit", "8bit"] and device in ["mps", "cpu"]:
            # Load in FP16 or FP32 for MPS/CPU
            # For MPS, use FP16 for better performance
            dtype = torch.float16 if device == "mps" else torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
            
            logger.info(f"Loaded {model_id} in {dtype} on {device}")
        
        return model
    
    def get_model_status(self, model_name: str) -> ModelStatus:
        """
        Get loading status of a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            ModelStatus
        """
        cached = self.cache.get(model_name)
        if cached:
            return cached.status
        return ModelStatus.UNLOADED
    
    def _preload_models(self, model_names: list):
        """Preload specified models."""
        logger.info(f"Preloading {len(model_names)} models")
        
        for model_name in model_names:
            try:
                self.load_model(model_name)
            except Exception as e:
                logger.error(f"Failed to preload {model_name}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear all cached models."""
        self.cache.clear()


# Example model loaders for different model types
def create_flan_t5_loader(model_path: str) -> Callable:
    """Create a loader function for FLAN-T5 model."""
    def loader():
        # In production, this would load the actual ONNX model
        # Requires onnxruntime: pip install onnxruntime
        
        logger.info(f"Loading FLAN-T5 from {model_path}")
        return {"type": "flan-t5", "path": model_path, "loaded": True}
    
    return loader


def create_indic_trans2_loader(model_path: str) -> Callable:
    """Create a loader function for IndicTrans2 model."""
    def loader():
        logger.info(f"Loading IndicTrans2 from {model_path}")
        return {"type": "indic-trans2", "path": model_path, "loaded": True}
    
    return loader


def create_mms_tts_loader(model_path: str) -> Callable:
    """Create a loader function for MMS-TTS model."""
    def loader():
        logger.info(f"Loading MMS-TTS from {model_path}")
        return {"type": "mms-tts", "path": model_path, "loaded": True}
    
    return loader


if __name__ == "__main__":
    # Example usage
    loader = LazyModelLoader(
        models_dir="data/models/optimized",
        max_cache_size_mb=500
    )
    
    # Register model loaders
    loader.register_loader(
        "flan-t5",
        create_flan_t5_loader("data/models/optimized/flan_t5_optimized"),
        size_mb=300,
        metadata={"type": "text-generation"}
    )
    
    loader.register_loader(
        "indic-trans2",
        create_indic_trans2_loader("data/models/optimized/indic_trans2_optimized"),
        size_mb=400,
        metadata={"type": "translation"}
    )
    
    loader.register_loader(
        "mms-tts",
        create_mms_tts_loader("data/models/optimized/mms_tts_optimized"),
        size_mb=100,
        metadata={"type": "tts"}
    )
    
    print("Lazy Model Loader Demo")
    print("=" * 60)
    
    # Load models on demand
    print("\n1. Loading FLAN-T5 (first time)...")
    model1 = loader.load_model("flan-t5")
    print(f"   Status: {model1.status.value}, Load time: {model1.load_time_ms:.1f}ms")
    
    print("\n2. Loading FLAN-T5 again (from cache)...")
    model2 = loader.load_model("flan-t5")
    print(f"   Status: {model2.status.value}, Access count: {model2.access_count}")
    
    print("\n3. Loading IndicTrans2...")
    model3 = loader.load_model("indic-trans2")
    print(f"   Status: {model3.status.value}, Load time: {model3.load_time_ms:.1f}ms")
    
    # Get cache stats
    print("\n4. Cache Statistics:")
    print("=" * 60)
    stats = loader.get_cache_stats()
    print(f"Cache: {stats['current_size_mb']:.1f}/{stats['max_size_mb']}MB")
    print(f"Models cached: {stats['models_cached']}")
    
    for model_info in stats['models']:
        print(f"  - {model_info['name']}: {model_info['size_mb']:.1f}MB "
              f"(accessed {model_info['access_count']} times)")
