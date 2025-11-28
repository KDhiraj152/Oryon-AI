"""
Model Tier Router for resource-aware inference.

Routes tasks to appropriate model sizes based on complexity and available resources.
Implements local-first strategy with graceful degradation to API fallback.
"""
import logging
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model size tiers for resource-aware inference."""
    SMALL = "small"      # 1-3B params, <2GB RAM, fast inference
    MEDIUM = "medium"    # 7B params, 3-8GB RAM, balanced
    LARGE = "large"      # 13B+ params or API, 8GB+ RAM, best quality
    API = "api"          # External API fallback (Bhashini, OpenAI)


@dataclass
class TaskComplexity:
    """Task complexity metrics for routing decisions."""
    token_count: int
    grade_level: int
    subject_technical: bool  # Science/Math vs Social/English
    translation_pairs: int   # Number of languages
    requires_cultural_context: bool
    complexity_score: float


class ModelTierRouter:
    """
    Routes inference tasks to appropriate model tiers.
    
    Strategy:
    1. Calculate task complexity score
    2. Check available resources (memory, device)
    3. Select optimal tier (SMALL/MEDIUM/LARGE/API)
    4. Return model configuration
    
    Thresholds are tuned for Apple Silicon M4 with 16GB unified memory.
    """
    
    # Token thresholds for complexity scoring
    TOKEN_SMALL = 512       # <512 tokens = SMALL model sufficient
    TOKEN_MEDIUM = 2048     # <2048 tokens = MEDIUM model
    TOKEN_LARGE = 4096      # >4096 tokens = LARGE model or API
    
    # Grade level thresholds
    GRADE_SIMPLE = 8        # Grade 5-8: simpler language
    GRADE_COMPLEX = 10      # Grade 9-10: more complex
    
    # Technical subjects need better models
    TECHNICAL_SUBJECTS = {'Mathematics', 'Science', 'Physics', 'Chemistry', 'Biology'}
    
    # Memory budgets per tier (in GB)
    MEMORY_BUDGET = {
        ModelTier.SMALL: 2.0,    # 1.5B model in 4-bit
        ModelTier.MEDIUM: 6.0,   # 7B model in 4-bit
        ModelTier.LARGE: 12.0,   # 13B model in 4-bit or API
        ModelTier.API: 0.1       # Minimal local memory
    }
    
    # Model configurations per tier
    MODEL_CONFIGS = {
        ModelTier.SMALL: {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "quantization": "4bit",
            "max_tokens": 512,
            "batch_size": 8,
            "device_preference": ["mps", "cuda", "cpu"],
        },
        ModelTier.MEDIUM: {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": "4bit",
            "max_tokens": 2048,
            "batch_size": 4,
            "device_preference": ["mps", "cuda", "cpu"],
        },
        ModelTier.LARGE: {
            "model_id": "Qwen/Qwen2.5-14B-Instruct",  # Or vLLM endpoint
            "quantization": "4bit",
            "max_tokens": 4096,
            "batch_size": 1,
            "device_preference": ["cuda", "api"],  # Prefer GPU or API
        },
        ModelTier.API: {
            "api_provider": "bhashini",
            "fallback": True,
            "max_tokens": 4096,
        }
    }
    
    def __init__(self, max_memory_gb: float = 8.0, device_type: str = "mps"):
        """
        Initialize model tier router.
        
        Args:
            max_memory_gb: Maximum memory budget in GB (default 8GB for M4)
            device_type: Device type (mps, cuda, cpu)
        """
        self.max_memory_gb = max_memory_gb
        self.device_type = device_type
        
        # Track current memory usage (simplified - actual tracking in ModelLoader)
        self.current_memory_gb = 0.0
        
        logger.info(
            f"ModelTierRouter initialized: max_memory={max_memory_gb}GB, device={device_type}"
        )
    
    def calculate_task_complexity(
        self,
        text: str,
        grade_level: int,
        subject: str,
        language_count: int = 1,
        requires_cultural_context: bool = False
    ) -> TaskComplexity:
        """
        Calculate task complexity score.
        
        Args:
            text: Input text
            grade_level: Target grade level (5-12)
            subject: Subject area
            language_count: Number of language pairs (for translation)
            requires_cultural_context: Whether cultural adaptation needed
            
        Returns:
            TaskComplexity object with metrics
        """
        # Estimate token count (rough: 1 token ≈ 4 chars)
        token_count = len(text) // 4
        
        # Technical subject increases complexity
        is_technical = subject in self.TECHNICAL_SUBJECTS
        
        # Calculate complexity score (0.0 - 1.0)
        score = 0.0
        
        # Token count factor (40% weight)
        if token_count < self.TOKEN_SMALL:
            score += 0.1
        elif token_count < self.TOKEN_MEDIUM:
            score += 0.25
        else:
            score += 0.4
        
        # Grade level factor (20% weight)
        if grade_level <= self.GRADE_SIMPLE:
            score += 0.05
        elif grade_level <= self.GRADE_COMPLEX:
            score += 0.15
        else:
            score += 0.2
        
        # Subject factor (20% weight)
        if is_technical:
            score += 0.2
        else:
            score += 0.1
        
        # Language factor (10% weight)
        if language_count > 1:
            score += 0.1
        
        # Cultural context (10% weight)
        if requires_cultural_context:
            score += 0.1
        
        complexity = TaskComplexity(
            token_count=token_count,
            grade_level=grade_level,
            subject_technical=is_technical,
            translation_pairs=language_count,
            requires_cultural_context=requires_cultural_context,
            complexity_score=min(score, 1.0)
        )
        
        logger.debug(f"Task complexity: {complexity.complexity_score:.2f} (tokens={token_count})")
        
        return complexity
    
    def select_tier(
        self,
        complexity: TaskComplexity,
        force_tier: Optional[ModelTier] = None,
        available_memory_gb: Optional[float] = None
    ) -> ModelTier:
        """
        Select optimal model tier based on complexity and resources.
        
        Args:
            complexity: TaskComplexity object
            force_tier: Force specific tier (for testing)
            available_memory_gb: Available memory (if known)
            
        Returns:
            Selected ModelTier
        """
        if force_tier:
            logger.info(f"Forced tier: {force_tier}")
            return force_tier
        
        # Check available memory
        available = available_memory_gb or (self.max_memory_gb - self.current_memory_gb)
        
        # Decision logic based on complexity score
        if complexity.complexity_score < 0.3:
            # Simple task - use SMALL model
            tier = ModelTier.SMALL
        elif complexity.complexity_score < 0.6:
            # Medium complexity - prefer MEDIUM model
            if available >= self.MEMORY_BUDGET[ModelTier.MEDIUM]:
                tier = ModelTier.MEDIUM
            else:
                logger.warning(
                    f"Insufficient memory for MEDIUM ({available:.1f}GB < "
                    f"{self.MEMORY_BUDGET[ModelTier.MEDIUM]}GB), using SMALL"
                )
                tier = ModelTier.SMALL
        else:
            # High complexity - prefer LARGE or API
            if self.device_type == "cuda" and available >= self.MEMORY_BUDGET[ModelTier.LARGE]:
                tier = ModelTier.LARGE
            else:
                # For MPS/CPU with high complexity, use API
                logger.info(
                    f"High complexity task on {self.device_type}, routing to API"
                )
                tier = ModelTier.API
        
        logger.info(
            f"Selected tier: {tier.value} (complexity={complexity.complexity_score:.2f}, "
            f"available_memory={available:.1f}GB)"
        )
        
        return tier
    
    def get_model_config(self, tier: ModelTier) -> Dict[str, Any]:
        """
        Get model configuration for the selected tier.
        
        Args:
            tier: Selected ModelTier
            
        Returns:
            Model configuration dictionary
        """
        config = self.MODEL_CONFIGS.get(tier, self.MODEL_CONFIGS[ModelTier.API])
        
        # Add runtime info
        config["tier"] = tier.value
        config["device_type"] = self.device_type
        config["memory_budget_gb"] = self.MEMORY_BUDGET.get(tier, 0.1)
        
        return config
    
    def route_task(
        self,
        text: str,
        grade_level: int,
        subject: str,
        language_count: int = 1,
        requires_cultural_context: bool = False,
        force_tier: Optional[ModelTier] = None
    ) -> Tuple[ModelTier, Dict[str, Any], TaskComplexity]:
        """
        Complete routing: calculate complexity, select tier, return config.
        
        Args:
            text: Input text
            grade_level: Target grade level
            subject: Subject area
            language_count: Number of languages
            requires_cultural_context: Cultural adaptation needed
            force_tier: Force specific tier
            
        Returns:
            Tuple of (selected_tier, model_config, task_complexity)
        """
        # Calculate complexity
        complexity = self.calculate_task_complexity(
            text=text,
            grade_level=grade_level,
            subject=subject,
            language_count=language_count,
            requires_cultural_context=requires_cultural_context
        )
        
        # Select tier
        tier = self.select_tier(complexity, force_tier)
        
        # Get configuration
        config = self.get_model_config(tier)
        
        logger.info(
            f"Route decision: tier={tier.value}, model={config.get('model_id', 'api')}, "
            f"complexity={complexity.complexity_score:.2f}"
        )
        
        return tier, config, complexity
    
    def update_memory_usage(self, tier: ModelTier, loaded: bool):
        """
        Update current memory usage tracking.
        
        Args:
            tier: Model tier loaded/unloaded
            loaded: True if loaded, False if unloaded
        """
        memory = self.MEMORY_BUDGET.get(tier, 0.0)
        if loaded:
            self.current_memory_gb += memory
        else:
            self.current_memory_gb = max(0, self.current_memory_gb - memory)
        
        logger.debug(
            f"Memory usage: {self.current_memory_gb:.1f}GB / {self.max_memory_gb}GB "
            f"({self.current_memory_gb / self.max_memory_gb * 100:.1f}%)"
        )


# Global router instance (initialized in main.py)
_router: Optional[ModelTierRouter] = None


def get_router() -> ModelTierRouter:
    """Get global router instance (lazy init)."""
    global _router
    if _router is None:
        from ..core.config import settings
        from ..utils.device_manager import get_device_manager
        
        device_manager = get_device_manager()
        
        # Default to 8GB for M4, can be overridden via settings
        max_memory = getattr(settings, 'MAX_MODEL_MEMORY_GB', 8.0)
        
        _router = ModelTierRouter(
            max_memory_gb=max_memory,
            device_type=device_manager.device
        )
    
    return _router


def init_router(max_memory_gb: float = 8.0, device_type: str = "mps") -> ModelTierRouter:
    """Initialize global router (called at startup)."""
    global _router
    _router = ModelTierRouter(max_memory_gb=max_memory_gb, device_type=device_type)
    return _router
