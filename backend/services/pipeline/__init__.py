# Pipeline orchestrator module - import from v2 directly for main classes
# Early stop heuristics (Principle L)
from .early_stop import (
    EarlyStopConfig,
    EarlyStopDetector,
    StopReason,
    StreamingEarlyStopCallback,
    check_should_stop,
    create_early_stop_detector,
    get_clean_output,
)

# NEW: Model Collaboration System (multi-model orchestration)
from .model_collaboration import (
    CollaborationConfig,
    CollaborationPattern,
    CollaborationResult,
    ModelCollaborator,
    ModelMessage,
    ModelRole,
    collaborate_and_simplify,
    collaborate_and_translate,
    ensemble_evaluate,
    full_content_pipeline,
    generate_best_output,
    get_model_collaborator,
    process_document,
    # NEW: 8-model collaboration functions
    verify_audio_output,
)

# Backward compatibility aliases
from .orchestrator import (
    ContentPipelineOrchestrator,  # Alias for ConcurrentPipelineOrchestrator
    PipelineStageError,
    PipelineValidationError,
)
from .orchestrator_v2 import (
    ConcurrentPipelineOrchestrator,
    PipelineCircuitBreaker,
    PipelineStage,
    ProcessedContentResult,
    ProcessingStatus,
    StageMetrics,
)
from .sentence_pipeline import (
    DocumentResult,
    SentenceResult,
)

# Sentence-level processing (Principle M)
from .sentence_pipeline import (
    SentencePipeline as SentenceLevelProcessor,
)

# Pre-tokenization (Principle G)
from .tokenization_worker import (
    PreTokenizationWorker,
    TokenizationConfig,
    TokenizedInput,
    TokenizerType,
    chunk_text,
    get_tokenization_worker,
    pre_tokenize,
    pre_tokenize_batch,
)

# NEW: Unified pipeline (optimized for Apple Silicon)
from .unified_pipeline import (
    ProcessingProgress,
    ProcessingRequest,
    ProcessingResult,
    UnifiedPipelineService,
    get_pipeline_service,
)
from .unified_pipeline import (
    ProcessingStage as UnifiedProcessingStage,
)

__all__ = [
    # Legacy Orchestrator (backward compatibility)
    "ContentPipelineOrchestrator",
    "DocumentResult",
    "EarlyStopConfig",
    # Early stop (Principle L)
    "EarlyStopDetector",
    "PipelineStage",
    "PipelineStageError",
    "PipelineValidationError",
    # Pre-tokenization (Principle G)
    "PreTokenizationWorker",
    "ProcessedContentResult",
    "ProcessingProgress",
    "ProcessingRequest",
    "ProcessingResult",
    "ProcessingStatus",
    # Sentence processing (Principle M)
    "SentenceLevelProcessor",
    "SentenceResult",
    "StageMetrics",
    "StopReason",
    "StreamingEarlyStopCallback",
    "TokenizationConfig",
    "TokenizedInput",
    "TokenizerType",
    # NEW: Unified Pipeline (recommended)
    "UnifiedPipelineService",
    "UnifiedProcessingStage",
    "check_should_stop",
    "chunk_text",
    "create_early_stop_detector",
    "get_clean_output",
    "get_pipeline_service",
    "get_tokenization_worker",
    "pre_tokenize",
    "pre_tokenize_batch",
]
