"""Centralized configuration management - Optimal 2025 Model Stack."""

import logging
import os
import secrets
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal

# Load .env file BEFORE any settings are read
# This ensures environment variables are available when Settings() is instantiated
try:
    from dotenv import load_dotenv

    # Load from project root (assuming this file is in backend/core/)
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars


DeploymentTier = Literal["local", "production"]


class ModelBackend(str, Enum):
    """Model inference backends."""

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    TGI = "tgi"


class Settings:
    """Application settings with optimal 2025 model stack."""

    # =================================================================
    # APPLICATION
    # =================================================================
    APP_NAME: str = "ShikshaSetu AI Education API"
    APP_VERSION: str = "4.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"  # Enabled until stable

    # Deployment Configuration
    DEPLOYMENT_TIER: DeploymentTier = os.getenv("DEPLOYMENT_TIER", "local")  # type: ignore[assignment]

    # =================================================================
    # DEVICE & COMPUTE
    # =================================================================
    DEVICE: str = os.getenv("DEVICE", "auto")  # auto | cuda | mps | cpu
    USE_QUANTIZATION: bool = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
    QUANTIZATION_TYPE: str = os.getenv(
        "QUANTIZATION_TYPE", "int4"
    )  # int4 | int8 | fp16
    USE_FLASH_ATTENTION: bool = (
        os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"
    )
    MAX_GPU_MEMORY_GB: float = float(os.getenv("MAX_GPU_MEMORY_GB", "16.0"))

    # =================================================================
    # DIRECTORIES
    # =================================================================
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "data/models"))
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "logs"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "shiksha_setu.log")

    # =================================================================
    # OPTIMAL MODEL STACK (2025)
    # =================================================================

    # --- Text Simplification: Qwen2.5-3B-Instruct ---
    SIMPLIFICATION_MODEL_ID: str = os.getenv(
        "SIMPLIFICATION_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"
    )
    SIMPLIFICATION_BACKEND: str = os.getenv("SIMPLIFICATION_BACKEND", "transformers")
    SIMPLIFICATION_MAX_LENGTH: int = int(
        os.getenv("SIMPLIFICATION_MAX_LENGTH", "4096")
    )  # Increased
    SIMPLIFICATION_TEMPERATURE: float = float(
        os.getenv("SIMPLIFICATION_TEMPERATURE", "0.7")
    )

    # --- Translation: IndicTrans2-1B ---
    TRANSLATION_MODEL_ID: str = os.getenv(
        "TRANSLATION_MODEL_ID", "ai4bharat/indictrans2-en-indic-1B"
    )
    TRANSLATION_BACKEND: str = os.getenv("TRANSLATION_BACKEND", "transformers")
    TRANSLATION_MAX_LENGTH: int = int(
        os.getenv("TRANSLATION_MAX_LENGTH", "2048")
    )  # Increased for Indian scripts

    # Supported Indian languages
    SUPPORTED_LANGUAGES: list[str] = [
        "Hindi",
        "Tamil",
        "Telugu",
        "Bengali",
        "Marathi",
        "Gujarati",
        "Kannada",
        "Malayalam",
        "Punjabi",
        "Odia",
    ]

    # --- Embeddings: BGE-M3 ---
    EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    EMBEDDING_MAX_LENGTH: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "8192"))
    # M4 Optimization: Batch size 64 achieves 348 texts/sec (benchmarked)
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

    # --- Reranker: BGE-Reranker-v2-M3 ---
    RERANKER_MODEL_ID: str = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-v2-m3")
    RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "10"))

    # --- TTS: MMS-TTS (Facebook) - Best for Indian languages ---
    # Primary: facebook/mms-tts-{lang} | Fallback: edge_tts
    TTS_BACKEND: str = os.getenv("TTS_BACKEND", "mms_tts")  # mms_tts | edge_tts
    TTS_MODEL_ID: str = os.getenv(
        "TTS_MODEL_ID",
        "facebook/mms-tts-hin",  # Default Hindi, auto-switches per language
    )
    TTS_SAMPLE_RATE: int = int(
        os.getenv("TTS_SAMPLE_RATE", "16000")
    )  # MMS-TTS uses 16kHz
    TTS_USE_GPU: bool = os.getenv("TTS_USE_GPU", "true").lower() == "true"

    # --- STT: Whisper Large V3 Turbo ---
    STT_MODEL_ID: str = os.getenv(
        "STT_MODEL_ID",
        "openai/whisper-large-v3-turbo",  # 8x faster, 99 languages
    )
    STT_USE_GPU: bool = os.getenv("STT_USE_GPU", "true").lower() == "true"

    # --- Validation: Gemma-2-2B-it ---
    VALIDATION_MODEL_ID: str = os.getenv("VALIDATION_MODEL_ID", "google/gemma-2-2b-it")
    VALIDATION_BACKEND: str = os.getenv("VALIDATION_BACKEND", "transformers")

    # --- OCR: GOT-OCR2 ---
    OCR_MODEL_ID: str = os.getenv("OCR_MODEL_ID", "ucaslcl/GOT-OCR2_0")
    OCR_BACKEND: str = os.getenv("OCR_BACKEND", "transformers")
    OCR_MAX_IMAGE_SIZE: int = int(os.getenv("OCR_MAX_IMAGE_SIZE", "2048"))

    # =================================================================
    # vLLM PRODUCTION SERVING (disabled by default for local M4 dev)
    # =================================================================
    VLLM_ENABLED: bool = os.getenv("VLLM_ENABLED", "false").lower() == "true"
    VLLM_HOST: str = os.getenv("VLLM_HOST", "localhost")
    VLLM_PORT: int = int(os.getenv("VLLM_PORT", "8001"))
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "")
    VLLM_TENSOR_PARALLEL_SIZE: int = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    VLLM_GPU_MEMORY_UTILIZATION: float = float(
        os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
    )
    VLLM_MAX_MODEL_LEN: int = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
    VLLM_QUANTIZATION: str = os.getenv("VLLM_QUANTIZATION", "awq")  # awq | gptq | None

    # --- Principle C: Generous Swap Space ---
    VLLM_SWAP_SPACE_GB: int = int(
        os.getenv("VLLM_SWAP_SPACE_GB", "4")
    )  # 4GB swap for KV cache overflow
    VLLM_CPU_OFFLOAD_GB: float = float(
        os.getenv("VLLM_CPU_OFFLOAD_GB", "0")
    )  # CPU offload (set > 0 to enable)

    # --- Principle D: Tensor Parallelism for Multi-GPU ---
    VLLM_PIPELINE_PARALLEL_SIZE: int = int(
        os.getenv("VLLM_PIPELINE_PARALLEL_SIZE", "1")
    )
    VLLM_ENFORCE_EAGER: bool = (
        os.getenv("VLLM_ENFORCE_EAGER", "false").lower() == "true"
    )  # Disable CUDA graph
    VLLM_MAX_PARALLEL_LOADING_WORKERS: int = int(
        os.getenv("VLLM_MAX_PARALLEL_LOADING_WORKERS", "2")
    )

    # --- Principle H: KV Cache Configuration ---
    VLLM_BLOCK_SIZE: int = int(
        os.getenv("VLLM_BLOCK_SIZE", "16")
    )  # KV cache block size
    VLLM_ENABLE_PREFIX_CACHING: bool = (
        os.getenv("VLLM_ENABLE_PREFIX_CACHING", "true").lower() == "true"
    )

    # --- Principle O: Threadpool for Non-Model I/O ---
    # M4 Optimization: Match P-core count (4) for ML inference, E-cores for async
    THREADPOOL_MAX_WORKERS: int = int(
        os.getenv(
            "THREADPOOL_MAX_WORKERS",
            "4",  # M4: 4 P-cores for ML inference threads
        )
    )
    ASYNC_POOL_SIZE: int = int(
        os.getenv(
            "ASYNC_POOL_SIZE",
            "6",  # M4: 6 E-cores for async I/O operations
        )
    )
    # M4 GPU pipeline optimization
    GPU_BATCH_COALESCE_MS: int = int(
        os.getenv("GPU_BATCH_COALESCE_MS", "10")
    )  # Batch window
    GPU_MAX_CONCURRENT: int = int(
        os.getenv("GPU_MAX_CONCURRENT", "2")
    )  # Metal queue limit

    # =================================================================
    # API CONFIGURATION
    # =================================================================
    API_PREFIX: str = "/api/v2"  # Current API version prefix
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # =================================================================
    # SECURITY
    # =================================================================
    SECRET_KEY: str = ""  # Set in __init__
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    BCRYPT_ROUNDS: int = int(os.getenv("BCRYPT_ROUNDS", "12"))

    # Password Requirements
    PASSWORD_MIN_LENGTH: int = 12
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True

    # =================================================================
    # STORAGE
    # =================================================================
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # local | s3
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "shiksha-setu-uploads")

    # =================================================================
    # DATABASE
    # =================================================================
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://localhost:5432/shiksha_setu",  # Set credentials via env
    )
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    DB_POOL_PRE_PING: bool = True

    # =================================================================
    # REDIS & CELERY
    # =================================================================
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/1"
    )
    CELERY_TASK_ALWAYS_EAGER: bool = (
        os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true"
    )

    # Celery Task Configuration
    TASK_RESULT_EXPIRES: int = 3600
    TASK_ACKS_LATE: bool = True
    TASK_REJECT_ON_WORKER_LOST: bool = True
    WORKER_PREFETCH_MULTIPLIER: int = 1

    # =================================================================
    # CORS
    # NOTE: In production, set ALLOWED_ORIGINS environment variable
    # to a comma-separated list of allowed origins (e.g., https://example.com)
    # Default localhost origins are for development only.
    # =================================================================
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ]
    ALLOW_CREDENTIALS: bool = True
    ALLOWED_METHODS: list[str] = ["*"]
    ALLOWED_HEADERS: list[str] = ["*"]

    # =================================================================
    # RATE LIMITING
    # =================================================================
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    RATE_LIMIT_BURST_MULTIPLIER: int = int(
        os.getenv("RATE_LIMIT_BURST_MULTIPLIER", "2")
    )
    RATE_LIMIT_STORAGE: str = os.getenv("RATE_LIMIT_STORAGE", "redis")

    # =================================================================
    # UNIVERSAL MODE (AI for India - No Content Restrictions)
    # =================================================================
    # When enabled, the system operates like ChatGPT/Perplexity - no NCERT,
    # grade level, or subject restrictions. Handles philosophy, politics,
    # music, poetry, history, art, and all human knowledge.
    UNIVERSAL_MODE: bool = os.getenv("UNIVERSAL_MODE", "true").lower() == "true"

    # Age consent requirement - when True, users must confirm 18+ for
    # mature content topics. This is the ONLY content-related restriction.
    AGE_CONSENT_REQUIRED: bool = (
        os.getenv("AGE_CONSENT_REQUIRED", "true").lower() == "true"
    )

    # Mature content topics that require age consent (optional safeguard)
    MATURE_CONTENT_TOPICS: list[str] = [
        "explicit violence",
        "adult content",
        "substance abuse",
        "graphic medical content",
        "extreme political violence",
    ]

    # =================================================================
    # LOGGING
    # =================================================================
    LOG_MAX_BYTES: int = 10485760  # 10 MB
    LOG_BACKUP_COUNT: int = 5
    SLOW_REQUEST_THRESHOLD: float = float(os.getenv("SLOW_REQUEST_THRESHOLD", "5.0"))

    def __init__(self):
        """Initialize settings and create necessary directories."""
        # Initialize SECRET_KEY
        key = os.getenv("JWT_SECRET_KEY", "")
        if not key or len(key) < 64:
            key = secrets.token_urlsafe(64)
            logger = logging.getLogger(__name__)
            logger.warning(
                "JWT_SECRET_KEY not set or too short. Generated temporary key. "
                "Set JWT_SECRET_KEY environment variable (64+ chars) for production!"
            )
            if self.ENVIRONMENT == "production":
                raise ValueError(
                    "JWT_SECRET_KEY must be set explicitly in production (min 64 characters). "
                    "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
                )
        self.SECRET_KEY = key

        # Parse ALLOWED_ORIGINS from environment
        allowed_env = os.getenv("ALLOWED_ORIGINS")
        if allowed_env:
            try:
                parsed = [o.strip() for o in allowed_env.split(",") if o.strip()]
                if parsed:
                    self.ALLOWED_ORIGINS = parsed
            except Exception:
                pass

        # Create directories
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def vllm_base_url(self) -> str:
        """Get vLLM API base URL."""
        return f"http://{self.VLLM_HOST}:{self.VLLM_PORT}/v1"

    def get_model_config(self, task: str) -> dict[str, Any]:
        """Get model configuration for a specific task."""
        configs: dict[str, dict[str, Any]] = {
            "simplification": {
                "model_id": self.SIMPLIFICATION_MODEL_ID,
                "backend": self.SIMPLIFICATION_BACKEND,
                "max_length": self.SIMPLIFICATION_MAX_LENGTH,
                "temperature": self.SIMPLIFICATION_TEMPERATURE,
            },
            "translation": {
                "model_id": self.TRANSLATION_MODEL_ID,
                "backend": self.TRANSLATION_BACKEND,
                "max_length": self.TRANSLATION_MAX_LENGTH,
            },
            "embedding": {
                "model_id": self.EMBEDDING_MODEL_ID,
                "dimension": self.EMBEDDING_DIMENSION,
                "max_length": self.EMBEDDING_MAX_LENGTH,
                "batch_size": self.EMBEDDING_BATCH_SIZE,
            },
            "reranker": {
                "model_id": self.RERANKER_MODEL_ID,
                "top_k": self.RERANKER_TOP_K,
            },
            "tts": {
                "model_id": self.TTS_MODEL_ID,
                "sample_rate": self.TTS_SAMPLE_RATE,
            },
            "validation": {
                "model_id": self.VALIDATION_MODEL_ID,
                "backend": self.VALIDATION_BACKEND,
            },
            "ocr": {
                "model_id": self.OCR_MODEL_ID,
                "backend": self.OCR_BACKEND,
                "max_image_size": self.OCR_MAX_IMAGE_SIZE,
            },
        }
        return configs.get(task, {})

    def _validate_production_env(self, issues: list[str]) -> None:
        """Validate production environment settings."""
        if self.ENVIRONMENT != "production":
            return

        if not os.getenv("JWT_SECRET_KEY"):
            issues.append("ERROR: JWT_SECRET_KEY not set (required in production)")
        if not os.getenv("DATABASE_URL"):
            issues.append("ERROR: DATABASE_URL not set (required in production)")
        if self.DEBUG:
            issues.append("WARNING: DEBUG=true in production")

    def _validate_models(self, issues: list[str]) -> None:
        """Validate model configuration."""
        model_checks = [
            ("SIMPLIFICATION_MODEL_ID", self.SIMPLIFICATION_MODEL_ID),
            ("TRANSLATION_MODEL_ID", self.TRANSLATION_MODEL_ID),
            ("EMBEDDING_MODEL_ID", self.EMBEDDING_MODEL_ID),
        ]
        for name, model_id in model_checks:
            if not model_id or "default" in model_id.lower():
                issues.append(f"WARNING: {name} using default value")

    def _validate_directories(self, issues: list[str]) -> None:
        """Validate directory permissions."""
        for dir_name, dir_path in [
            ("UPLOAD_DIR", self.UPLOAD_DIR),
            ("MODEL_CACHE_DIR", self.MODEL_CACHE_DIR),
            ("LOG_DIR", self.LOG_DIR),
        ]:
            if not os.access(dir_path, os.W_OK):
                issues.append(f"WARNING: {dir_name} ({dir_path}) not writable")

    def validate_required(self) -> list[str]:
        """
        Validate required configuration at startup.
        Returns list of warnings/errors.
        """
        issues: list[str] = []
        logger = logging.getLogger(__name__)

        self._validate_production_env(issues)
        self._validate_models(issues)
        self._validate_directories(issues)

        # Log issues
        for issue in issues:
            log_method = logger.error if issue.startswith("ERROR") else logger.warning
            log_method(issue)

        return issues

    @staticmethod
    def check_tts_availability() -> tuple[bool, str]:
        """
        Check if TTS is available (MMS-TTS or Edge-TTS).
        Returns: (is_available, reason)
        """
        # Check MMS-TTS (primary)
        try:
            from transformers import VitsModel

            return True, "MMS-TTS available (facebook/mms-tts-*)"
        except ImportError:
            pass

        # Check Edge-TTS (fallback)
        try:
            import edge_tts

            return True, "Edge-TTS available (fallback)"
        except ImportError:
            pass

        return False, "No TTS backend available (install transformers or edge-tts)"

    @staticmethod
    def check_stt_availability() -> tuple[bool, str]:
        """
        Check if STT is available (Whisper via transformers).
        Returns: (is_available, reason)
        """
        try:
            from transformers import pipeline

            return True, "Whisper available (openai/whisper-large-v3-turbo)"
        except ImportError:
            return False, "Whisper not available (install transformers)"

    @staticmethod
    def check_feature_availability() -> dict:
        """Check availability of all optional features."""
        features = {}

        # TTS
        tts_available, tts_reason = Settings.check_tts_availability()
        features["tts"] = {"available": tts_available, "reason": tts_reason}

        # pydub for audio processing
        try:
            from pydub import AudioSegment

            features["audio_processing"] = {
                "available": True,
                "reason": "pydub available",
            }
        except ImportError:
            features["audio_processing"] = {
                "available": False,
                "reason": "pydub not installed",
            }

        # OCR
        try:
            import fitz  # PyMuPDF

            features["ocr"] = {"available": True, "reason": "PyMuPDF available"}
        except ImportError:
            features["ocr"] = {"available": False, "reason": "PyMuPDF not installed"}

        return features


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
