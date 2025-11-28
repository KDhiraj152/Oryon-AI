"""Centralized configuration management."""
import os
import logging
from typing import List, Literal
from pathlib import Path
from datetime import timedelta
import secrets


DeploymentTier = Literal["local", "production"]


class Settings:
    """Application settings and configuration."""
    
    # Application
    APP_NAME: str = "ShikshaSetu AI Education API"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # Deployment Tier
    DEPLOYMENT_TIER: DeploymentTier = os.getenv("DEPLOYMENT_TIER", "local")  # local | production
    
    # Device Configuration
    DEVICE: str = os.getenv("DEVICE", "auto")  # auto | cuda | mps | cpu
    USE_QUANTIZATION: bool = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
    USE_FLASH_ATTENTION: bool = os.getenv("USE_FLASH_ATTENTION", "false").lower() == "true"
    
    # Model Configuration
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "data/models"))
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "logs"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "shiksha_setu.log")
    
    # Optimal Model Stack Configuration
    USE_LEGACY_MODELS: bool = os.getenv("USE_LEGACY_MODELS", "false").lower() == "true"
    
    # Content Generation (Qwen2.5)
    CONTENT_GEN_MODEL_ID: str = os.getenv("CONTENT_GEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    CONTENT_GEN_QUANTIZATION: str = os.getenv("CONTENT_GEN_QUANTIZATION", "4bit")
    CONTENT_GEN_USE_VLLM: bool = os.getenv("CONTENT_GEN_USE_VLLM", "false").lower() == "true"
    
    # Translation (IndicTrans2 Optimized)
    INDICTRANS2_QUANTIZATION: str = os.getenv("INDICTRANS2_QUANTIZATION", "int8")
    INDICTRANS2_USE_TORCHSCRIPT: bool = os.getenv("INDICTRANS2_USE_TORCHSCRIPT", "true").lower() == "true"
    
    # Embeddings (E5-Large)
    EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-large")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    EMBEDDING_USE_ONNX: bool = os.getenv("EMBEDDING_USE_ONNX", "true").lower() == "true"
    
    # Speech (MMS-TTS Optimized)
    VITS_QUANTIZATION: str = os.getenv("VITS_QUANTIZATION", "int8")
    
    # Curriculum Validator (IndicBERT)
    VALIDATOR_MODEL_ID: str = os.getenv("VALIDATOR_MODEL_ID", "ai4bharat/indic-bert")
    VALIDATOR_FINE_TUNE_PATH: str = os.getenv("VALIDATOR_FINE_TUNE_PATH", "./data/models/indic-bert-ncert-finetuned")
    
    # Lazy Loading Configuration
    LAZY_LOAD_ENABLED: bool = os.getenv("LAZY_LOAD_ENABLED", "true").lower() == "true"
    LAZY_LOAD_PRIORITY_1: List[str] = os.getenv("LAZY_LOAD_PRIORITY_1", "translation,speech,embeddings").split(",")
    LAZY_LOAD_PRIORITY_2: List[str] = os.getenv("LAZY_LOAD_PRIORITY_2", "content_generation,validator").split(",")
    
    # Memory Management
    MAX_GPU_MEMORY_GB: float = float(os.getenv("MAX_GPU_MEMORY_GB", "16.0"))
    MAX_MODEL_MEMORY_GB: float = float(os.getenv("MAX_MODEL_MEMORY_GB", "8.0"))  # For M4
    ENABLE_MODEL_OFFLOADING: bool = os.getenv("ENABLE_MODEL_OFFLOADING", "true").lower() == "true"
    MODEL_IDLE_TIMEOUT_MINUTES: int = int(os.getenv("MODEL_IDLE_TIMEOUT_MINUTES", "5"))
    
    # vLLM Production Settings
    VLLM_ENABLED: bool = os.getenv("VLLM_ENABLED", "false").lower() == "true"
    VLLM_TENSOR_PARALLEL_SIZE: int = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    VLLM_GPU_MEMORY_UTILIZATION: float = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
    
    # Bhashini API Configuration
    BHASHINI_API_KEY: str = os.getenv("BHASHINI_API_KEY", "")
    BHASHINI_API_URL: str = os.getenv(
        "BHASHINI_API_URL",
        "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    )
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Security Configuration
    @property
    def SECRET_KEY(self) -> str:
        """Get or generate JWT secret key."""
        key = os.getenv("JWT_SECRET_KEY", "")
        if not key or len(key) < 64:
            # Generate a secure random key for development
            key = secrets.token_hex(32)  # 64 character hex string
            if self.ENVIRONMENT == "production":
                raise ValueError(
                    "JWT_SECRET_KEY must be set in production environment! "
                    "Generate one with: python -c 'import secrets; print(secrets.token_hex(32))'"
                )
            logger = logging.getLogger(__name__)
            logger.warning(
                "JWT_SECRET_KEY not set or too short. Generated temporary key. "
                "Set JWT_SECRET_KEY environment variable (64+ chars) for production!"
            )
        return key
    
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    
    # Bcrypt Configuration
    BCRYPT_ROUNDS: int = int(os.getenv("BCRYPT_ROUNDS", "12"))
    
    # S3 Storage
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # local | s3
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "shiksha-setu-uploads")
    
    @property
    def is_secret_key_secure(self) -> bool:
        """Validate SECRET_KEY meets security requirements."""
        if not self.SECRET_KEY:
            return False
        if len(self.SECRET_KEY) < 64:  # Minimum 64 chars for production
            return False
        return True
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://shiksha_user:shiksha_pass@127.0.0.1:5432/shiksha_setu"
    )
    
    # Database Connection Pool
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    DB_POOL_PRE_PING: bool = True
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    CELERY_TASK_ALWAYS_EAGER: bool = os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true"
    
    # Task Configuration
    TASK_RESULT_EXPIRES: int = 3600  # 1 hour
    TASK_ACKS_LATE: bool = True
    TASK_REJECT_ON_WORKER_LOST: bool = True
    WORKER_PREFETCH_MULTIPLIER: int = 1
    
    # Security - JWT
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ]
    ALLOW_CREDENTIALS: bool = True
    ALLOWED_METHODS: List[str] = ["*"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    RATE_LIMIT_BURST_MULTIPLIER: int = int(os.getenv("RATE_LIMIT_BURST_MULTIPLIER", "2"))
    RATE_LIMIT_STORAGE: str = os.getenv("RATE_LIMIT_STORAGE", "memory")  # memory | redis
    
    # Password Requirements
    MIN_PASSWORD_LENGTH: int = 12
    PASSWORD_MIN_LENGTH: int = 12  # Alias for MIN_PASSWORD_LENGTH
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    
    # Logging Configuration
    LOG_MAX_BYTES: int = 10485760  # 10 MB
    LOG_BACKUP_COUNT: int = 5
    SLOW_REQUEST_THRESHOLD: float = float(os.getenv("SLOW_REQUEST_THRESHOLD", "5.0"))  # seconds
    
    def __init__(self):
        """Initialize settings and create necessary directories."""
        # Validate JWT secret key strength
        if not self.SECRET_KEY or len(self.SECRET_KEY) < 64:
            import secrets
            self.SECRET_KEY = secrets.token_urlsafe(64)
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
        
        # Allow overriding allowed origins via environment variable (comma-separated)
        allowed_env = os.getenv("ALLOWED_ORIGINS")
        if allowed_env:
            try:
                parsed = [o.strip() for o in allowed_env.split(",") if o.strip()]
                if parsed:
                    self.ALLOWED_ORIGINS = parsed
            except Exception as e:
                # keep defaults on parse failure
                logger.warning(f"Failed to parse ALLOWED_ORIGINS: {e}")

        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_content_gen_model(self) -> str:
        """Get content generation model based on feature flag."""
        if self.USE_LEGACY_MODELS:
            return "google/flan-t5-base"
        return self.CONTENT_GEN_MODEL_ID
    
    def get_embedding_model(self) -> str:
        """Get embedding model based on feature flag."""
        if self.USE_LEGACY_MODELS:
            return "sentence-transformers/all-MiniLM-L6-v2"
        return self.EMBEDDING_MODEL_ID
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension based on active model."""
        if self.USE_LEGACY_MODELS:
            return 384  # MiniLM dimension
        return self.EMBEDDING_DIMENSION  # E5-large dimension


# Global settings instance
settings = Settings()
