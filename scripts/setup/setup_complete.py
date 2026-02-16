#!/usr/bin/env python3
"""Complete setup script for Oryon production environment."""

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SetupManager:
    """Manage complete system setup."""

    def __init__(self):
        self.root_dir = Path(__file__).parent.absolute()
        self.errors = []

    def run_command(self, cmd: list, description: str, critical: bool = True) -> bool:
        """Run shell command with error handling."""
        logger.info(f"⏳ {description}...")
        try:
            result = subprocess.run(
                cmd, cwd=self.root_dir, check=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info(f"✅ {description} completed")
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            error_msg = f"❌ {description} failed: {e.stderr}"
            if critical:
                self.errors.append(error_msg)
                logger.error(error_msg)
            else:
                logger.warning(error_msg)
            return False

    def check_python_version(self) -> bool:
        """Check Python version - 3.11 required for ML stack."""
        logger.info("🔍 Checking Python version...")
        version = sys.version_info
        if version.major == 3 and version.minor == 11:
            logger.info(
                f"✅ Python {version.major}.{version.minor}.{version.micro} (optimal for ML stack)"
            )
            return True
        elif version.major >= 3 and version.minor >= 11 and version.minor <= 13:
            logger.warning(
                f"⚠️ Python {version.major}.{version.minor}.{version.micro} (3.11 recommended for best compatibility)"
            )
            return True
        else:
            error = f"❌ Python 3.11 required, found {version.major}.{version.minor}"
            self.errors.append(error)
            logger.error(error)
            return False

    def install_system_dependencies(self) -> bool:
        """Install system-level dependencies."""
        logger.info("📦 Installing system dependencies...")

        # Detect OS
        import platform

        system = platform.system()

        if system == "Darwin":  # macOS
            commands = [
                (["brew", "install", "tesseract"], "Tesseract OCR", False),
                (
                    ["brew", "install", "tesseract-lang"],
                    "Tesseract language packs",
                    False,
                ),
                (["brew", "install", "ffmpeg"], "FFmpeg", False),
                (["brew", "install", "redis"], "Redis", False),
                (["brew", "install", "postgresql@15"], "PostgreSQL 15", False),
            ]
        elif system == "Linux":
            commands = [
                (["sudo", "apt-get", "update"], "Update package list", False),
                (
                    [
                        "sudo",
                        "apt-get",
                        "install",
                        "-y",
                        "tesseract-ocr",
                        "tesseract-ocr-hin",
                        "tesseract-ocr-eng",
                    ],
                    "Tesseract OCR",
                    False,
                ),
                (["sudo", "apt-get", "install", "-y", "ffmpeg"], "FFmpeg", False),
                (["sudo", "apt-get", "install", "-y", "redis-server"], "Redis", False),
                (
                    ["sudo", "apt-get", "install", "-y", "postgresql-15"],
                    "PostgreSQL",
                    False,
                ),
            ]
        else:
            logger.warning(
                f"⚠️  Unsupported OS: {system}. Please install dependencies manually."
            )
            return True

        success = True
        for cmd, desc, critical in commands:
            if not self.run_command(cmd, desc, critical):
                success = False

        return success

    def create_directories(self) -> bool:
        """Create required directories."""
        logger.info("📁 Creating directories...")

        directories = [
            "data/uploads",
            "data/audio",
            "data/cache",
            "data/captions",
            "data/models",
            "data/test_samples",
            "logs",
        ]

        for dir_path in directories:
            full_path = self.root_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

        logger.info("✅ Directories created")
        return True

    def install_python_dependencies(self) -> bool:
        """Install Python packages."""
        logger.info("🐍 Installing Python dependencies...")

        # Install production dependencies
        if not self.run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            "Upgrade pip",
            critical=True,
        ):
            return False

        if not self.run_command(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            "Install production dependencies",
            critical=True,
        ):
            return False

        # Install development dependencies
        self.run_command(
            [sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"],
            "Install development dependencies",
            critical=False,
        )

        return True

    def setup_database(self) -> bool:
        """Setup PostgreSQL database."""
        logger.info("🗄️  Setting up database...")

        # Create database
        self.run_command(
            ["createdb", "-h", "localhost", "-U", "postgres", "oryon"],
            "Create database",
            critical=False,
        )

        # Run migrations
        if not self.run_command(
            ["alembic", "upgrade", "head"], "Run database migrations", critical=True
        ):
            return False

        logger.info("✅ Database setup complete")
        return True

    def download_ml_models(self) -> bool:
        """Download required ML models."""
        logger.info("🤖 Downloading ML models (this may take a while)...")

        # Create model download script
        download_script = """
import logging
from backend.utils.model_manager import get_model_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

manager = get_model_manager()

# Download optimal 2025 model stack
try:
    logger.info("Downloading Qwen3-8B (LLM via MLX)...")
    manager.load_text_model("mlx-community/Qwen3-8B-4bit")

    logger.info("Downloading BGE-M3 (Embeddings)...")
    manager.load_embedding_model("BAAI/bge-m3")

    logger.info("Downloading Whisper Large V3 Turbo (STT)...")
    manager.load_whisper("large-v3-turbo")

    logger.info("Downloading MMS-TTS Hindi (TTS)...")
    manager.load_tts_model("facebook/mms-tts-hin")

    logger.info("All models downloaded successfully!")
except Exception as e:
    logger.error(f"Model download failed: {e}")
    exit(1)
"""

        script_path = self.root_dir / "scripts" / "download_models_temp.py"
        script_path.parent.mkdir(exist_ok=True)
        script_path.write_text(download_script)

        result = self.run_command(
            [sys.executable, str(script_path)], "Download ML models", critical=False
        )

        script_path.unlink()
        return result

    def setup_env_file(self) -> bool:
        """Create .env file if not exists."""
        logger.info("🔐 Setting up environment file...")

        env_file = self.root_dir / ".env"

        if env_file.exists():
            logger.info("⚠️  .env file already exists, skipping")
            return False

        env_template = """# Oryon Environment Configuration

# Database
DATABASE_URL=postgresql://postgres:CHANGE_ME@localhost:5432/oryon
POSTGRES_USER=postgres
POSTGRES_PASSWORD=CHANGE_ME
POSTGRES_DB=oryon
POSTGRES_PORT=5432

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# JWT Authentication
JWT_SECRET_KEY=your-secret-key-change-in-production-minimum-32-characters-long

# API Configuration
FASTAPI_PORT=8000
FLASK_PORT=5000

# Hugging Face (optional - for model downloads)
HUGGINGFACE_API_KEY=

# Model Configuration (Optimal 2025 Stack)
SIMPLIFICATION_MODEL_ID=mlx-community/Qwen3-8B-4bit
TRANSLATION_MODEL_ID=ai4bharat/indictrans2-en-indic-1B
VALIDATION_MODEL_ID=mlx-community/Qwen3-8B-4bit
EMBEDDING_MODEL_ID=BAAI/bge-m3
RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3
TTS_MODEL_ID=facebook/mms-tts-hin
STT_MODEL_ID=openai/whisper-large-v3-turbo
OCR_MODEL_ID=ucaslcl/GOT-OCR2_0

# Storage
AUDIO_STORAGE_DIR=data/audio
UPLOAD_DIR=data/uploads
CACHE_DIR=data/cache
"""

        env_file.write_text(env_template)
        logger.info("✅ .env file created - PLEASE UPDATE WITH YOUR CREDENTIALS")
        return True

    def run_tests(self) -> bool:
        """Run test suite to verify installation."""
        logger.info("🧪 Running tests...")

        return self.run_command(
            [sys.executable, "-m", "pytest", "tests/", "-v", "-m", "not slow"],
            "Run fast tests",
            critical=False,
        )

    def display_next_steps(self):
        """Display next steps for user."""
        print("\n" + "=" * 60)
        print("🎉 Setup Complete!")
        print("=" * 60)

        if self.errors:
            print("\n⚠️  Some non-critical errors occurred:")
            for error in self.errors:
                print(f"  - {error}")

        print("\n📋 Next Steps:")
        print("1. Update .env file with your database credentials")
        print("2. Start Redis: redis-server")
        print("3. Start Celery worker: make celery-worker")
        print("4. Start API server: make api")
        print("\n📚 Useful Commands:")
        print("  make help          - Show all available commands")
        print("  make test          - Run test suite")
        print("  make docker-up     - Start with Docker")
        print("  make lint          - Check code quality")
        print("\n📖 Documentation:")
        print("  README.md          - Complete setup guide")
        print("  /docs endpoint     - API reference (when running)")
        print("\n🚀 Access Points:")
        print("  API:               http://localhost:8000")
        print("  API Docs:          http://localhost:8000/docs")
        print("  Celery Flower:     http://localhost:5555 (after 'make celery-flower')")
        print("\n" + "=" * 60)

    def run_full_setup(self):
        """Run complete setup process."""
        print("🚀 Oryon Production Setup")
        print("=" * 60)

        steps = [
            (self.check_python_version, "Python version check", True),
            (self.create_directories, "Directory creation", True),
            (self.install_system_dependencies, "System dependencies", False),
            (self.install_python_dependencies, "Python dependencies", True),
            (self.setup_env_file, "Environment configuration", True),
            (self.setup_database, "Database setup", True),
            (self.download_ml_models, "ML models download", False),
            (self.run_tests, "Test suite", False),
        ]

        for step_func, description, critical in steps:
            success = step_func()
            if not success and critical:
                logger.error(f"❌ Critical step '{description}' failed. Aborting.")
                sys.exit(1)

        self.display_next_steps()


if __name__ == "__main__":
    manager = SetupManager()
    manager.run_full_setup()
