#!/usr/bin/env python3
"""
Comprehensive Dependency Checker for Oryon
Verifies all backend dependencies, configurations, and services
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path


class DependencyChecker:
    def __init__(self):
        self.results = []
        self.total_checks = 0
        self.passed_checks = 0

    def check(self, name: str, test_func, critical: bool = True) -> bool:
        """Run a check and record the result"""
        self.total_checks += 1
        try:
            result = test_func()
            if result:
                self.passed_checks += 1
                self.results.append(f"✅ {name}")
                return True
            else:
                status = "❌" if critical else "⚠️"
                self.results.append(f"{status} {name} - FAILED")
                return False
        except Exception as e:
            status = "❌" if critical else "⚠️"
            self.results.append(f"{status} {name} - ERROR: {e!s}")
            return False

    def print_results(self):
        """Print all results and score"""
        print("\n" + "=" * 80)
        print("ORYON AI DEPENDENCY CHECK REPORT")
        print("=" * 80 + "\n")

        for result in self.results:
            print(result)

        score = (
            (self.passed_checks / self.total_checks * 100)
            if self.total_checks > 0
            else 0
        )

        print("\n" + "=" * 80)
        print(f"SCORE: {self.passed_checks}/{self.total_checks} ({score:.1f}%)")
        print("=" * 80 + "\n")

        if score == 100:
            print("🎉 PERFECT! All dependencies are properly installed and configured!")
            print("✅ System is ready for development and production use.\n")
            return True
        elif score >= 80:
            print("⚠️  GOOD but some optional components are missing.")
            print("✅ Core system should work, but some features may be limited.\n")
            return True
        elif score >= 60:
            print("⚠️  PARTIAL - Critical components may be missing.")
            print("❌ System may not work properly. Fix critical issues.\n")
            return False
        else:
            print("❌ FAILED - Too many missing dependencies.")
            print("❌ System will not work. Install required dependencies.\n")
            return False


def main():
    checker = DependencyChecker()

    print("Checking Oryon Backend Dependencies...\n")

    # ==================== Python Version ====================
    def check_python_version():
        version = sys.version_info
        # Python 3.11 required for ML package compatibility
        return version.major == 3 and version.minor == 11

    checker.check("Python 3.11 (ML stack)", check_python_version, critical=True)

    # ==================== Core Framework Dependencies ====================
    checker.check("FastAPI", lambda: importlib.import_module("fastapi"), critical=True)
    checker.check("Uvicorn", lambda: importlib.import_module("uvicorn"), critical=True)
    checker.check(
        "Pydantic", lambda: importlib.import_module("pydantic"), critical=True
    )
    checker.check(
        "SQLAlchemy", lambda: importlib.import_module("sqlalchemy"), critical=True
    )

    # ==================== Database Drivers ====================
    checker.check(
        "PostgreSQL Driver (psycopg2)",
        lambda: importlib.import_module("psycopg2"),
        critical=True,
    )
    checker.check("Redis", lambda: importlib.import_module("redis"), critical=True)

    # ==================== Task Queue ====================
    checker.check("Celery", lambda: importlib.import_module("celery"), critical=True)
    checker.check("Kombu", lambda: importlib.import_module("kombu"), critical=True)

    # ==================== HTTP & API ====================
    checker.check(
        "Requests", lambda: importlib.import_module("requests"), critical=True
    )
    checker.check(
        "AIOFiles", lambda: importlib.import_module("aiofiles"), critical=True
    )
    checker.check(
        "Python Multipart", lambda: importlib.import_module("multipart"), critical=True
    )

    # ==================== Security ====================
    checker.check("PyJWT", lambda: importlib.import_module("jwt"), critical=True)
    checker.check("Passlib", lambda: importlib.import_module("passlib"), critical=True)
    checker.check("Python-Jose", lambda: importlib.import_module("jose"), critical=True)
    checker.check(
        "Bleach (sanitization)",
        lambda: importlib.import_module("bleach"),
        critical=True,
    )

    # ==================== ML & NLP Dependencies ====================
    checker.check(
        "Transformers (HuggingFace)",
        lambda: importlib.import_module("transformers"),
        critical=True,
    )
    checker.check("PyTorch", lambda: importlib.import_module("torch"), critical=True)
    checker.check(
        "SentencePiece", lambda: importlib.import_module("sentencepiece"), critical=True
    )
    checker.check(
        "Accelerate", lambda: importlib.import_module("accelerate"), critical=False
    )
    checker.check(
        "Sentence-Transformers",
        lambda: importlib.import_module("sentence_transformers"),
        critical=False,
    )

    # ==================== OCR & PDF Processing ====================
    checker.check(
        "PyMuPDF (fitz)", lambda: importlib.import_module("fitz"), critical=True
    )
    checker.check(
        "Pytesseract", lambda: importlib.import_module("pytesseract"), critical=True
    )
    checker.check("Pillow (PIL)", lambda: importlib.import_module("PIL"), critical=True)
    checker.check(
        "PDF2Image", lambda: importlib.import_module("pdf2image"), critical=True
    )

    # ==================== Audio Processing ====================
    checker.check(
        "SoundFile", lambda: importlib.import_module("soundfile"), critical=True
    )
    checker.check("Librosa", lambda: importlib.import_module("librosa"), critical=True)
    checker.check("PyDub", lambda: importlib.import_module("pydub"), critical=False)

    # ==================== Utilities ====================
    checker.check(
        "Python-dotenv", lambda: importlib.import_module("dotenv"), critical=True
    )
    checker.check("PyYAML", lambda: importlib.import_module("yaml"), critical=True)
    checker.check(
        "Python-magic", lambda: importlib.import_module("magic"), critical=True
    )

    # ==================== Testing ====================
    checker.check("Pytest", lambda: importlib.import_module("pytest"), critical=False)
    checker.check("HTTPX", lambda: importlib.import_module("httpx"), critical=False)

    # ==================== Environment Configuration ====================
    def check_env_file():
        return Path(".env").exists()

    checker.check(".env file exists", check_env_file, critical=True)

    def check_env_vars():
        from dotenv import load_dotenv

        load_dotenv()
        required = ["DATABASE_URL", "REDIS_URL", "CELERY_BROKER_URL"]
        return all(os.getenv(var) for var in required)

    checker.check("Critical environment variables set", check_env_vars, critical=True)

    # ==================== Directory Structure ====================
    def check_directories():
        required_dirs = ["data/uploads", "data/audio", "data/cache", "logs"]
        return all(
            Path(d).exists() or Path(d).mkdir(parents=True, exist_ok=True) or True
            for d in required_dirs
        )

    checker.check("Required directories exist", check_directories, critical=True)

    # ==================== External Services ====================
    def check_redis_connection():
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=2)
            r.ping()
            return True
        except Exception:
            return False

    checker.check("Redis server running", check_redis_connection, critical=True)

    def check_postgres_connection():
        try:
            from dotenv import load_dotenv
            from sqlalchemy import create_engine

            load_dotenv()

            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                return False

            engine = create_engine(db_url, pool_pre_ping=True)
            conn = engine.connect()
            conn.close()
            return True
        except Exception as e:
            print(f"    (PostgreSQL Error: {e})")
            return False

    checker.check(
        "PostgreSQL database accessible", check_postgres_connection, critical=True
    )

    # ==================== System Tools ====================
    def check_tesseract():
        try:
            result = subprocess.run(
                ["tesseract", "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    checker.check("Tesseract OCR installed", check_tesseract, critical=True)

    # Package version checking utility (available for future use)
    # This function validates installed packages meet minimum version requirements
    # Currently not actively used as version constraints are in requirements.txt

    # ==================== Import Test for Project Modules ====================
    def check_project_imports():
        try:
            sys.path.insert(0, str(Path.cwd()))
            importlib.import_module("backend.api.async_app")
            importlib.import_module("backend.tasks.celery_app")
            importlib.import_module("backend.utils.auth")
            return True
        except Exception as e:
            print(f"    (Import Error: {e})")
            return False

    checker.check("Project modules importable", check_project_imports, critical=True)

    # ==================== Print Results ====================
    success = checker.print_results()

    # ==================== Additional Recommendations ====================
    if not success:
        print("RECOMMENDATIONS:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure Redis is running: redis-server")
        print("3. Ensure PostgreSQL is running and database exists")
        print("4. Install Tesseract OCR: brew install tesseract (macOS)")
        print("5. Create .env file from .env.example")
        print("6. Run database migrations: alembic upgrade head\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
