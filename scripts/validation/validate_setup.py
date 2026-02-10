#!/usr/bin/env python3
"""
ShikshaSetu Setup Validation Script
Validates environment configuration, dependencies, and connections
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_status(check: str, status: bool, message: str = ""):
    """Print check status with color"""
    status_str = "✓ PASS" if status else "✗ FAIL"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    msg = f" - {message}" if message else ""
    print(f"{color}{status_str}{reset} {check}{msg}")


def check_python_version() -> tuple[bool, str]:
    """Check Python version - 3.11 required for ML stack compatibility"""
    version = sys.version_info
    if version >= (3, 11) and version < (3, 14):
        is_optimal = version.minor == 11
        status = "(optimal)" if is_optimal else "(3.11 recommended)"
        return True, f"Python {version.major}.{version.minor}.{version.micro} {status}"
    elif version >= (3, 14):
        return (
            False,
            f"Python {version.major}.{version.minor}.{version.micro} (3.14+ not supported, use 3.11)",
        )
    return (
        False,
        f"Python {version.major}.{version.minor}.{version.micro} (requires 3.11)",
    )


def check_environment_file() -> tuple[bool, str]:
    """Check if .env file exists"""
    env_file = project_root / ".env"
    if env_file.exists():
        return True, ".env file found"
    return False, ".env file not found (copy from .env.example)"


def check_required_env_vars() -> list[tuple[str, bool, str]]:
    """Check required environment variables"""
    from dotenv import load_dotenv

    load_dotenv()

    required_vars = {
        "DATABASE_URL": "Database connection string",
        "JWT_SECRET_KEY": "JWT signing key",
        "REDIS_URL": "Redis connection string",
        "CELERY_BROKER_URL": "Celery broker URL",
    }

    results = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value and not value.endswith("_here") and "change" not in value.lower():
            results.append((var, True, f"{description} - configured"))
        else:
            results.append((var, False, f"{description} - NOT configured"))

    return results


def check_dependencies() -> list[tuple[str, bool, str]]:
    """Check key Python dependencies"""
    results = []

    dependencies = {
        "fastapi": "FastAPI web framework",
        "sqlalchemy": "SQLAlchemy ORM",
        "celery": "Celery task queue",
        "redis": "Redis client",
        "transformers": "HuggingFace transformers",
        "sentence_transformers": "Sentence transformers",
        "psycopg2": "PostgreSQL adapter",
    }

    for package, description in dependencies.items():
        try:
            __import__(package)
            results.append((package, True, description))
        except ImportError:
            results.append((package, False, f"{description} - NOT installed"))

    return results


def check_database_connection() -> tuple[bool, str]:
    """Check database connectivity"""
    try:
        from dotenv import load_dotenv
        from sqlalchemy import create_engine, text

        load_dotenv()

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return False, "DATABASE_URL not configured"

        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return True, "Database connection successful"
    except Exception as e:
        return False, f"Database connection failed: {str(e)[:50]}"


def check_redis_connection() -> tuple[bool, str]:
    """Check Redis connectivity"""
    try:
        import redis
        from dotenv import load_dotenv

        load_dotenv()

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(redis_url)
        r.ping()

        return True, "Redis connection successful"
    except Exception as e:
        return False, f"Redis connection failed: {str(e)[:50]}"


def check_directories() -> list[tuple[str, bool, str]]:
    """Check required directories"""
    required_dirs = ["data/uploads", "data/audio", "data/cache", "data/models", "logs"]

    results = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            results.append((dir_path, True, "exists"))
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            results.append((dir_path, True, "created"))

    return results


def check_frontend() -> tuple[bool, str]:
    """Check frontend setup"""
    frontend_dir = project_root / "frontend"
    node_modules = frontend_dir / "node_modules"

    if not frontend_dir.exists():
        return False, "Frontend directory not found"

    if not node_modules.exists():
        return (
            False,
            "Frontend dependencies not installed (run: cd frontend && npm install)",
        )

    return True, "Frontend dependencies installed"


def main():
    """Main validation function"""
    print("\n" + "=" * 70)
    print("  🎓 SHIKSHA SETU - SETUP VALIDATION")
    print("=" * 70)

    all_checks_passed = True

    # Python version
    print_header("Python Version")
    status, msg = check_python_version()
    print_status("Python Version", status, msg)
    all_checks_passed = all_checks_passed and status

    # Environment file
    print_header("Environment Configuration")
    status, msg = check_environment_file()
    print_status("Environment File", status, msg)
    all_checks_passed = all_checks_passed and status

    # Environment variables
    if status:
        print("\nRequired Environment Variables:")
        for var, check_status, msg in check_required_env_vars():
            print_status(var, check_status, msg)
            all_checks_passed = all_checks_passed and check_status

    # Dependencies
    print_header("Python Dependencies")
    for package, status, description in check_dependencies():
        print_status(package, status, description)
        all_checks_passed = all_checks_passed and status

    # Database connection
    print_header("Service Connectivity")
    status, msg = check_database_connection()
    print_status("PostgreSQL", status, msg)
    all_checks_passed = all_checks_passed and status

    # Redis connection
    status, msg = check_redis_connection()
    print_status("Redis", status, msg)
    # Don't fail if Redis is not available (optional for basic features)
    if not status:
        print("  ⚠️  WARNING: Redis is not available. Async tasks will not work.")

    # Directories
    print_header("Required Directories")
    for dir_path, status, msg in check_directories():
        print_status(dir_path, status, msg)

    # Frontend
    print_header("Frontend Setup")
    status, msg = check_frontend()
    print_status("Frontend", status, msg)
    # Don't fail if frontend is not set up
    if not status:
        print("  ℹ️  INFO: Frontend not required for backend-only development")

    # Final summary
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("  ✓ ALL CRITICAL CHECKS PASSED")
        print("  🚀 Your setup is ready! You can start the application.")
        print("\n  To start services:")
        print("    Backend:  uvicorn backend.api.main:app --reload")
        print("    Worker:   celery -A backend.tasks.celery_app worker --loglevel=info")
        print("    Frontend: cd frontend && npm run dev")
    else:
        print("  ✗ SOME CHECKS FAILED")
        print("  ⚠️  Please fix the issues above before starting the application.")
        print("\n  Common fixes:")
        print("    1. Copy .env.example to .env and configure it")
        print("    2. Install missing dependencies: pip install -r requirements.txt")
        print("    3. Start PostgreSQL and Redis services")
        print(
            "    4. Run database migrations: python -c 'from backend.database import init_db; init_db()'"
        )
    print("=" * 70 + "\n")

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
