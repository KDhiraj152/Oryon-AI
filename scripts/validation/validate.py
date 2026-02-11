#!/usr/bin/env python3
"""
ShikshaSetu Startup Validation Script
Checks all components before startup
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_ok(text):
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text):
    print(f"{RED}✗{RESET} {text}")

def print_warning(text):
    print(f"{YELLOW}⚠{RESET} {text}")

def check_python():
    """Check Python version and basic imports"""
    print_header("Python Environment Check")

    version = sys.version.split()[0]
    print_ok(f"Python {version}")

    # Check critical imports
    try:
        import sqlalchemy
        print_ok("SQLAlchemy installed")
    except ImportError:
        print_error("SQLAlchemy not installed - run: pip install -r requirements.txt")
        return False

    try:
        import fastapi
        print_ok("FastAPI installed")
    except ImportError:
        print_error("FastAPI not installed")
        return False

    try:
        import pydantic
        print_ok("Pydantic installed")
    except ImportError:
        print_error("Pydantic not installed")
        return False

    return True

def check_file_structure():
    """Check required directories and files exist"""
    print_header("File Structure Check")

    required_dirs = [
        "src",
        "backend/api",
        "backend/core",
        "backend/models.py",
        "frontend",
        "alembic",
        "data",
    ]

    for path in required_dirs:
        if Path(path).exists():
            print_ok(f"Found: {path}")
        else:
            print_error(f"Missing: {path}")
            return False

    return True

def check_env_file():
    """Check .env file exists and has required keys"""
    print_header("Environment Configuration Check")

    if not Path(".env").exists():
        print_error(".env file not found")
        return False

    print_ok(".env file exists")

    with open(".env") as f:
        content = f.read()

    required_keys = [
        "DATABASE_URL",
        "JWT_SECRET_KEY",
        "REDIS_URL",
    ]

    for key in required_keys:
        if key in content:
            print_ok(f"Found {key} in .env")
        else:
            print_warning(f"Missing {key} in .env (may cause issues)")

    return True

def check_database_connection():
    """Check database connection"""
    print_header("Database Connection Check")

    try:
        from backend.database import init_db, engine
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print_ok("Database connection successful")

        return True
    except Exception as e:
        print_error(f"Database connection failed: {e}")
        print_warning("Ensure DATABASE_URL in .env is correct")
        return False

def check_redis_connection():
    """Check Redis connection"""
    print_header("Redis Connection Check")

    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        print_ok("Redis connection successful")
        return True
    except Exception as e:
        print_warning(f"Redis connection failed: {e}")
        print_warning("Some features will be limited without Redis")
        return False

def check_python_syntax():
    """Check all Python files for syntax errors"""
    print_header("Python Syntax Check")

    import ast

    py_files = list(Path("src").rglob("*.py")) + list(Path("tests").rglob("*.py"))
    errors = []

    for py_file in py_files:
        try:
            with open(py_file, 'r') as f:
                ast.parse(f.read())
        except SyntaxError as e:
            errors.append(f"{py_file}: {e}")

    if errors:
        for error in errors:
            print_error(error)
        return False
    else:
        print_ok(f"All {len(py_files)} Python files have valid syntax")
        return True

def check_migrations():
    """Check Alembic migrations"""
    print_header("Database Migrations Check")

    alembic_versions = list(Path("alembic/versions").glob("*.py"))
    print_ok(f"Found {len(alembic_versions)} migrations")

    return True

def main():
    print(f"\n{BLUE}ShikshaSetu Startup Validation{RESET}\n")

    checks = [
        ("Python Environment", check_python),
        ("File Structure", check_file_structure),
        ("Environment Configuration", check_env_file),
        ("Python Syntax", check_python_syntax),
        ("Database Migrations", check_migrations),
        ("Database Connection", check_database_connection),
        ("Redis Connection", check_redis_connection),
    ]

    results = []

    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Check failed: {e}")
            results.append((name, False))

    # Summary
    print_header("Validation Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {status} - {name}")

    print(f"\n{BLUE}Result: {passed}/{total} checks passed{RESET}\n")

    if passed == total:
        print(f"{GREEN}✓ All checks passed! Ready to start.{RESET}\n")
        return 0
    else:
        print(f"{RED}✗ Some checks failed. Fix issues before starting.{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
