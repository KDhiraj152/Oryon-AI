"""
Comprehensive Testing Framework

Issue: CODE-REVIEW-GPT #8 (HIGH)
Provides production-grade testing infrastructure with unit, integration, e2e, and performance tests.
"""

import asyncio
import os
import sys
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set test environment BEFORE importing modules
os.environ["TESTING"] = "true"
os.environ["ENVIRONMENT"] = "test"
os.environ["JWT_SECRET_KEY"] = (
    "test-secret-key-minimum-64-characters-required-for-testing-purposes-only"
)

# Disable rate limiting by default for tests (specific tests can enable it)
os.environ["RATE_LIMIT_ENABLED"] = "false"

# Use PostgreSQL test database (default works for most setups)
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/shiksha_setu_test",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

# Redis configuration (test database 15 to avoid interfering with dev/prod)
TEST_REDIS_URL = "redis://localhost:6379/15"
os.environ["REDIS_URL"] = TEST_REDIS_URL
os.environ["CELERY_BROKER_URL"] = TEST_REDIS_URL
os.environ["CELERY_RESULT_BACKEND"] = TEST_REDIS_URL

# Celery should run tasks synchronously during testing
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
os.environ["CELERY_TASK_EAGER_PROPAGATES"] = "true"

# Import after environment setup
from backend.api.main import app
from backend.core.config import settings
from backend.database import Base, get_db
from backend.models import ContentValidation, ProcessedContent, User
from backend.utils.auth import create_access_token, get_password_hash

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (database, external services)"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests (full workflows)")
    config.addinivalue_line("markers", "performance: Performance and load tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


# =============================================================================
# DATABASE FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    engine = create_engine(TEST_DATABASE_URL, echo=False)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def db_session(test_engine) -> Generator[Session, None, None]:
    """Create a new database session for each test."""
    testing_session_local = sessionmaker(bind=test_engine)
    session = testing_session_local()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@pytest.fixture
def clean_db(db_session):
    """Clean database before each test."""
    for table in reversed(Base.metadata.sorted_tables):
        db_session.execute(table.delete())
    db_session.commit()
    return


# =============================================================================
# API CLIENT FIXTURES
# =============================================================================


@pytest.fixture
def client(db_session):
    """Create FastAPI test client with test database."""

    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def authenticated_client(client, test_user, test_token):
    """Create authenticated test client."""
    client.headers.update({"Authorization": f"Bearer {test_token}"})
    return client


@pytest.fixture
def auth_headers(test_token) -> dict:
    """Create authentication headers for API requests."""
    return {"Authorization": f"Bearer {test_token}"}


# =============================================================================
# USER FIXTURES
# =============================================================================


@pytest.fixture
def test_user(db_session, clean_db) -> User:
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("TestPassword123!"),
        full_name="Test User",
        role="user",
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_token(test_user) -> str:
    """Create JWT token for test user."""
    return create_access_token({"sub": test_user.email, "user_id": str(test_user.id)})


@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory."""
    data_dir = Path("data/test_samples")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def celery_enable_logging():
    """Enable Celery logging in tests."""
    return True


@pytest.fixture(scope="session")
def celery_includes():
    """Celery task modules to include."""
    return ["backend.tasks.pipeline_tasks"]


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Use test database URL if not set
    if "DATABASE_URL" not in os.environ:
        # Default to PostgreSQL test database
        os.environ["DATABASE_URL"] = (
            "postgresql://user:password@localhost:5432/test_education_content"
        )

    # Disable SQL echo for cleaner test output
    if "SQL_ECHO" not in os.environ:
        os.environ["SQL_ECHO"] = "false"

    # Set test API keys if needed
    if "HUGGINGFACE_API_KEY" not in os.environ:
        # Use test key - tests will be skipped if real API is needed
        os.environ["HUGGINGFACE_API_KEY"] = "test_key_placeholder"

    return


@pytest.fixture
def clean_database():
    """Clean database between tests."""
    from backend.database import Base, engine

    # Drop all tables
    Base.metadata.drop_all(bind=engine)

    # Recreate all tables
    Base.metadata.create_all(bind=engine)

    yield

    # Cleanup after test
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_content():
    """Sample educational content for testing."""
    return {
        "original_text": "Photosynthesis is the process by which plants make food.",
        "subject": "Science",
        "language": "English",
    }
