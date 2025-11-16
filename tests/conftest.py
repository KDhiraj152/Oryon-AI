"""
Pytest configuration and fixtures for integration tests.
"""
import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test configuration constants
TEST_REDIS_URL = 'redis://localhost:6379/15'

# Set test environment
os.environ['TESTING'] = 'true'
os.environ['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'postgresql://test_user:test_password@localhost:5432/test_db')
os.environ['REDIS_URL'] = TEST_REDIS_URL
os.environ['CELERY_BROKER_URL'] = TEST_REDIS_URL
os.environ['CELERY_RESULT_BACKEND'] = TEST_REDIS_URL


@pytest.fixture(scope='session')
def test_data_dir():
    """Create test data directory."""
    data_dir = Path('data/test_samples')
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(scope='session')
def celery_enable_logging():
    """Enable Celery logging in tests."""
    return True


@pytest.fixture(scope='session')
def celery_includes():
    """Celery task modules to include."""
    return ['src.tasks.pipeline_tasks']


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Use test database URL if not set
    if 'DATABASE_URL' not in os.environ:
        # Default to PostgreSQL test database
        os.environ['DATABASE_URL'] = 'postgresql://user:password@localhost:5432/test_education_content'
    
    # Disable SQL echo for cleaner test output
    if 'SQL_ECHO' not in os.environ:
        os.environ['SQL_ECHO'] = 'false'
    
    # Set test API keys if needed
    if 'HUGGINGFACE_API_KEY' not in os.environ:
        # Use test key - tests will be skipped if real API is needed
        os.environ['HUGGINGFACE_API_KEY'] = 'test_key_placeholder'
    
    yield


@pytest.fixture(scope="function")
def clean_database():
    """Clean database between tests."""
    from src.database import SessionLocal
    
    db = get_db()
    
    # Drop and recreate tables for each test
    db.drop_tables()
    db.create_tables()
    
    yield
    
    # Cleanup after test
    db.close_session()


@pytest.fixture
def sample_content():
    """Sample educational content for testing."""
    return {
        "original_text": "Photosynthesis is the process by which plants make food.",
        "grade_level": 6,
        "subject": "Science",
        "language": "English"
    }
