"""Unit tests for core configuration."""
import os
from src.core.config import Settings, settings


def test_settings_initialization():
    """Test settings are initialized properly."""
    assert settings.APP_NAME == "ShikshaSetu AI Education API"
    assert settings.APP_VERSION == "2.0.0"
    assert settings.API_V1_PREFIX == "/api/v1"


def test_settings_directories_created():
    """Test that required directories are created on init."""
    assert settings.UPLOAD_DIR.exists()
    assert settings.MODEL_CACHE_DIR.exists()
    assert settings.LOG_DIR.exists()


def test_settings_environment_defaults():
    """Test default environment values."""
    assert settings.HOST == os.getenv("HOST", "0.0.0.0")
    assert settings.PORT == int(os.getenv("PORT", "8000"))
    

def test_settings_rate_limiting():
    """Test rate limiting is configured for testing."""
    assert settings.RATE_LIMIT_PER_MINUTE >= 1000  # High for testing
    assert settings.RATE_LIMIT_PER_HOUR >= 10000  # High for testing


def test_settings_password_requirements():
    """Test password requirements are relaxed for testing."""
    assert settings.MIN_PASSWORD_LENGTH == 8
    assert settings.REQUIRE_SPECIAL_CHARS == False
    assert settings.REQUIRE_NUMBERS == False
    assert settings.REQUIRE_UPPERCASE == False
