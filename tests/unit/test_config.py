"""Unit tests for core configuration."""

import os

from backend.core.config import Settings, settings


def test_settings_initialization():
    """Test settings are initialized properly."""
    assert settings.APP_NAME == "Oryon AI Platform API"
    assert settings.APP_VERSION == "4.0.0"
    assert settings.API_PREFIX == "/api/v2"  # V2 API


def test_settings_directories_created():
    """Test that required directories are created on init."""
    assert settings.UPLOAD_DIR.exists()
    assert settings.MODEL_CACHE_DIR.exists()
    assert settings.LOG_DIR.exists()


def test_settings_environment_defaults():
    """Test default environment values."""
    assert os.getenv("HOST", "0.0.0.0") == settings.HOST
    assert int(os.getenv("PORT", "8000")) == settings.PORT


def test_settings_rate_limiting():
    """Test rate limiting configuration."""
    assert settings.RATE_LIMIT_PER_MINUTE >= 60  # At least 60/min
    assert settings.RATE_LIMIT_PER_HOUR >= 1000  # At least 1000/hour
    assert hasattr(settings, "RATE_LIMIT_STORAGE")  # Storage backend configured


def test_settings_password_requirements():
    """Test password requirements are configured."""
    assert settings.PASSWORD_MIN_LENGTH >= 8  # Minimum 8 chars
    assert hasattr(settings, "PASSWORD_REQUIRE_UPPERCASE")
    assert hasattr(settings, "PASSWORD_REQUIRE_LOWERCASE")
    assert hasattr(settings, "PASSWORD_REQUIRE_DIGIT")
