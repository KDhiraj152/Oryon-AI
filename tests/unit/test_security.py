"""Unit tests for security utilities."""

import pytest

from backend.core.exceptions import AuthenticationError
from backend.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    get_password_hash,
    validate_password_strength,
    verify_password,
)


@pytest.mark.xfail(reason="bcrypt passlib compatibility issue with Python 3.13")
def test_password_hashing():
    """Test password hashing and verification."""
    password = "testpass123"  # Shorter password to avoid bcrypt 72-byte limit
    hashed = get_password_hash(password)

    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpass", hashed) is False


def test_password_validation():
    """Test password strength validation."""
    # Valid password with proper requirements
    is_valid, error = validate_password_strength("TestPass123!")
    assert is_valid is True
    assert error is None
    assert error is None

    # Too short
    is_valid, error = validate_password_strength("short")
    assert is_valid is False
    assert error is not None


def test_create_access_token():
    """Test JWT access token creation."""
    data = {"sub": "test@example.com", "user_id": "123"}
    token = create_access_token(data)

    assert isinstance(token, str)
    assert len(token) > 0

    # Decode and verify
    decoded = decode_token(token)
    assert decoded["sub"] == "test@example.com"
    assert decoded["user_id"] == "123"
    assert decoded["type"] == "access"


def test_create_refresh_token():
    """Test JWT refresh token creation."""
    data = {"sub": "test@example.com", "user_id": "123"}
    token = create_refresh_token(data)

    assert isinstance(token, str)
    assert len(token) > 0

    # Decode and verify
    decoded = decode_token(token)
    assert decoded["sub"] == "test@example.com"
    assert decoded["type"] == "refresh"


def test_decode_invalid_token():
    """Test decoding invalid token raises exception."""
    with pytest.raises(AuthenticationError):
        decode_token("invalid_token")


def test_generate_api_key():
    """Test API key generation."""
    key1 = generate_api_key()
    key2 = generate_api_key()

    assert isinstance(key1, str)
    assert len(key1) > 20
    assert key1 != key2  # Should be random
