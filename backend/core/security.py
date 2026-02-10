"""Security utilities and configuration."""

import secrets
from datetime import UTC, datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings
from .exceptions import AuthenticationError

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bool(pwd_context.verify(plain_password, hashed_password))


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return str(pwd_context.hash(password))


def validate_password_strength(password: str) -> tuple[bool, str | None]:
    """
    Validate password meets security requirements.

    Returns:
        tuple: (is_valid, error_message)
    """
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        return (
            False,
            f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters",
        )

    if settings.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if settings.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if settings.PASSWORD_REQUIRE_DIGIT and not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"

    if settings.PASSWORD_REQUIRE_SPECIAL and not any(
        c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password
    ):
        return False, "Password must contain at least one special character"

    return True, None


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return str(encoded_jwt)


def create_refresh_token(data: dict) -> str:
    """
    Create JWT refresh token.

    Args:
        data: Data to encode in token

    Returns:
        Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return str(encoded_jwt)


def decode_token(token: str) -> dict:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token to decode

    Returns:
        Decoded token payload

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return dict(payload)
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {e!s}")


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(32)


def generate_session_id() -> str:
    """Generate a secure random session ID."""
    return secrets.token_urlsafe(16)
