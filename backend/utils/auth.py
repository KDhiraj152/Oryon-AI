"""JWT Authentication system for API security."""

import hashlib
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt

from ..core.config import get_settings

# Import schemas for backward compatibility
from ..schemas.auth import Token, TokenData, UserCreate, UserLogin, UserResponse

logger = logging.getLogger(__name__)

# Use global settings instance (already created in config.py)
settings = get_settings()

# Configuration - MUST be from environment, no fallback generation
SECRET_KEY = settings.SECRET_KEY
if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable not set. "
        "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
    )

if len(SECRET_KEY) < 64:
    logger.error(
        f"JWT_SECRET_KEY too short: {len(SECRET_KEY)} chars (minimum 64 required)"
    )
    if settings.ENVIRONMENT == "production":
        raise RuntimeError(
            "JWT_SECRET_KEY must be at least 64 characters in production"
        )
    else:
        logger.warning("JWT_SECRET_KEY too short for production use")

ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS
BCRYPT_ROUNDS = settings.BCRYPT_ROUNDS

# Re-export schemas for backward compatibility
__all__ = [
    "Token",
    "TokenData",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "create_access_token",
    "create_refresh_token",
    "create_tokens",
    "get_current_user",
    "get_password_hash",
    "verify_password",
    "verify_token",
]

# Password utilities

def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password meets security policy.

    Returns:
        (is_valid, error_message)
    """
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        return (
            False,
            f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters",
        )

    if settings.PASSWORD_REQUIRE_UPPERCASE and not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"

    if settings.PASSWORD_REQUIRE_LOWERCASE and not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"

    if settings.PASSWORD_REQUIRE_DIGIT and not re.search(r"\d", password):
        return False, "Password must contain at least one digit"

    if settings.PASSWORD_REQUIRE_SPECIAL and not re.search(
        r'[!@#$%^&*(),.?":{}|<>]', password
    ):
        return False, "Password must contain at least one special character"

    return True, ""

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        # Truncate to 72 bytes for bcrypt compatibility
        password_bytes = plain_password.encode("utf-8")[:72]

        # Ensure hashed_password is bytes
        if isinstance(hashed_password, str):
            hashed_bytes = hashed_password.encode("utf-8")
        else:
            hashed_bytes = hashed_password

        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except (ValueError, RuntimeError) as e:
        logger.error("Password verification error: %s", e)
        return False

def get_password_hash(password: str) -> str:
    """Hash password using bcrypt with configurable work factor."""
    # Validate password strength before hashing
    is_valid, error_msg = validate_password_strength(password)
    if not is_valid:
        raise ValueError(error_msg)

    # Truncate to 72 bytes for bcrypt compatibility
    password_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")

# Token utilities

def generate_jti() -> str:
    """Generate unique JWT ID for token tracking."""
    return str(uuid.uuid4())

def generate_device_fingerprint(user_agent: str, ip_address: str) -> str:
    """Generate device fingerprint from user agent and IP."""
    fingerprint_data = f"{user_agent}:{ip_address}"
    return hashlib.sha256(fingerprint_data.encode()).hexdigest()

def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
    jti: str | None = None,
) -> str:
    """
    Create JWT access token with JTI for blacklist tracking.

    Args:
        data: Token payload data
        expires_delta: Token expiration time
        jti: JWT ID (generated if not provided)

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    token_jti = jti or generate_jti()

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "access",
            "jti": token_jti,
        }
    )

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return str(encoded_jwt)

def create_refresh_token(
    data: dict[str, Any], parent_jti: str | None = None, rotation_count: int = 0
) -> tuple[str, str, datetime]:
    """
    Create JWT refresh token with rotation tracking.

    Args:
        data: Token payload data
        parent_jti: JTI of token being rotated
        rotation_count: Number of times token has been rotated

    Returns:
        Tuple of (encoded token, jti, expiration datetime)
    """
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    token_jti = generate_jti()

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "refresh",
            "jti": token_jti,
            "parent_jti": parent_jti,
            "rotation_count": rotation_count,
        }
    )

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, token_jti, expire

def verify_token(token: str, token_type: str = "access") -> TokenData | None:
    """
    Verify and decode JWT token.

    Args:
        token: JWT token to verify
        token_type: Expected token type ('access' or 'refresh')

    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Verify token type
        if payload.get("type") != token_type:
            logger.warning(
                f"Invalid token type: expected {token_type}, got {payload.get('type')}"
            )
            return None

        # Get user_id from payload, fallback to 'sub' for backwards compatibility
        user_id: str = payload.get("user_id") or payload.get("sub")
        email: str = payload.get("email") or payload.get(
            "sub"
        )  # 'sub' often contains email
        role: str = payload.get("role", "user")

        if user_id is None:
            return None

        return TokenData(user_id=user_id, email=email, role=role)

    except JWTError as e:
        logger.error("JWT verification failed: %s", e)
        return None

def create_tokens(user_id: str, email: str, role: str = "user") -> Token:
    """
    Create access and refresh tokens for a user.

    Args:
        user_id: User ID
        email: User email
        role: User role

    Returns:
        Token object with access and refresh tokens
    """
    token_data = {"sub": user_id, "email": email, "role": role}

    access_token = create_access_token(token_data)
    refresh_token_encoded, _, _ = create_refresh_token(token_data)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token_encoded,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

def refresh_access_token(refresh_token: str) -> str | None:
    """
    Create new access token from refresh token.

    Args:
        refresh_token: Valid refresh token

    Returns:
        New access token or None if invalid
    """
    token_data = verify_token(refresh_token, token_type="refresh")

    if not token_data:
        return None

    new_access_token = create_access_token(
        {"sub": token_data.user_id, "email": token_data.email, "role": token_data.role}
    )

    return new_access_token

# FastAPI dependencies
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """
    FastAPI dependency to get current authenticated user.

    Args:
        credentials: HTTP bearer token

    Returns:
        TokenData for authenticated user

    Raises:
        HTTPException: If token is invalid
    """
    token = credentials.credentials
    token_data = verify_token(token)

    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token_data

def get_current_active_user(
    current_user: TokenData = Depends(get_current_user),
) -> TokenData:
    """
    FastAPI dependency to ensure user is active.

    Args:
        current_user: Current user token data

    Returns:
        TokenData if user is active

    Raises:
        HTTPException: If user is inactive
    """
    # In production, check database for user.is_active
    return current_user

def require_admin(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """
    FastAPI dependency to require admin role.

    Args:
        current_user: Current user token data

    Returns:
        TokenData if user is admin

    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )

    return current_user

def get_current_user_ws(token: str) -> TokenData:
    """
    Authenticate WebSocket connection using JWT token.

    Args:
        token: JWT token from query parameter

    Returns:
        TokenData if valid

    Raises:
        ValueError: If token invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("role", "user")

        if user_id is None:
            raise ValueError("Invalid token: missing user_id")

        return TokenData(user_id=user_id, email=email, role=role)

    except JWTError as e:
        raise ValueError(f"Token validation failed: {e}")

# Export
__all__ = [
    "Token",
    "TokenData",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "create_tokens",
    "get_current_active_user",
    "get_current_user",
    "get_current_user_ws",
    "get_password_hash",
    "refresh_access_token",
    "require_admin",
    "verify_password",
    "verify_token",
]
