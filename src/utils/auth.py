"""JWT Authentication system for API security."""
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
import logging

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY or SECRET_KEY == "your-secret-key-change-in-production-minimum-32-chars":
    logger.warning("JWT_SECRET_KEY not set or using default. Generating secure key.")
    SECRET_KEY = secrets.token_urlsafe(32)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


# Models
class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = "user"


class UserCreate(BaseModel):
    """User registration model."""
    email: EmailStr
    password: str = Field(min_length=8, max_length=72, description="Password (max 72 characters for bcrypt)")
    full_name: Optional[str] = None
    organization: Optional[str] = None


class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr
    password: str


class User(BaseModel):
    """User model."""
    id: str
    email: str
    full_name: Optional[str] = None
    organization: Optional[str] = None
    role: str = "user"
    is_active: bool = True
    created_at: datetime


# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        # Truncate to 72 bytes for bcrypt compatibility
        password_bytes = plain_password.encode('utf-8')[:72]
        
        # Ensure hashed_password is bytes
        if isinstance(hashed_password, str):
            hashed_bytes = hashed_password.encode('utf-8')
        else:
            hashed_bytes = hashed_password
            
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Hash a password."""
    # Truncate to 72 bytes for bcrypt compatibility
    password_bytes = password.encode('utf-8')[:72]
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode('utf-8')


# Token utilities
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Token payload data
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create JWT refresh token.
    
    Args:
        data: Token payload data
        
    Returns:
        Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
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
            logger.warning(f"Invalid token type: expected {token_type}, got {payload.get('type')}")
            return None
        
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("role", "user")
        
        if user_id is None:
            return None
        
        return TokenData(user_id=user_id, email=email, role=role)
        
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
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
    token_data = {
        "sub": user_id,
        "email": email,
        "role": role
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


def refresh_access_token(refresh_token: str) -> Optional[str]:
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
    
    new_access_token = create_access_token({
        "sub": token_data.user_id,
        "email": token_data.email,
        "role": token_data.role
    })
    
    return new_access_token


# FastAPI dependencies
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
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


def get_current_active_user(current_user: TokenData = Depends(get_current_user)) -> TokenData:
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
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


# Export
__all__ = [
    'Token',
    'TokenData',
    'UserCreate',
    'UserLogin',
    'User',
    'verify_password',
    'get_password_hash',
    'create_tokens',
    'verify_token',
    'refresh_access_token',
    'get_current_user',
    'get_current_active_user',
    'require_admin'
]
