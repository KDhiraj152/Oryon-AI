"""Authentication endpoints."""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel

from ...utils.auth import (
    Token,
    UserCreate,
    UserLogin,
    create_tokens,
    get_password_hash,
    verify_password,
    get_current_user,
    TokenData,
    verify_token
)
from ...database import get_db_session
from ...models import User
from ...monitoring import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    refresh_token: str = Body(..., embed=True)
):
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        payload = verify_token(refresh_token, token_type="refresh")
        
        if not payload:
            raise HTTPException(
                status_code=401,
                detail="Invalid refresh token"
            )
        
        # Check if user still exists and is active
        with get_db_session() as session:
            user = session.query(User).filter(
                User.id == payload.user_id
            ).first()
            
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=401,
                    detail="User not found or inactive"
                )
            
            # Create new tokens
            tokens = create_tokens(
                user_id=str(user.id),
                email=user.email,
                role=user.role
            )
            
            logger.info(f"Token refreshed for user: {user.email}")
            return tokens
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


@router.post("/register", response_model=Token)
async def register_user(user_data: UserCreate):
    """Register new user."""
    try:
        with get_db_session() as session:
            # Check if user exists
            existing_user = session.query(User).filter(
                User.email == user_data.email
            ).first()
            
            if existing_user:
                raise HTTPException(
                    status_code=400,
                    detail="Email already registered"
                )
            
            # Create user
            hashed_password = get_password_hash(user_data.password)
            new_user = User(
                email=user_data.email,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                organization=user_data.organization,
                role="user"
            )
            
            session.add(new_user)
            session.flush()  # Get ID without committing
            
            # Create tokens
            tokens = create_tokens(
                user_id=str(new_user.id),
                email=new_user.email,
                role=new_user.role
            )
            
            session.commit()  # Commit the user to database
            logger.info(f"New user registered: {user_data.email}")
            return tokens
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login", response_model=Token)
async def login_user(login_data: UserLogin):
    """Login user and return tokens."""
    try:
        with get_db_session() as session:
            # Find user
            user = session.query(User).filter(
                User.email == login_data.email
            ).first()
            
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect email or password"
                )
            
            # Verify password
            if not verify_password(login_data.password, user.hashed_password):
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect email or password"
                )
            
            # Check if active
            if not user.is_active:
                raise HTTPException(
                    status_code=403,
                    detail="Account is inactive"
                )
            
            # Update last login
            user.last_login = datetime.now(timezone.utc)
            
            # Create tokens
            tokens = create_tokens(
                user_id=str(user.id),
                email=user.email,
                role=user.role
            )
            
            logger.info(f"User logged in: {login_data.email}")
            return tokens
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me")
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information."""
    try:
        with get_db_session() as session:
            user = session.query(User).filter(
                User.id == current_user.user_id
            ).first()
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            return {
                "id": str(user.id),
                "email": user.email,
                "username": user.email.split('@')[0],  # Generate username from email
                "full_name": user.full_name,
                "organization": user.organization,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
