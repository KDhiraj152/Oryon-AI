"""
V2 API - Authentication Routes
===============================

Endpoints for user authentication and management.
OPTIMIZED: Using async database sessions to avoid blocking event loop.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from ...database import get_async_db_session
from ...models import User
from ...schemas.auth import AuthResponse, Token, UserCreate, UserData, UserLogin
from ...utils.auth import (
    TokenData,
    create_tokens,
    get_current_user,
    get_password_hash,
    verify_password,
    verify_token,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


# ==================== Models ====================


class RefreshRequest(BaseModel):
    refresh_token: str


class UserUpdateRequest(BaseModel):
    name: str | None = None
    preferences: dict[str, Any] | None = None


# ==================== Helper Functions ====================


def user_to_response(user: User) -> UserData:
    """Convert User model to UserData response."""
    return UserData(
        id=str(user.id),
        email=user.email,
        name=user.full_name or user.email.split("@")[0],
        role=user.role,
        preferences={},
        created_at=user.created_at.isoformat() if user.created_at else "",
    )


# ==================== Endpoints ====================


@router.post("/auth/register", response_model=AuthResponse)
async def register_user(user_data: UserCreate):
    """Register new user with optimized async database."""
    try:
        async with get_async_db_session() as session:
            # Check existing user with async query
            result = await session.execute(
                select(User).where(User.email == user_data.email)
            )
            existing_user = result.scalar_one_or_none()
            if existing_user:
                raise HTTPException(status_code=400, detail="Email already registered")

            user_name = (
                user_data.name or user_data.full_name or user_data.email.split("@")[0]
            )
            # FIX: bcrypt is CPU-bound (~100ms) — offload to executor
            loop = asyncio.get_running_loop()
            hashed_pw = await loop.run_in_executor(
                None, get_password_hash, user_data.password
            )
            new_user = User(
                email=user_data.email,
                hashed_password=hashed_pw,
                full_name=user_name,
                is_active=True,
                role="user",
            )
            session.add(new_user)
            await session.flush()
            await session.refresh(new_user)

            tokens = create_tokens(
                user_id=str(new_user.id), email=new_user.email, role=new_user.role
            )

            return AuthResponse(
                user=user_to_response(new_user),
                access_token=tokens.access_token,
                refresh_token=tokens.refresh_token,
                token_type="bearer",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/auth/login", response_model=AuthResponse)
async def login_user(credentials: UserLogin):
    """Login with email and password using async database."""
    try:
        async with get_async_db_session() as session:
            result = await session.execute(
                select(User).where(User.email == credentials.email)
            )
            user = result.scalar_one_or_none()

            # FIX: bcrypt verify is CPU-bound (~100ms) — offload to executor
            loop = asyncio.get_running_loop()
            password_valid = await loop.run_in_executor(
                None, verify_password, credentials.password,
                user.hashed_password if user else "",
            )
            if not user or not password_valid:
                raise HTTPException(status_code=401, detail="Invalid credentials")

            if not user.is_active:
                raise HTTPException(status_code=403, detail="Account is disabled")

            tokens = create_tokens(
                user_id=str(user.id), email=user.email, role=user.role
            )

            return AuthResponse(
                user=user_to_response(user),
                access_token=tokens.access_token,
                refresh_token=tokens.refresh_token,
                token_type="bearer",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/auth/refresh", response_model=Token)
async def refresh_token(request: RefreshRequest):
    """Refresh access token using async database."""
    payload = verify_token(request.refresh_token, token_type="refresh")
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    async with get_async_db_session() as session:
        result = await session.execute(select(User).where(User.id == payload.user_id))
        user = result.scalar_one_or_none()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")

        tokens = create_tokens(user_id=str(user.id), email=user.email, role=user.role)
        return tokens


@router.get("/auth/me", response_model=UserData)
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information using async database."""
    async with get_async_db_session() as session:
        result = await session.execute(
            select(User).where(User.id == current_user.user_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user_to_response(user)


@router.put("/auth/me", response_model=UserData)
async def update_current_user(
    update_data: UserUpdateRequest, current_user: TokenData = Depends(get_current_user)
):
    """Update current user information using async database."""
    async with get_async_db_session() as session:
        result = await session.execute(
            select(User).where(User.id == current_user.user_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if update_data.name:
            user.full_name = update_data.name

        await session.flush()
        await session.refresh(user)
        return user_to_response(user)


@router.post("/auth/logout")
async def logout(current_user: TokenData = Depends(get_current_user)):
    """Logout current user (invalidate tokens)."""
    # In a production system, you'd add the token to a blacklist
    return {"message": "Logged out successfully"}
