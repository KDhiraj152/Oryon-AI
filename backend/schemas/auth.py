"""Authentication and authorization schemas."""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes in seconds


class UserPreferences(BaseModel):
    """User preferences model."""

    default_target_language: str = "hi"
    ui_language: str = "en"
    theme: str = "system"
    auto_detect_source: bool = True
    include_audio: bool = False
    voice_type: str = "female"
    speech_speed: float = 1.0


class UserData(BaseModel):
    """User data for auth response."""

    id: str
    email: str
    name: str
    role: str | None = None
    avatar: str | None = None
    preferences: UserPreferences | dict = UserPreferences()
    created_at: str = ""


class AuthResponse(BaseModel):
    """Authentication response with tokens and user data."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserData


class TokenData(BaseModel):
    """JWT token payload data."""

    email: str | None = None
    user_id: str | None = None  # UUID as string
    role: str | None = None


class UserCreate(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str = Field(min_length=6, max_length=100)
    name: str | None = Field(min_length=2, max_length=100, default=None)
    full_name: str | None = Field(min_length=2, max_length=100, default=None)
    organization: str | None = None
    role: str | None = Field(default="user", pattern="^(user|educator|admin)$")

    def get_name(self) -> str:
        """Get the user's name from either field."""
        return self.name or self.full_name or ""


class UserLogin(BaseModel):
    """User login request."""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User information response."""

    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: str

    class Config:
        from_attributes = True


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str


__all__ = [
    "AuthResponse",
    "RefreshTokenRequest",
    "Token",
    "TokenData",
    "UserCreate",
    "UserData",
    "UserLogin",
    "UserPreferences",
    "UserResponse",
]
