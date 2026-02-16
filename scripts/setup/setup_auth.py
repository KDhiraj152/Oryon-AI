#!/usr/bin/env python3
"""Setup authentication system and create test users."""

import os
import secrets
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.config import settings
from backend.database import get_db_session
from backend.models import User
from backend.utils.auth import create_tokens, get_password_hash


def generate_jwt_secret():
    """Generate secure JWT secret key."""
    return secrets.token_urlsafe(64)


def setup_jwt_secret():
    """Setup or verify JWT secret key."""
    env_file = Path(".env")

    if settings.SECRET_KEY and len(settings.SECRET_KEY) >= 64:
        print(
            f"✅ JWT_SECRET_KEY already configured ({len(settings.SECRET_KEY)} chars)"
        )
        return settings.SECRET_KEY

    # Generate new secret
    new_secret = generate_jwt_secret()

    print(f"🔑 Generated new JWT_SECRET_KEY ({len(new_secret)} chars)")

    # Update or create .env file
    if env_file.exists():
        content = env_file.read_text()
        if "JWT_SECRET_KEY=" in content:
            # Replace existing
            lines = content.split("\n")
            new_lines = []
            for line in lines:
                if line.startswith("JWT_SECRET_KEY="):
                    new_lines.append(f"JWT_SECRET_KEY={new_secret}")
                else:
                    new_lines.append(line)
            env_file.write_text("\n".join(new_lines))
        else:
            # Append
            with open(env_file, "a") as f:
                f.write(f"\nJWT_SECRET_KEY={new_secret}\n")
    else:
        # Create new
        env_file.write_text(f"JWT_SECRET_KEY={new_secret}\n")

    print(f"✅ JWT_SECRET_KEY saved to {env_file}")
    print("⚠️  Please restart the backend server for changes to take effect")

    return new_secret


def create_test_user(email: str, password: str, full_name: str, role: str = "user"):
    """Create a test user and return user data dict."""
    try:
        with get_db_session() as session:
            # Check if user exists
            existing = session.query(User).filter(User.email == email).first()
            if existing:
                user_id = str(existing.id)
                user_email = existing.email
                user_role = existing.role
                print(f"⚠️  User {email} already exists (ID: {user_id})")
                return {"id": user_id, "email": user_email, "role": user_role}

            # Create user
            hashed_password = get_password_hash(password)
            user = User(
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                role=role,
                is_active=True,
            )
            session.add(user)
            session.flush()

            # Extract data before session closes
            user_id = str(user.id)
            user_email = user.email
            user_role = user.role

            print(f"✅ Created user: {email} (ID: {user_id}, Role: {role})")

            return {"id": user_id, "email": user_email, "role": user_role}

    except Exception as e:
        print(f"❌ Failed to create user {email}: {e}")
        return None


def generate_test_token(user_id: str, email: str, role: str):
    """Generate test access token."""
    try:
        tokens = create_tokens(user_id=user_id, email=email, role=role)
        return tokens
    except Exception as e:
        print(f"❌ Failed to generate token: {e}")
        return None


def main():
    """Main setup function."""
    print("=" * 60)
    print("Oryon Authentication Setup")
    print("=" * 60)
    print()

    # 1. Setup JWT secret
    print("📋 Step 1: JWT Secret Key Configuration")
    print("-" * 60)
    _ = setup_jwt_secret()  # Result logged internally
    print()

    # 2. Create test users
    print("📋 Step 2: Creating Test Users")
    print("-" * 60)

    test_users = [
        {
            "email": "test@oryon.com",
            "password": "Test@1234567",
            "full_name": "Test User",
            "role": "user",
        },
        {
            "email": "reviewer@oryon.com",
            "password": "Reviewer@123456",
            "full_name": "Test Reviewer",
            "role": "teacher",
        },
        {
            "email": "admin@oryon.com",
            "password": "Admin@123456",
            "full_name": "Admin User",
            "role": "admin",
        },
    ]

    created_users = []
    for user_data in test_users:
        user_dict = create_test_user(**user_data)
        if user_dict:
            # Add password to dict for display
            user_dict["password"] = user_data["password"]
            created_users.append(user_dict)

    print()

    # 3. Generate test tokens
    print("📋 Step 3: Test Access Tokens")
    print("-" * 60)

    for user_data in created_users:
        tokens = generate_test_token(
            user_id=user_data["id"], email=user_data["email"], role=user_data["role"]
        )
        if tokens:
            print(f"\n✅ Token for {user_data['email']}:")
            print(f"   Access Token: {tokens.access_token[:50]}...")
            print(f"   Token Type: {tokens.token_type}")

    print()
    print("=" * 60)
    print("✅ Authentication Setup Complete!")
    print("=" * 60)
    print()
    print("📝 Test User Credentials:")
    print("-" * 60)
    for user_data in test_users:
        print(f"   Email: {user_data['email']}")
        print(f"   Password: {user_data['password']}")
        print(f"   Role: {user_data['role']}")
        print()

    print("🔧 Usage Examples:")
    print("-" * 60)
    print("1. Login to get token:")
    print("   curl -X POST http://localhost:8000/api/v2/auth/login \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"email": "test@oryon.com", "password": "Test@123456"}\'')
    print()
    print("2. Use token to access protected endpoints:")
    print("   curl -X POST http://localhost:8000/api/v2/chat/guest \\")
    print('     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"message": "What is photosynthesis?", "language": "English"}\'')
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
