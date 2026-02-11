#!/usr/bin/env python3
"""
Create demo accounts for ShikshaSetu
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.database import SessionLocal
from backend.models import User
from backend.utils.auth import get_password_hash


def create_demo_accounts():
    """Create demo user accounts"""

    demo_users = [
        {
            "email": "demo@shiksha.com",
            "password": "demo123",
            "full_name": "Demo User",
            "organization": "ShikshaSetu Demo",
            "role": "user",
        },
        {
            "email": "reviewer@shiksha.com",
            "password": "teacher123",
            "full_name": "Reviewer Demo",
            "organization": "Demo Org",
            "role": "user",
        },
        {
            "email": "admin@shiksha.com",
            "password": "admin123",
            "full_name": "Admin User",
            "organization": "ShikshaSetu",
            "role": "admin",
        },
        {
            "email": "test@test.com",
            "password": "test123",
            "full_name": "Test User",
            "organization": "Testing",
            "role": "user",
        },
    ]

    print("\n" + "=" * 60)
    print("CREATING DEMO ACCOUNTS")
    print("=" * 60 + "\n")

    with get_db_session() as session:
        for user_data in demo_users:
            # Check if user already exists
            existing_user = (
                session.query(User).filter(User.email == user_data["email"]).first()
            )

            if existing_user:
                print(f"⚠️  User already exists: {user_data['email']}")
                continue

            # Create new user
            hashed_password = get_password_hash(user_data["password"])
            new_user = User(
                email=user_data["email"],
                hashed_password=hashed_password,
                full_name=user_data["full_name"],
                organization=user_data["organization"],
                role=user_data["role"],
                is_active=True,
            )

            session.add(new_user)
            session.flush()

            print(f"✅ Created: {user_data['email']}")
            print(f"   Password: {user_data['password']}")
            print(f"   Role: {user_data['role']}\n")

    print("=" * 60)
    print("DEMO ACCOUNTS READY!")
    print("=" * 60 + "\n")

    print("📋 LOGIN CREDENTIALS:\n")
    print("1. Regular User:")
    print("   Email: demo@shiksha.com")
    print("   Password: demo123\n")

    print("2. Teacher Account:")
    print("   Email: reviewer@shiksha.com")
    print("   Password: teacher123\n")

    print("3. Admin Account:")
    print("   Email: admin@shiksha.com")
    print("   Password: admin123\n")

    print("4. Test Account:")
    print("   Email: test@test.com")
    print("   Password: test123\n")

    print("🌐 Login at: http://localhost:5173/login\n")


if __name__ == "__main__":
    try:
        create_demo_accounts()
    except Exception as e:
        print(f"\n❌ Error creating demo accounts: {e}\n")
        sys.exit(1)
