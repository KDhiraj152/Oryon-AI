#!/usr/bin/env python3
"""
Reset and recreate demo accounts with proper password handling
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.database import SessionLocal, get_db_session
from backend.models import User
from backend.utils.auth import get_password_hash


def reset_demo_accounts():
    """Delete and recreate demo accounts"""
    
    demo_users = [
        {
            "email": "demo@oryon.com",
            "password": "demo123",
            "full_name": "Demo User",
            "organization": "Oryon Demo",
            "role": "user"
        },
        {
            "email": "test@test.com",
            "password": "test123",
            "full_name": "Test User",
            "organization": "Testing",
            "role": "user"
        }
    ]
    
    print("\n" + "="*60)
    print("RESETTING DEMO ACCOUNTS")
    print("="*60 + "\n")
    
    with get_db_session() as session:
        for user_data in demo_users:
            # Delete existing user
            existing_user = session.query(User).filter(
                User.email == user_data["email"]
            ).first()
            
            if existing_user:
                session.delete(existing_user)
                session.flush()
                print(f"🗑️  Deleted old account: {user_data['email']}")
            
            # Create new user with properly truncated password
            password = user_data["password"][:72]  # Ensure password is truncated
            hashed_password = get_password_hash(password)
            
            new_user = User(
                email=user_data["email"],
                hashed_password=hashed_password,
                full_name=user_data["full_name"],
                organization=user_data["organization"],
                role=user_data["role"],
                is_active=True
            )
            
            session.add(new_user)
            session.flush()
            
            print(f"✅ Created: {user_data['email']}")
            print(f"   Password: {password}\n")
    
    print("="*60)
    print("✅ DEMO ACCOUNTS READY!")
    print("="*60 + "\n")
    
    print("🔐 SIMPLE LOGIN CREDENTIALS:\n")
    print("Email: demo@oryon.com")
    print("Password: demo123\n")
    
    print("Email: test@test.com")
    print("Password: test123\n")
    
    print("🌐 Login at: http://localhost:5173/login\n")

if __name__ == "__main__":
    try:
        reset_demo_accounts()
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
