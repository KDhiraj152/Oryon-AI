#!/usr/bin/env python3
"""Quick API test script for Oryon."""

import random
import string

import requests

BASE = "http://localhost:8000"


def main():
    print("Oryon API Comprehensive Test")
    print("=" * 60)

    # Test health endpoints
    endpoints = [
        ("GET", "/api/v2/health"),
        ("GET", "/api/v2/health/detailed"),
        ("GET", "/api/v2/policy"),
        ("GET", "/api/v2/hardware/status"),
        ("GET", "/api/v2/models/status"),
    ]

    for method, path in endpoints:
        try:
            r = requests.get(f"{BASE}{path}", timeout=5)
            status_icon = "✓" if r.status_code == 200 else "✗"
            print(f"{status_icon} {method:4} {path:30} -> {r.status_code}")
        except Exception as e:
            print(f"✗ {method:4} {path:30} -> ERROR: {e}")

    print()
    print("Auth Flow Test:")
    print("-" * 60)

    # Generate unique test user
    rand_suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    test_email = f"test_{rand_suffix}@example.com"
    test_password = "TestPass123!@#"  # Meets all requirements

    # Test registration
    try:
        r = requests.post(
            f"{BASE}/api/v2/auth/register",
            json={"email": test_email, "password": test_password, "name": "Test User"},
            timeout=10,
        )
        print(f"Register ({test_email}): {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            access_token = data.get("access_token")
            print(f"  Got access_token: {bool(access_token)}")
            print(f'  User ID: {data.get("user", {}).get("id", "N/A")}')
        else:
            print(f"  Response: {r.text[:200]}")
    except Exception as e:
        print(f"Register: ERROR - {e}")

    # Test login
    try:
        r = requests.post(
            f"{BASE}/api/v2/auth/login",
            json={"email": test_email, "password": test_password},
            timeout=10,
        )
        print(f"Login: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            access_token = data.get("access_token")
            print(f"  Got access_token: {bool(access_token)}")

            # Test authenticated endpoint
            if access_token:
                headers = {"Authorization": f"Bearer {access_token}"}
                r2 = requests.get(f"{BASE}/api/v2/auth/me", headers=headers, timeout=5)
                print(f"  Get /auth/me: {r2.status_code}")
        else:
            print(f"  Response: {r.text[:200]}")
    except Exception as e:
        print(f"Login: ERROR - {e}")

    print()
    print("=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
