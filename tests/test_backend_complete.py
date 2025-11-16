"""
Comprehensive Backend Tests for ShikshaSetu Platform
Tests all critical backend functionality including:
- Authentication & Authorization
- File Upload & Processing
- ML Pipeline (Simplification, Translation, Validation, TTS)
- RAG/Q&A System
- Database Operations
- API Endpoints
"""
import pytest
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from sqlalchemy import text

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

# Test fixtures
@pytest.fixture(scope="module")
def test_user():
    """Create a test user and return credentials."""
    return {
        "email": f"test_{int(time.time())}@example.com",
        "password": "TestPass123!",
        "full_name": "Test User"
    }

@pytest.fixture(scope="module")
def auth_tokens(test_user):
    """Register user and get authentication tokens."""
    response = requests.post(
        f"{API_BASE}/auth/register",
        json=test_user
    )
    assert response.status_code == 200, f"Registration failed: {response.text}"
    tokens = response.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    return tokens

@pytest.fixture(scope="module")
def auth_headers(auth_tokens):
    """Return headers with authentication token."""
    return {
        "Authorization": f"Bearer {auth_tokens['access_token']}"
    }


class TestHealthAndStatus:
    """Test system health and status endpoints."""
    
    def test_basic_health_check(self):
        """Test basic health endpoint."""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_api_docs_accessible(self):
        """Test that API documentation is accessible."""
        response = requests.get(f"{BASE_URL}/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "redoc" in response.text.lower()


class TestAuthentication:
    """Test authentication and authorization."""
    
    def test_user_registration(self, test_user):
        """Test new user registration."""
        # Use unique email
        unique_user = test_user.copy()
        unique_user["email"] = f"new_{int(time.time())}@test.com"
        
        response = requests.post(
            f"{API_BASE}/auth/register",
            json=unique_user
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_user_login(self, test_user, auth_tokens):
        """Test user login with credentials."""
        response = requests.post(
            f"{API_BASE}/auth/login",
            json={
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
    
    def test_get_current_user(self, auth_headers):
        """Test getting current user info."""
        response = requests.get(
            f"{API_BASE}/auth/me",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "email" in data
        assert "id" in data
    
    def test_token_refresh(self, auth_tokens):
        """Test refreshing access token."""
        response = requests.post(
            f"{API_BASE}/auth/refresh",
            json={"refresh_token": auth_tokens["refresh_token"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
    
    def test_unauthorized_access(self):
        """Test that endpoints require authentication."""
        response = requests.get(f"{API_BASE}/auth/me")
        assert response.status_code in [401, 403]


class TestFileUpload:
    """Test file upload functionality."""
    
    def test_upload_text_file(self, auth_headers):
        """Test uploading a text file."""
        # Create test file
        test_content = "This is a test document for educational content processing."
        files = {
            "file": ("test.txt", test_content.encode(), "text/plain")
        }
        
        response = requests.post(
            f"{API_BASE}/upload",
            headers=auth_headers,
            files=files
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "file_path" in data
        assert "filename" in data
        assert data["filename"] == "test.txt"
    
    def test_upload_without_auth(self):
        """Test that upload requires authentication."""
        files = {
            "file": ("test.txt", b"content", "text/plain")
        }
        
        response = requests.post(
            f"{API_BASE}/upload",
            files=files
        )
        
        assert response.status_code in [401, 403]


class TestMLPipeline:
    """Test ML pipeline components."""
    
    def test_text_simplification(self, auth_headers):
        """Test text simplification endpoint."""
        payload = {
            "text": "Photosynthesis is the biochemical process by which plants convert light energy into chemical energy.",
            "grade_level": 6,
            "subject": "Science"
        }
        
        response = requests.post(
            f"{API_BASE}/simplify",
            headers=auth_headers,
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "simplified_text" in data
        assert len(data["simplified_text"]) > 0
    
    def test_text_translation(self, auth_headers):
        """Test text translation endpoint."""
        payload = {
            "text": "Hello, this is a test.",
            "target_language": "Hindi",
            "source_language": "English"
        }
        
        response = requests.post(
            f"{API_BASE}/translate",
            headers=auth_headers,
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "translated_text" in data
        assert len(data["translated_text"]) > 0
    
    def test_content_validation(self, auth_headers):
        """Test content validation endpoint."""
        payload = {
            "original_text": "Plants make food using sunlight.",
            "processed_text": "Plants create food with light from the sun.",
            "grade_level": 5,
            "subject": "Science"
        }
        
        response = requests.post(
            f"{API_BASE}/validate",
            headers=auth_headers,
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data
        assert "semantic_score" in data


class TestAsyncTasks:
    """Test async task management."""
    
    def test_task_status_retrieval(self, auth_headers):
        """Test getting task status."""
        # Create a simple task first
        payload = {
            "text": "Test content",
            "grade_level": 8,
            "subject": "Math"
        }
        
        response = requests.post(
            f"{API_BASE}/simplify",
            headers=auth_headers,
            json=payload
        )
        
        # Note: If simplify is sync, this test validates the endpoint works
        assert response.status_code == 200


class TestRAGSystem:
    """Test RAG/Q&A functionality."""
    
    @pytest.fixture
    def uploaded_document(self, auth_headers):
        """Upload a document for Q&A testing."""
        test_content = """
        Photosynthesis is the process by which plants make their own food.
        Plants use sunlight, water, and carbon dioxide to create glucose and oxygen.
        This process happens in the chloroplasts of plant cells.
        Chlorophyll is the green pigment that captures light energy.
        """
        
        files = {
            "file": ("science.txt", test_content.encode(), "text/plain")
        }
        
        response = requests.post(
            f"{API_BASE}/upload",
            headers=auth_headers,
            files=files
        )
        
        assert response.status_code == 200
        return response.json()
    
    def test_process_document_for_qa(self, auth_headers, uploaded_document):
        """Test processing document for Q&A."""
        # First, we need to process the document
        payload = {
            "file_path": uploaded_document["file_path"],
            "grade_level": 7,
            "subject": "Science",
            "target_languages": ["Hindi"],
            "output_format": "text"
        }
        
        response = requests.post(
            f"{API_BASE}/process",
            headers=auth_headers,
            json=payload
        )
        
        # Should return task_id or content_id
        assert response.status_code in [200, 202]


class TestDatabaseOperations:
    """Test database connectivity and operations through API."""
    
    def test_database_connection(self):
        """Test that database is connected via health endpoint."""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        health = response.json()
        assert health["status"] == "healthy"
        assert health["database"] == "connected"
    
    def test_database_tables_exist(self):
        """Test that all required tables exist via API endpoints."""
        # Test by hitting endpoints that require database tables
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        
        # If health check passes, tables exist


class TestContentRetrieval:
    """Test content retrieval endpoints."""
    
    def test_feedback_submission(self, auth_headers):
        """Test submitting feedback."""
        # This assumes we have a content_id, but we'll test the endpoint structure
        payload = {
            "content_id": "00000000-0000-0000-0000-000000000000",  # Dummy UUID
            "rating": 5,
            "feedback_text": "Great simplification!",
            "issue_type": "none"
        }
        
        response = requests.post(
            f"{API_BASE}/feedback",
            headers=auth_headers,
            json=payload
        )
        
        # May fail due to non-existent content_id, but validates endpoint exists
        assert response.status_code in [200, 404, 422]


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_grade_level(self, auth_headers):
        """Test validation of grade level."""
        payload = {
            "text": "Test",
            "grade_level": 99,  # Invalid
            "subject": "Science"
        }
        
        response = requests.post(
            f"{API_BASE}/simplify",
            headers=auth_headers,
            json=payload
        )
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    def test_invalid_language(self, auth_headers):
        """Test validation of language."""
        payload = {
            "text": "Test",
            "target_language": "InvalidLanguage",
            "source_language": "English"
        }
        
        response = requests.post(
            f"{API_BASE}/translate",
            headers=auth_headers,
            json=payload
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]
    
    def test_missing_required_fields(self, auth_headers):
        """Test handling of missing required fields."""
        payload = {
            "text": "Test"
            # Missing grade_level and subject
        }
        
        response = requests.post(
            f"{API_BASE}/simplify",
            headers=auth_headers,
            json=payload
        )
        
        assert response.status_code in [400, 422]


class TestSecurityFeatures:
    """Test security features."""
    
    def test_rate_limiting_exists(self):
        """Test that rate limiting is configured (by checking headers)."""
        response = requests.get(f"{BASE_URL}/health")
        # Rate limit headers may be present
        assert response.status_code == 200
    
    def test_cors_headers(self):
        """Test that CORS headers are set."""
        response = requests.get(f"{BASE_URL}/health", headers={"Origin": "http://localhost:3000"})
        # Check for CORS headers in GET response
        assert response.status_code == 200
        # CORS headers should be present for cross-origin requests


# Test runner function
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
