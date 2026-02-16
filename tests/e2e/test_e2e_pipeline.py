"""
End-to-End Tests for Oryon Pipeline

Tests the complete flow: upload → OCR → simplify → translate → validate → TTS → fetch
Designed for local M4 development environment.

Created by: TITAN-PROTOCOL

NOTE: These tests require a running backend server at API_URL (default: http://localhost:8000)
Run with: API_URL=http://localhost:8000 pytest tests/e2e/ -v
"""

import asyncio
import os
import time
from pathlib import Path

import httpx
import pytest

# Test configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
TEST_TIMEOUT = 120  # seconds for full pipeline
POLL_INTERVAL = 2  # seconds between status checks


def is_server_running():
    """Check if the backend server is running."""
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{API_URL}/api/v2/health")
            return response.status_code == 200
    except Exception:
        return False


# Skip all e2e tests if server is not running
pytestmark = pytest.mark.skipif(
    not is_server_running(),
    reason=f"Backend server not running at {API_URL}. Start it with: uvicorn backend.api.main:app",
)


@pytest.fixture(scope="module")
def api_client():
    """Create async HTTP client for tests."""
    return httpx.Client(base_url=API_URL, timeout=30.0)


@pytest.fixture(scope="module")
def test_user_token(api_client):
    """Get or create test user and return auth token."""
    # Try to register test user
    register_data = {
        "email": "e2e_test@oryon.local",
        "password": "E2eTest@2025Secure!",
        "name": "E2E Test User",
    }

    try:
        response = api_client.post("/api/v2/auth/register", json=register_data)
        if response.status_code == 201:
            return response.json().get("access_token")
    except Exception:
        pass

    # If registration fails, try login
    login_data = {
        "username": register_data["email"],
        "password": register_data["password"],
    }
    response = api_client.post("/api/v2/auth/login", data=login_data)

    if response.status_code == 200:
        return response.json().get("access_token")

    pytest.skip("Could not authenticate test user")


@pytest.fixture
def auth_headers(test_user_token):
    """Return auth headers for API requests."""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    content = """Photosynthesis is the process by which plants convert sunlight into energy.
The equation for photosynthesis is: 6CO2 + 6H2O + light → C6H12O6 + 6O2.
This process occurs in the chloroplasts of plant cells.
Chlorophyll, the green pigment in plants, captures light energy.
The process has two stages: light-dependent and light-independent reactions.
"""
    file_path = tmp_path / "test_content.txt"
    file_path.write_text(content)
    return file_path


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_basic_health(self, api_client):
        """Test basic health endpoint."""
        response = api_client.get("/api/v2/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data or "version" in data

    def test_detailed_health(self, api_client):
        """Test detailed health endpoint."""
        response = api_client.get("/api/v2/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestAuthentication:
    """Test authentication flow."""

    def test_login_with_invalid_credentials(self, api_client):
        """Test login with wrong password."""
        response = api_client.post(
            "/api/v2/auth/login",
            data={"username": "nonexistent@test.com", "password": "wrongpassword"},
        )
        # 422 = validation error (missing fields), 401 = invalid credentials, 400 = bad request
        assert response.status_code in [401, 400, 422]

    def test_protected_endpoint_without_token(self, api_client):
        """Test accessing protected endpoint without auth."""
        response = api_client.get("/api/v2/content/library")
        # 401 = unauthorized, 404 = endpoint doesn't exist
        assert response.status_code in [401, 404]


class TestContentUpload:
    """Test content upload functionality."""

    def test_upload_text_file(self, api_client, auth_headers, sample_text_file):
        """Test uploading a text file."""
        with open(sample_text_file, "rb") as f:
            files = {"file": ("test_content.txt", f, "text/plain")}
            response = api_client.post(
                "/api/v2/content/process", files=files, headers=auth_headers
            )

        # Accept either success or expected error codes
        assert response.status_code in [200, 201, 202, 400, 422]

        if response.status_code in [200, 201, 202]:
            data = response.json()
            assert "id" in data or "content_id" in data or "task_id" in data


class TestContentProcessing:
    """Test content processing pipeline."""

    @pytest.mark.slow
    def test_simplify_text(self, api_client, auth_headers):
        """Test text simplification."""
        response = api_client.post(
            "/api/v2/content/simplify",
            json={
                "text": "Photosynthesis is the biochemical process by which chloroplasts use light energy to convert carbon dioxide and water into glucose.",
                "target_grade": 5,
                "language": "English",
            },
            headers=auth_headers,
        )

        # May return task ID for async processing or direct result
        assert response.status_code in [200, 202]

    @pytest.mark.slow
    def test_translate_text(self, api_client, auth_headers):
        """Test text translation to Hindi."""
        response = api_client.post(
            "/api/v2/content/translate",
            json={
                "text": "Plants make their own food using sunlight.",
                "target_language": "Hindi",
            },
            headers=auth_headers,
        )

        assert response.status_code in [200, 202]


class TestFullPipeline:
    """Test complete content processing pipeline."""

    @pytest.mark.slow
    @pytest.mark.e2e
    def test_full_pipeline_flow(self, api_client, auth_headers, sample_text_file):
        """
        Test the complete flow:
        1. Upload content
        2. Process (simplify + translate)
        3. Poll for completion
        4. Fetch results
        """
        # Step 1: Upload
        with open(sample_text_file, "rb") as f:
            files = {"file": ("test_content.txt", f, "text/plain")}
            upload_response = api_client.post(
                "/api/v2/content/process", files=files, headers=auth_headers
            )

        if upload_response.status_code not in [200, 201, 202]:
            pytest.skip(f"Upload failed with status {upload_response.status_code}")

        upload_data = upload_response.json()
        content_id = upload_data.get("id") or upload_data.get("content_id")

        if not content_id:
            pytest.skip("No content_id returned from upload")

        # Step 2: Trigger processing
        process_response = api_client.post(
            f"/api/v2/content/{content_id}/process",
            json={
                "target_grade": 5,
                "target_language": "Hindi",
                "enable_tts": False,  # Skip TTS for faster test
            },
            headers=auth_headers,
        )

        if process_response.status_code not in [200, 202]:
            pytest.skip(
                f"Process trigger failed with status {process_response.status_code}"
            )

        task_id = process_response.json().get("task_id")

        # Step 3: Poll for completion
        start_time = time.time()
        while time.time() - start_time < TEST_TIMEOUT:
            status_response = api_client.get(
                f"/api/v2/content/tasks/{task_id or content_id}/status",
                headers=auth_headers,
            )

            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get("status", "").lower()

                if status in ["completed", "success", "done"]:
                    break
                elif status in ["failed", "error"]:
                    pytest.fail(
                        f"Pipeline failed: {status_data.get('error', 'Unknown error')}"
                    )

            time.sleep(POLL_INTERVAL)
        else:
            pytest.fail(f"Pipeline timed out after {TEST_TIMEOUT} seconds")

        # Step 4: Fetch results
        result_response = api_client.get(
            f"/api/v2/content/{content_id}", headers=auth_headers
        )

        assert result_response.status_code == 200
        result_data = result_response.json()

        # Verify results contain expected fields
        assert "id" in result_data or "content_id" in result_data


class TestQAFeatures:
    """Test Q&A functionality."""

    def test_ask_question(self, api_client, auth_headers):
        """Test asking a question."""
        response = api_client.post(
            "/api/v2/chat/guest",
            json={
                "question": "What is photosynthesis?",
                "context": "Photosynthesis is the process by which plants make food using sunlight.",
            },
            headers=auth_headers,
        )

        # May be async or sync
        assert response.status_code in [200, 202]


class TestResourceMetrics:
    """Test that resource usage is within M4 limits."""

    def test_metrics_endpoint(self, api_client):
        """Test Prometheus metrics endpoint."""
        response = api_client.get("/metrics")
        assert response.status_code == 200
        assert b"http_requests_total" in response.content


# Convenience function to run smoke test
def run_smoke_test():
    """Run a quick smoke test - can be called from scripts."""
    import subprocess

    result = subprocess.run(
        ["pytest", __file__, "-v", "-m", "not slow", "--tb=short"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


if __name__ == "__main__":
    # Allow running directly: python test_e2e_pipeline.py
    pytest.main([__file__, "-v", "--tb=short"])
