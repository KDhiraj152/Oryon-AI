"""
Enhanced FastAPI application with async task processing.

BACKWARD COMPATIBILITY WRAPPER:
This file now imports the new modular application from main.py.
The application has been refactored into:
- src/api/routes/ (health.py, auth.py, content.py, qa.py)
- src/core/ (config.py, security.py, exceptions.py)
- src/api/middleware.py (security headers, request timing, logging)
- src/utils/logging_config.py (centralized logging)

All functionality is maintained, just better organized!
"""

# Import the new modular app
from .main import app

# For any code that imports from async_app, expose everything
__all__ = ["app"]
