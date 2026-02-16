"""
Enhanced API Documentation

Issue: CODE-REVIEW-GPT #21 (LOW)
Purpose: Comprehensive OpenAPI/Swagger documentation
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi_schema(app: FastAPI) -> dict[str, Any]:
    """
    Generate custom OpenAPI schema with enhanced documentation.

    Args:
        app: FastAPI application

    Returns:
        OpenAPI schema dictionary
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Oryon AI Platform API",
        version="2.1.0",
        description="""
# Oryon - Multilingual AI Education Platform

## Overview

Oryon provides AI-powered content processing with:
- **Multilingual Support**: Content in 22 Indian languages
- **Grade-Level Adaptation**: Automatic complexity adjustment for grades 1-12
- **Curriculum Alignment**: content_domain standard validation
- **Cultural Context**: Region-specific content adaptation
- **Quality Assurance**: Comprehensive validation pipeline

## Authentication

Most endpoints require authentication via JWT tokens:

```
Authorization: Bearer <your_jwt_token>
```

Get tokens from `/api/auth/login` endpoint.

## Rate Limiting

API requests are rate-limited based on user role:
- **Users**: 60 requests/minute, 600/hour
- **Teachers**: 200 requests/minute, 2000/hour
- **Admins**: 1000 requests/minute, 10000/hour

Rate limit headers included in responses:
- `X-RateLimit-Limit-Minute`
- `X-RateLimit-Remaining-Minute`

## Error Handling

Standard error response format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input provided",
    "details": {}
  }
}
```

## Content Pipeline

Content processing flow:
1. **Upload** → 2. **Simplification** → 3. **Translation** →
4. **Validation** → 5. **TTS Generation** → 6. **Quality Check**

## Support

- **Documentation**: https://docs.oryon-ai.com
- **Email**: support@oryon-ai.com
- **GitHub**: https://github.com/KDhiraj152/Siksha-Setu
        """,
        routes=app.routes,
        tags=[
            {
                "name": "Authentication",
                "description": "User authentication and authorization",
            },
            {
                "name": "Content",
                "description": "Content creation, processing, and management",
            },
            {"name": "Translation", "description": "Multilingual translation services"},
            {"name": "Q&A", "description": "Question-answering and semantic search"},
            {
                "name": "Progress",
                "description": "User progress tracking and analytics",
            },
            {
                "name": "Content Review",
                "description": "Reviewer performance evaluation and metrics",
            },
            {
                "name": "Validation",
                "description": "Content validation and quality checks",
            },
            {"name": "Health", "description": "System health and monitoring"},
        ],
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token from /api/auth/login",
        },
        "APIKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service-to-service authentication",
        },
    }

    # Add servers
    openapi_schema["servers"] = [
        {"url": "https://api.oryon-ai.com", "description": "Production server"},
        {
            "url": "https://staging-api.oryon-ai.com",
            "description": "Staging server",
        },
        {"url": "http://localhost:8000", "description": "Development server"},
    ]

    # Add contact info
    openapi_schema["info"]["contact"] = {
        "name": "Oryon Support",
        "email": "support@oryon-ai.com",
        "url": "https://oryon-ai.com/support",
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    # Add external docs
    openapi_schema["externalDocs"] = {
        "description": "Complete Documentation",
        "url": "https://docs.oryon-ai.com",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

def configure_api_docs(app: FastAPI):
    """
    Configure enhanced API documentation.

    Args:
        app: FastAPI application
    """
    # Set custom OpenAPI schema
    app.openapi = lambda: custom_openapi_schema(app)

    # Configure Swagger UI
    app.swagger_ui_parameters = {
        "deepLinking": True,
        "persistAuthorization": True,
        "displayRequestDuration": True,
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
        "tryItOutEnabled": True,
    }

# Example schemas for common responses
COMMON_RESPONSES = {
    "400": {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Invalid input",
                        "details": {"field": "text", "error": "Text is required"},
                    }
                }
            }
        },
    },
    "401": {
        "description": "Unauthorized",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Authentication required",
                    }
                }
            }
        },
    },
    "403": {
        "description": "Forbidden",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "FORBIDDEN",
                        "message": "Insufficient permissions",
                    }
                }
            }
        },
    },
    "429": {
        "description": "Too Many Requests",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Rate limit exceeded",
                        "retry_after": 60,
                    }
                }
            }
        },
    },
    "500": {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred",
                        "request_id": "req_123abc",
                    }
                }
            }
        },
    },
}
