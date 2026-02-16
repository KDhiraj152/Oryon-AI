"""
Enhanced API Documentation for Oryon AI

Generates a custom OpenAPI schema and Swagger UI configuration
that accurately reflects the platform's capabilities.
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
    # Return cached schema if already generated
    cached: dict[str, Any] | None = getattr(app, "openapi_schema", None)
    if cached:
        return cached

    openapi_schema = get_openapi(
        title="Oryon AI Platform API",
        version="3.0.0",
        description="""
# Oryon AI — Self-Hosted ML Orchestration Engine

## Overview

Oryon AI is a domain-agnostic, self-hosted ML orchestration platform that runs
LLM chat, RAG, voice I/O, document analysis, semantic search, and multilingual
translation entirely on your own hardware.

All processing is **local-first** — no external API calls or telemetry.

## Capabilities

| Capability | Model | Endpoint Prefix |
|:--|:--|:--|
| Chat & Reasoning | Qwen3-8B (MLX 4-bit) | `/api/v2/chat` |
| Translation | IndicTrans2-1B | `/api/v2/content/translate` |
| Voice Input (STT) | Whisper V3 Turbo | `/api/v2/content/stt` |
| Voice Output (TTS) | MMS-TTS + Edge-TTS | `/api/v2/content/tts` |
| Semantic Search | BGE-M3 + BGE-Reranker-v2-M3 | `/api/v2/content/embeddings` |
| Document Intelligence | GOT-OCR2 + PyMuPDF | `/api/v2/content/ocr` |
| Batch Processing | Multi-model pipeline | `/api/v2/batch` |
| Agent System | 7 specialized agents | `/api/v2/agents` |

## Authentication

Most endpoints work without authentication for quick prototyping.
Protected endpoints require a JWT token:

```
Authorization: Bearer <your_jwt_token>
```

Obtain tokens via `POST /api/v2/auth/login`.

## Rate Limiting

Requests are rate-limited by role:
- **Guests / Users**: 60 req/min, 600/hr
- **Admins**: 1000 req/min, 10,000/hr

Rate limit headers are included in all responses:
- `X-RateLimit-Limit-Minute`
- `X-RateLimit-Remaining-Minute`

## Error Format

All errors follow a consistent structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input provided",
    "details": {}
  }
}
```

## Support

- **GitHub**: https://github.com/KDhiraj152/Shiksha-setu
        """,
        routes=app.routes,
        tags=[
            # Auth
            {
                "name": "auth",
                "description": "User registration, login, token refresh, and profile management.",
            },
            # Chat
            {
                "name": "chat",
                "description": (
                    "Conversational AI with RAG-enhanced responses, SSE streaming, "
                    "guest mode, conversation management, and TTS playback."
                ),
            },
            # Content pipeline
            {
                "name": "content",
                "description": (
                    "Full content processing pipeline — simplify, translate, "
                    "and generate audio from text in a single call."
                ),
            },
            {
                "name": "qa",
                "description": (
                    "Document Q&A — ingest PDFs/docs into a vector index, "
                    "then ask questions with context-aware retrieval."
                ),
            },
            {
                "name": "stt",
                "description": "Speech-to-text transcription via Whisper V3 Turbo.",
            },
            {
                "name": "ocr",
                "description": "Document text extraction via GOT-OCR2 and PyMuPDF.",
            },
            {
                "name": "embeddings",
                "description": (
                    "Embedding generation (BGE-M3) and document reranking "
                    "(BGE-Reranker-v2-M3) for semantic search."
                ),
            },
            {
                "name": "ai",
                "description": "AI explanation, prompt templates, and content safety checks.",
            },
            # Batch & multi-model
            {
                "name": "batch",
                "description": "Hardware-optimized batch processing for bulk text and embeddings.",
            },
            {
                "name": "multimodel",
                "description": (
                    "Multi-model collaboration modes: verify, chain, ensemble, "
                    "and back-translation."
                ),
            },
            # Agents
            {
                "name": "agents",
                "description": (
                    "Multi-agent system — status, metrics, SLA compliance, "
                    "regressions, and optimization triggers."
                ),
            },
            # Middleware
            {
                "name": "middleware",
                "description": (
                    "Middleware pipeline observability — request classification, "
                    "heuristic tuning, memory and latency stats."
                ),
            },
            # System & monitoring
            {
                "name": "health",
                "description": "Basic and detailed health checks for the API.",
            },
            {
                "name": "policy",
                "description": "Runtime policy configuration and mode switching.",
            },
            {
                "name": "monitoring",
                "description": "API statistics, device info, and uptime.",
            },
            {
                "name": "system",
                "description": (
                    "Hardware status, ML model status, cache status, "
                    "batch metrics, and performance benchmarks."
                ),
            },
            {
                "name": "progress",
                "description": "User learning progress, quiz generation, and quiz submission.",
            },
            {
                "name": "admin",
                "description": "Admin-only operations — database backups.",
            },
            {
                "name": "review",
                "description": "Content review queue for flagged AI responses (teacher/admin).",
            },
            {
                "name": "profile",
                "description": "User profile retrieval and updates.",
            },
        ],
    )

    # Security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from POST /api/v2/auth/login",
        },
    }

    # Server list
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local development"},
    ]

    # Contact info
    openapi_schema["info"]["contact"] = {
        "name": "K Dhiraj",
        "email": "k.dhiraj.srihari@gmail.com",
        "url": "https://github.com/KDhiraj152",
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    openapi_schema["externalDocs"] = {
        "description": "Full Documentation",
        "url": "https://github.com/KDhiraj152/Shiksha-setu/tree/main/docs",
    }

    # Cache the schema on the app instance so FastAPI.openapi() returns it
    setattr(app, "openapi_schema", openapi_schema)
    return openapi_schema


def configure_api_docs(app: FastAPI) -> None:
    """
    Wire the custom OpenAPI schema and Swagger UI settings into the app.

    Call this during application startup (e.g. in a lifespan handler)
    to override FastAPI's default schema with the enriched version.

    Args:
        app: FastAPI application instance.
    """
    # Pre-generate and cache the schema so FastAPI's openapi() returns it
    # directly without regenerating.  This avoids assigning to the bound
    # method (which Pyright flags) while achieving the same result.
    custom_openapi_schema(app)  # populates app.openapi_schema internally

    # Swagger UI preferences
    setattr(app, "swagger_ui_parameters", {
        "deepLinking": True,
        "persistAuthorization": True,
        "displayRequestDuration": True,
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
        "tryItOutEnabled": True,
    })


# ---------------------------------------------------------------------------
# Reusable response schemas for route decorators
# ---------------------------------------------------------------------------
# Usage in routes:
#   from backend.api.documentation import COMMON_RESPONSES
#   @router.post("/endpoint", responses={**COMMON_RESPONSES})

COMMON_RESPONSES: dict[str, Any] = {
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
