"""
Structured Logger — Research-Grade Logging
============================================

Provides structured JSON logging with automatic correlation ID injection,
request context propagation, and hierarchical log levels optimized for
distributed systems debugging.

Design:
  - JSON-structured output for machine parsing
  - Human-readable fallback for development
  - Automatic context injection (request_id, trace_id, span_id)
  - Performance-safe (lazy formatting, sampling for DEBUG)
  - Thread-safe and async-compatible

Architecture:
  - This is the lowest-level telemetry primitive
  - All other layers import from here
  - Zero external dependencies beyond stdlib + structlog
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from contextvars import ContextVar
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

# ── Context Variables ──────────────────────────────────────────────

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
_tenant_id: ContextVar[str | None] = ContextVar("tenant_id", default=None)

def set_request_context(
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> None:
    """Set request-scoped context for log enrichment."""
    if request_id is not None:
        _request_id.set(request_id)
    if trace_id is not None:
        _trace_id.set(trace_id)
    if span_id is not None:
        _span_id.set(span_id)
    if user_id is not None:
        _user_id.set(user_id)
    if tenant_id is not None:
        _tenant_id.set(tenant_id)

def clear_request_context() -> None:
    """Clear all request-scoped context."""
    _request_id.set(None)
    _trace_id.set(None)
    _span_id.set(None)
    _user_id.set(None)
    _tenant_id.set(None)

# ── Structured Formatter ──────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter with automatic context injection."""

    def __init__(self, *, json_output: bool = True, include_traceback: bool = True):
        super().__init__()
        self._json = json_output
        self._include_tb = include_traceback
        self._hostname = os.uname().nodename
        self._pid = os.getpid()

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "pid": self._pid,
            "host": self._hostname,
        }

        # Inject context vars
        ctx_fields = {
            "request_id": _request_id.get(None),
            "trace_id": _trace_id.get(None),
            "span_id": _span_id.get(None),
            "user_id": _user_id.get(None),
            "tenant_id": _tenant_id.get(None),
        }
        entry["context"] = {k: v for k, v in ctx_fields.items() if v is not None}

        # Extra structured fields — use isinstance checks instead of
        # json.dumps() per field to avoid serialization overhead on every log line.
        _JSON_SAFE = (str, int, float, bool, type(None))
        extras: dict[str, str | int | float | None] = {}
        skip = {
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "pathname", "filename", "module", "levelno", "levelname",
            "thread", "threadName", "process", "processName", "msecs",
            "taskName", "message",
        }
        for key, val in record.__dict__.items():
            if key.startswith("_") or key in skip:
                continue
            if isinstance(val, _JSON_SAFE):
                extras[key] = val
            elif isinstance(val, (list, dict)):
                try:
                    json.dumps(val)
                    extras[key] = str(val)
                except (TypeError, ValueError):
                    extras[key] = str(val)
            else:
                extras[key] = str(val)

        if extras:
            entry["data"] = extras

        # Exception info
        if record.exc_info and self._include_tb:
            entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
                if record.exc_info[2]
                else None,
            }

        if self._json:
            return json.dumps(entry, default=str, ensure_ascii=False)

        # Human-readable fallback
        ctx = entry.get("context", {})
        req_id = ctx.get("request_id", "-")[:8]
        return (
            f"{entry['timestamp']} | {entry['level']:8s} | "
            f"{req_id} | {entry['logger']}:{entry['line']} | "
            f"{entry['message']}"
        )

# ── Structured Logger ─────────────────────────────────────────────

class StructuredLogger:
    """
    Wrapper around stdlib logger providing structured logging helpers.

    Usage:
        log = StructuredLogger("backend.infra.runtime")
        log.info("inference_complete", model="qwen3-8b", latency_ms=42.3)
        log.warning("memory_pressure", used_gb=14.2, threshold_gb=12.0)
    """

    __slots__ = ("_logger", "_name")

    def __init__(self, name: str):
        self._name = name
        self._logger = logging.getLogger(name)

    @property
    def name(self) -> str:
        return self._name

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        self._logger.log(level, event, extra=kwargs, stacklevel=3)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, exc: BaseException | None = None, **kwargs: Any) -> None:
        if exc:
            self._logger.error(event, extra=kwargs, exc_info=exc, stacklevel=2)
        else:
            self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, exc: BaseException | None = None, **kwargs: Any) -> None:
        if exc:
            self._logger.critical(event, extra=kwargs, exc_info=exc, stacklevel=2)
        else:
            self._log(logging.CRITICAL, event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        self._logger.exception(event, extra=kwargs, stacklevel=2)

    def bind(self, **context: Any) -> BoundLogger:
        """Create a child logger with bound context fields."""
        return BoundLogger(self, context)

class BoundLogger:
    """Logger with pre-bound context fields."""

    __slots__ = ("_context", "_parent")

    def __init__(self, parent: StructuredLogger, context: dict[str, Any]):
        self._parent = parent
        self._context = context

    def _merged(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = {**self._context, **kwargs}
        return merged

    def debug(self, event: str, **kwargs: Any) -> None:
        self._parent.debug(event, **self._merged(kwargs))

    def info(self, event: str, **kwargs: Any) -> None:
        self._parent.info(event, **self._merged(kwargs))

    def warning(self, event: str, **kwargs: Any) -> None:
        self._parent.warning(event, **self._merged(kwargs))

    def error(self, event: str, exc: BaseException | None = None, **kwargs: Any) -> None:
        self._parent.error(event, exc=exc, **self._merged(kwargs))

    def critical(self, event: str, exc: BaseException | None = None, **kwargs: Any) -> None:
        self._parent.critical(event, exc=exc, **self._merged(kwargs))

# ── Setup ──────────────────────────────────────────────────────────

_initialized = False

def setup_logging(
    *,
    level: str = "INFO",
    json_output: bool | None = None,
    log_dir: str | None = None,
) -> None:
    """
    Initialize the logging system. Call once at application startup.

    Args:
        level: Root log level
        json_output: Force JSON output. Auto-detects if None (JSON in production, human in dev)
        log_dir: Directory for log files. None = stdout only.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    if json_output is None:
        json_output = os.getenv("ENVIRONMENT", "development") != "development"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(StructuredFormatter(json_output=json_output))
    console.setLevel(logging.DEBUG)
    root.addHandler(console)

    # File handlers
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Rotating file handler (all levels)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "backend.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(StructuredFormatter(json_output=True))
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

        # Error-only file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "errors.log",
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=3,
            encoding="utf-8",
        )
        error_handler.setFormatter(StructuredFormatter(json_output=True))
        error_handler.setLevel(logging.ERROR)
        root.addHandler(error_handler)

    # Reduce noise from third-party libraries
    for noisy in (
        "httpx", "httpcore", "urllib3", "asyncio", "uvicorn.access",
        "uvicorn.error", "watchfiles", "multipart",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)
