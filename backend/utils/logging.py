"""Centralized logging configuration for ShikshaSetu."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from ..core.config import settings


def setup_logging(
    name: str | None = None, log_level: str | None = None
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting and handlers.

    Args:
        name: Logger name (defaults to settings.APP_NAME)
        log_level: Log level (defaults to settings.LOG_LEVEL)

    Returns:
        Configured logger instance
    """
    logger_name = name or settings.APP_NAME
    level = log_level or settings.LOG_LEVEL

    # Get or create logger
    logger = logging.getLogger(logger_name)

    # Only configure if not already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(fmt="%(levelname)s - %(name)s - %(message)s")

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        simple_formatter if settings.ENVIRONMENT == "production" else detailed_formatter
    )
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Ensure log directory exists
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # File handler with rotation (all logs)
    try:
        file_handler = RotatingFileHandler(
            settings.LOG_DIR / settings.LOG_FILE,
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")

    # Error file handler (errors only)
    try:
        error_handler = RotatingFileHandler(
            settings.LOG_DIR / f"{settings.LOG_FILE.replace('.log', '_errors.log')}",
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)
    except Exception as e:
        logger.warning(f"Could not create error handler: {e}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize root logger on import
root_logger = setup_logging()
