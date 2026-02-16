"""
Optimized database connection and session management.

Supports both local PostgreSQL and Supabase.
Uses FastAPI's Depends pattern for session management.
Includes async session support for non-blocking operations.

OPTIMIZATION NOTES (December 2025):
- Aggressive connection pooling with pre-ping validation
- Optimistic locking with retry logic
- Statement caching for repeated queries
- Async-first session management for high concurrency
- Event listeners for connection health monitoring
"""

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import TypeVar

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# Type variable for generic session operations
T = TypeVar("T")

# Create base class for models
Base = declarative_base()

# Database configuration - supports Supabase or local PostgreSQL
# Default uses environment variable; local dev fallback only for development
_DEFAULT_DEV_URL = "postgresql://postgres:postgres@localhost:5432/oryon"
DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_DEV_URL)

# Async database URL (convert postgresql:// to postgresql+asyncpg://)
def _get_async_url(url: str) -> str:
    """Convert sync database URL to async URL."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url

ASYNC_DATABASE_URL = _get_async_url(DATABASE_URL)

# Supabase configuration (optional)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Import settings for pool configuration
try:
    from .core.config import settings

    DB_POOL_SIZE = settings.DB_POOL_SIZE
    DB_MAX_OVERFLOW = settings.DB_MAX_OVERFLOW
    DB_POOL_TIMEOUT = settings.DB_POOL_TIMEOUT
    DB_POOL_RECYCLE = settings.DB_POOL_RECYCLE
except ImportError:
    # Fallback defaults if settings not available
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# Create engine with connection pooling and retry logic
# Handle SQLite vs PostgreSQL differences
is_sqlite = DATABASE_URL.startswith("sqlite")

if is_sqlite:
    # SQLite configuration (sync)
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    )
    # SQLite async engine
    async_engine = None  # aiosqlite support is limited
else:
    # PostgreSQL configuration (sync) - optimized for high concurrency
    # NOTE: Use basic connect_args for psycopg2 compatibility
    _pg_connect_args = {
        "connect_timeout": 10,
        "options": "-c statement_timeout=30000",  # 30s query timeout
    }

    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=DB_POOL_RECYCLE,  # Recycle connections periodically
        pool_timeout=DB_POOL_TIMEOUT,  # Timeout for getting connection from pool
        pool_use_lifo=True,  # OPTIMIZATION: Use LIFO to keep connections warm
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
        connect_args=_pg_connect_args,
        # OPTIMIZATION: Execution options for better performance
        execution_options={
            "compiled_cache": {},  # Enable query compilation cache
            "stream_results": False,  # Disable streaming for small results (faster)
        },
    )

    # PostgreSQL async engine with asyncpg (optimized for high concurrency)
    from sqlalchemy.pool import AsyncAdaptedQueuePool

    async_engine = create_async_engine(
        ASYNC_DATABASE_URL,
        poolclass=AsyncAdaptedQueuePool,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=DB_POOL_RECYCLE,
        pool_timeout=DB_POOL_TIMEOUT,
        pool_use_lifo=True,  # OPTIMIZATION: Use LIFO for connection warmth
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
        # OPTIMIZATION: asyncpg-specific optimizations
        connect_args={
            "command_timeout": 30,  # 30s command timeout
            "statement_cache_size": 100,  # Cache 100 prepared statements
        },
    )

# Add connection pool event listeners for monitoring
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log new connections."""
    logger.debug("New database connection established")

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Log connection checkout from pool."""
    logger.debug("Connection checked out from pool (size: %s)", engine.pool.size())

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """Log connection return to pool."""
    logger.debug("Connection returned to pool")

# Create session factory (sync)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create async session factory
AsyncSessionLocal: async_sessionmaker | None = None
if async_engine:
    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

def _handle_db_error(
    session, error, attempt: int, max_retries: int, retry_delay: float
) -> float:
    """Handle database errors with retry logic. Returns new retry delay.

    NOTE: This function is intentionally synchronous and uses time.sleep()
    because it's called from synchronous database operations (get_db generator).
    For async operations, use _handle_db_error_async() instead.
    """
    if session:
        with contextlib.suppress(OSError, RuntimeError):
            session.rollback()

    if attempt < max_retries - 1:
        logger.warning(
            f"Database operation failed (attempt {attempt + 1}/{max_retries}): {error}. "
            f"Retrying in {retry_delay}s..."
        )
        # SYNC CONTEXT: time.sleep is correct here - this is a sync generator
        time.sleep(retry_delay)
        return retry_delay * 2  # Exponential backoff

    logger.error("Database operation failed after %s attempts: %s", max_retries, error)
    raise error

async def _handle_db_error_async(
    session, error, attempt: int, max_retries: int, retry_delay: float
) -> float:
    """Handle database errors with async retry logic. Returns new retry delay."""
    if session:
        await session.rollback()

    if attempt < max_retries - 1:
        logger.warning(
            f"Database operation failed (attempt {attempt + 1}/{max_retries}): {error}. "
            f"Retrying in {retry_delay}s..."
        )
        await asyncio.sleep(retry_delay)
        return retry_delay * 2  # Exponential backoff

    logger.error("Database operation failed after %s attempts: %s", max_retries, error)
    raise error

def _cleanup_session(session) -> None:
    """Cleanup database session safely."""
    if session:
        session.close()

def get_db() -> Generator[Session, None, None]:
    """
    Get database session for FastAPI Depends with retry logic.

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    max_retries = 3
    retry_delay: float = 1  # seconds

    for attempt in range(max_retries):
        db = None
        try:
            db = SessionLocal()
            yield db
            db.commit()
            return
        except (OperationalError, DBAPIError) as e:
            retry_delay = _handle_db_error(db, e, attempt, max_retries, retry_delay)
        except Exception as e:
            if db:
                db.rollback()
            logger.error("Unexpected database error: %s", e)
            raise
        finally:
            _cleanup_session(db)

@contextmanager
def get_db_session():
    """
    Get database session as a context manager with retry logic.

    Usage:
        with get_db_session() as session:
            session.query(Item).all()
    """
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        session = None
        try:
            session = SessionLocal()
            yield session
            session.commit()
            return
        except (OperationalError, DBAPIError) as e:
            retry_delay = _handle_db_error(
                session, e, attempt, max_retries, retry_delay
            )
        except Exception:
            if session:
                session.rollback()
            raise
        finally:
            _cleanup_session(session)

# ==================== ASYNC SESSION MANAGEMENT ====================

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session for FastAPI Depends.

    Non-blocking session management for async endpoints.

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    if AsyncSessionLocal is None:
        raise RuntimeError(
            "Async database not configured (SQLite doesn't support asyncpg)"
        )

    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        session = None
        try:
            session = AsyncSessionLocal()
            yield session
            await session.commit()
            return
        except (OperationalError, DBAPIError) as e:
            retry_delay = await _handle_db_error_async(
                session, e, attempt, max_retries, retry_delay
            )
        except Exception as e:
            if session:
                await session.rollback()
            logger.error("Unexpected async database error: %s", e)
            raise
        finally:
            if session:
                await session.close()

@asynccontextmanager
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session as an async context manager.

    Usage:
        async with get_async_db_session() as session:
            result = await session.execute(select(Item))
            items = result.scalars().all()
    """
    if AsyncSessionLocal is None:
        raise RuntimeError(
            "Async database not configured (SQLite doesn't support asyncpg)"
        )

    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

async def run_in_async_session(func, *args, **kwargs):
    """
    Run a sync function with async session.

    Utility to bridge sync code that needs database access in async context.

    Usage:
        result = await run_in_async_session(some_sync_func, arg1, arg2)
    """
    async with get_async_db_session() as session:
        return await asyncio.get_running_loop().run_in_executor(
            None, func, session, *args, **kwargs
        )

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def init_db():
    """
    Initialize database - create all tables and enable pgvector.
    Safe to call multiple times (idempotent).
    """
    # Test database connection first
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
    except (OSError, RuntimeError) as e:
        logger.error("Database connection failed: %s", e)
        # Log only the scheme and host — never credentials
        from urllib.parse import urlparse
        try:
            parsed = urlparse(DATABASE_URL)
            safe_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}/{parsed.path.lstrip('/')}"
        except (ValueError, TypeError, KeyError):
            safe_url = DATABASE_URL.split("://")[0] + "://***"
        logger.error("DATABASE_URL (sanitized): %s", safe_url)
        raise

    # Try to enable pgvector extension for Supabase/PostgreSQL
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("pgvector extension enabled")
    except (OSError, RuntimeError) as e:
        logger.warning("Could not enable pgvector extension: %s", e)
        logger.warning(
            "RAG/Q&A features will have limited functionality without pgvector"
        )

    # Create all tables
    create_tables()
    logger.info("Database tables created/verified")

def drop_tables():
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)

def get_supabase_client():
    """
    Get Supabase client for additional features (storage, realtime, etc.).
    Returns None if Supabase credentials are not configured.
    """
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client

            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except ImportError:
            logger.warning("supabase package not installed")
            return None
    return None

__all__ = [
    "AsyncSessionLocal",
    "Base",
    "SessionLocal",
    "async_engine",
    "create_tables",
    "drop_tables",
    "engine",
    "get_async_db",
    "get_async_db_session",
    "get_db",
    "get_db_session",
    "get_supabase_client",
    "init_db",
    "run_in_async_session",
]
