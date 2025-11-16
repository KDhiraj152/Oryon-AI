"""
Simplified database connection and session management.

Supports both local PostgreSQL and Supabase.
Uses FastAPI's Depends pattern for session management.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
from typing import Generator
from contextlib import contextmanager
import os
import logging

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

# Database configuration - supports Supabase or local PostgreSQL
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:password@localhost:5432/education_content'
)

# Supabase configuration (optional)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Get database session for FastAPI Depends.
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session():
    """
    Get database session as a context manager.
    
    Usage:
        with get_db_session() as session:
            session.query(Item).all()
    """
    return SessionLocal()


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def init_db():
    """
    Initialize database - create all tables and enable pgvector.
    Safe to call multiple times (idempotent).
    """
    try:
        # Try to enable pgvector extension for Supabase/PostgreSQL
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("pgvector extension enabled")
    except Exception as e:
        logger.warning(f"Could not enable pgvector extension: {e}")
    
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


# Legacy compatibility for existing code
class Database:
    """Legacy wrapper for backward compatibility"""
    
    def __init__(self):
        self.engine = engine
        self.session_local = SessionLocal
    
    def create_tables(self):
        create_tables()
    
    def drop_tables(self):
        drop_tables()
    
    def get_session(self):
        return self.session_local()
    
    def close_session(self):
        pass


# Global instance for legacy code
_db_instance = None

def get_db_instance() -> Database:
    """Get global database instance (legacy)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


__all__ = ['Base', 'engine', 'SessionLocal', 'get_db', 'create_tables', 'init_db', 'drop_tables', 'get_supabase_client', 'Database', 'get_db_instance']
