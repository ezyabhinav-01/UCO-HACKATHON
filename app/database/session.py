"""
app/database/session.py

Database engine and session management.

Uses SQLAlchemy 2.0's async engine with asyncpg as the driver. Provides a
FastAPI dependency `get_db` that yields an `AsyncSession` per request and
guarantees the session is closed afterwards.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
log = get_logger(__name__)

# The async engine maintains a connection pool to PostgreSQL.
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_DEBUG,
    pool_pre_ping=True,
    future=True,
)

# Factory for creating new AsyncSession objects bound to the engine.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session.

    Ensures the session is properly closed after the request finishes,
    and that any uncommitted changes are rolled back if an exception
    propagates out of the request handler.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            log.exception("Database session rolled back due to an exception")
            raise
        finally:
            await session.close()
