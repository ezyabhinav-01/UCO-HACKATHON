"""
app/database/base.py

Declarative base class for all SQLAlchemy ORM models.

All model classes in app/models/ inherit from `Base`. Importing this module
also serves as the single source of truth that Alembic's env.py points to
when autogenerating migrations (Base.metadata).
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models in PhaseGuard."""

    pass
