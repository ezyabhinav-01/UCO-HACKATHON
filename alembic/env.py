"""
alembic/env.py

Alembic environment configuration for PhaseGuard - Layer 2.

Uses the synchronous database URL (DATABASE_URL_SYNC, psycopg2 driver) from
app.core.config.settings, and points Alembic's autogeneration at
app.database.base.Base.metadata (which aggregates all models imported via
app/models/__init__.py).
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make sure all models are imported so Base.metadata is fully populated.
from app.core.config import get_settings
from app.database.base import Base
from app.models import *  # noqa: F401,F403 - ensures model metadata is registered

settings = get_settings()

# this is the Alembic Config object, which provides access to the values
# within the .ini file in use.
config = context.config

# Inject the synchronous database URL from application settings so a single
# source of truth (the .env file) drives both the app and migrations.
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL_SYNC)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL without a DB connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (executes against a live DB connection)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
