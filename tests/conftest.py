"""
tests/conftest.py

Shared pytest fixtures and configuration.

These tests focus on units that do NOT require a live PostgreSQL instance
or downloading the ECAPA-TDNN model, so they can run quickly in CI without
external dependencies. Integration tests against a real database / model
should be added separately (e.g. using a docker-compose-based test
environment) and marked accordingly.
"""

import os
import sys
from pathlib import Path

# Ensure the project root is importable as `app.*` regardless of where
# pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use sensible defaults for settings during tests so importing app.core.config
# doesn't require a real .env file to be present.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://phaseguard:phaseguard_secret@localhost:5432/phaseguard_test",
)
os.environ.setdefault(
    "DATABASE_URL_SYNC",
    "postgresql+psycopg2://phaseguard:phaseguard_secret@localhost:5432/phaseguard_test",
)
os.environ.setdefault("TEMP_UPLOAD_DIR", "./data/temp_uploads_test")
os.environ.setdefault("LOG_DIR", "./logs_test")
os.environ.setdefault("ECAPA_MODEL_SAVE_DIR", "./pretrained_models_test/ecapa")
