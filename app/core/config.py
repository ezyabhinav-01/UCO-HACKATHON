"""
app/core/config.py

Centralized application configuration.

All configuration values are loaded from environment variables (and a local
.env file during development) using pydantic-settings. This avoids hardcoded
values scattered throughout the codebase and makes the application portable
across dev / staging / production environments and Docker containers.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "PhaseGuard-Layer2"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # --- Database ---
    POSTGRES_USER: str = "phaseguard"
    POSTGRES_PASSWORD: str = "phaseguard_secret"
    POSTGRES_DB: str = "phaseguard_db"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432

    DATABASE_URL: str = (
        "postgresql+asyncpg://phaseguard:phaseguard_secret@db:5432/phaseguard_db"
    )
    DATABASE_URL_SYNC: str = (
        "postgresql+psycopg2://phaseguard:phaseguard_secret@db:5432/phaseguard_db"
    )

    # --- ML / SpeechBrain ---
    ECAPA_MODEL_SOURCE: str = "speechbrain/spkrec-ecapa-voxceleb"
    ECAPA_MODEL_SAVE_DIR: str = "./pretrained_models/ecapa"
    EMBEDDING_DIM: int = 192
    TARGET_SAMPLE_RATE: int = 16000

    # --- Verification ---
    SIMILARITY_THRESHOLD: float = 0.65
    MIN_ENROLLMENT_RECORDINGS: int = 3

    # --- Risk Engine ---
    LAYER1_FRAUD_THRESHOLD: float = 0.70

    # --- File handling ---
    TEMP_UPLOAD_DIR: str = "./data/temp_uploads"
    MAX_UPLOAD_SIZE_MB: int = 20

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        Path(self.TEMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.ECAPA_MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache ensures the .env file is parsed only once and the same
    settings object is reused across the application (cheap dependency
    injection for FastAPI routes).
    """
    settings = Settings()
    settings.ensure_directories()
    return settings


# Convenience module-level singleton for non-FastAPI contexts (e.g. scripts,
# Alembic env.py, Streamlit app).
settings = get_settings()
