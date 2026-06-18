"""
app/core/logging.py

Centralized logging configuration using loguru.

Provides:
- Console logging with colorized output (development friendly)
- Rotating file logs for production diagnostics
- A `get_logger(name)` helper so every module logs with a consistent format
  and can be filtered/identified by module name.
"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import get_settings

settings = get_settings()

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[module]}</cyan> | "
    "<level>{message}</level>"
)

_configured = False


def configure_logging() -> None:
    """Configure global loguru sinks. Safe to call multiple times."""
    global _configured
    if _configured:
        return

    logger.remove()

    # Console sink
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=_LOG_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=settings.APP_DEBUG,
    )

    # Rotating file sink
    log_path = Path(settings.LOG_DIR) / "phaseguard_layer2.log"
    logger.add(
        str(log_path),
        level=settings.LOG_LEVEL,
        format=_LOG_FORMAT,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        backtrace=True,
        diagnose=False,
    )

    _configured = True


def get_logger(module_name: str):
    """
    Return a logger bound with a module name for structured, filterable logs.

    Usage:
        log = get_logger(__name__)
        log.info("Something happened")
    """
    configure_logging()
    return logger.bind(module=module_name)
