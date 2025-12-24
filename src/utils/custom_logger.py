"""Custom logger for the project.

Provides a `get_logger(name: str)` function that returns a configured logger
that writes to a rotating file in the project's `logs/` directory and to
stdout.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

# Directory for logs: relative `logs/` under current working directory
_LOG_DIR: Path = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE: Path = _LOG_DIR / "app.log"

_DEFAULT_LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _make_file_handler() -> RotatingFileHandler:
    handler = RotatingFileHandler(
        filename=str(_LOG_FILE),
        mode="a",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, _DEFAULT_DATE_FORMAT))
    return handler


def _make_stream_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, _DEFAULT_DATE_FORMAT))
    return handler


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger.

    The logger will have a rotating file handler that writes to `logs/app.log`
    and a stream handler that outputs to stdout. Calling this function
    repeatedly for the same `name` will not add duplicate handlers.
    """
    logger_name = name or "app"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Add file handler and stream handler once
        logger.addHandler(_make_file_handler())
        logger.addHandler(_make_stream_handler())

    return logger


# Module-level default logger
LOGGER = get_logger("app")


if __name__ == "__main__":
    LOGGER.info("Logger initialized. Log file: %s", str(_LOG_FILE))