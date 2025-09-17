"""Rich logging utilities."""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logger with Rich handler."""

    if isinstance(level, str):
        numeric_level = logging.getLevelName(level.upper())
    else:
        numeric_level = level
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger, ensuring setup has been applied."""

    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
