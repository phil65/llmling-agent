"""Logging configuration for llmling_agent with structlog support."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog


if TYPE_CHECKING:
    from collections.abc import Sequence

    from slashed import OutputWriter


LogLevel = int | str


def configure_logging(
    level: LogLevel = "INFO",
    *,
    use_colors: bool | None = None,
    json_logs: bool = False,
) -> None:
    """Configure structlog and standard logging.

    Args:
        level: Logging level
        use_colors: Whether to use colored output (auto-detected if None)
        json_logs: Force JSON output regardless of TTY detection
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Configure standard logging as backend
    logging.basicConfig(
        level=level,
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
        format="%(message)s",  # structlog handles formatting
    )

    # Determine output format
    if use_colors is None:
        use_colors = sys.stderr.isatty() and not json_logs

    # Configure structlog processors
    processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add final renderer
    if json_logs or (not use_colors and not sys.stderr.isatty()):
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=use_colors))

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(
    name: str, log_level: LogLevel | None = None
) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger for the given name.

    Args:
        name: The name of the logger, will be prefixed with 'llmling_agent.'
        log_level: The logging level to set for the logger

    Returns:
        A structlog BoundLogger instance
    """
    logger = structlog.get_logger(f"llmling_agent.{name}")
    if log_level is not None:
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())
        # Set level on underlying stdlib logger
        stdlib_logger = logging.getLogger(f"llmling_agent.{name}")
        stdlib_logger.setLevel(log_level)
    return logger


@contextmanager
def set_handler_level(
    level: int,
    logger_names: Sequence[str],
    *,
    session_handler: OutputWriter | None = None,
):
    """Temporarily set logging level and optionally add session handler.

    Args:
        level: Logging level to set
        logger_names: Names of loggers to configure
        session_handler: Optional output writer for session logging
    """
    loggers = [logging.getLogger(name) for name in logger_names]
    old_levels = [logger.level for logger in loggers]

    handler = None
    if session_handler:
        from slashed.log import SessionLogHandler

        handler = SessionLogHandler(session_handler)
        for logger in loggers:
            logger.addHandler(handler)

    try:
        for logger in loggers:
            logger.setLevel(level)
        yield
    finally:
        for logger, old_level in zip(loggers, old_levels, strict=True):
            logger.setLevel(old_level)
            if handler:
                logger.removeHandler(handler)
