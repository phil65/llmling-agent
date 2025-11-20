"""Logging configuration for llmling_agent with structlog support."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from slashed import OutputWriter


LogLevel = int | str


_LOGGING_CONFIGURED = False


def configure_logging(
    level: LogLevel = "INFO",
    *,
    use_colors: bool | None = None,
    json_logs: bool = False,
    force: bool = False,
) -> None:
    """Configure structlog and standard logging.

    Args:
        level: Logging level
        use_colors: Whether to use colored output (auto-detected if None)
        json_logs: Force JSON output regardless of TTY detection
        force: Force reconfiguration even if already configured
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED and not force:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Determine output format first
    colors = sys.stderr.isatty() and not json_logs if use_colors is not None else False
    use_console_renderer = not (json_logs or (not colors and not sys.stderr.isatty()))

    # Configure standard logging as backend
    if use_console_renderer:
        # For console output, don't show level in stdlib logging (structlog handles it)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.basicConfig(level=level, handlers=[handler], force=True)
    else:
        # For structured output, use minimal formatting
        logging.basicConfig(
            level=level,
            handlers=[logging.StreamHandler(sys.stderr)],
            force=True,
            format="%(message)s",
        )

    # Configure structlog processors
    processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Add logger name only for non-console renderers (avoid duplication with stdlib)
    if not use_console_renderer:
        processors.insert(1, structlog.stdlib.add_logger_name)
        processors.append(structlog.processors.format_exc_info)

    # Add final renderer
    if use_console_renderer:
        processors.append(structlog.dev.ConsoleRenderer(colors=colors))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    _LOGGING_CONFIGURED = True


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
    # Ensure basic structlog configuration exists for tests
    if not _LOGGING_CONFIGURED and not structlog.is_configured():
        # Minimal configuration that doesn't interfere with stdio
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.ConsoleRenderer(colors=False),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    logger = structlog.get_logger(f"llmling_agent.{name}")
    if log_level is not None:
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())
            assert log_level
        # Set level on underlying stdlib logger
        stdlib_logger = logging.getLogger(f"llmling_agent.{name}")
        stdlib_logger.setLevel(log_level)
    return logger  # type: ignore[no-any-return]


@contextmanager
def set_handler_level(
    level: int,
    logger_names: Sequence[str],
    *,
    session_handler: OutputWriter | None = None,
) -> Iterator[None]:
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
