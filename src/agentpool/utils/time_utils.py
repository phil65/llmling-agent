"""Date and time utilities."""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Literal


TimeZoneMode = Literal["utc", "local"]


def get_now(tz_mode: TimeZoneMode = "utc") -> datetime:
    """Get current datetime in UTC or local timezone."""
    now = datetime.now(UTC)
    return now.astimezone() if tz_mode == "local" else now


def now_ms() -> int:
    """Return current time in milliseconds as integer."""
    return int(time.time() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime (UTC)."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def datetime_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)


def parse_iso_timestamp(value: str, *, fallback: datetime | None = None) -> datetime:
    """Parse an ISO 8601 timestamp string, handling 'Z' suffix.

    Falls back to the provided fallback or current UTC time on parse failure.

    Args:
        value: ISO timestamp string (may use 'Z' instead of '+00:00')
        fallback: Datetime to return on parse failure (defaults to current UTC time)

    Returns:
        Parsed timezone-aware datetime, or fallback on failure.
    """
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return fallback if fallback is not None else get_now()
