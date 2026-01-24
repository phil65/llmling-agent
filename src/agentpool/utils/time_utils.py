"""Date and time utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal


TimeZoneMode = Literal["utc", "local"]


def get_now(tz_mode: TimeZoneMode = "utc") -> datetime:
    """Get current datetime in UTC or local timezone.

    Args:
        tz_mode: "utc" or "local" (default: "utc")

    Returns:
        Timezone-aware datetime object
    """
    now = datetime.now(UTC)
    return now.astimezone() if tz_mode == "local" else now


def parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp string, handling 'Z' suffix.

    Falls back to current UTC time on parse failure.

    Args:
        value: ISO timestamp string (may use 'Z' instead of '+00:00')

    Returns:
        Parsed timezone-aware datetime, or current time on failure.
    """
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return get_now()
