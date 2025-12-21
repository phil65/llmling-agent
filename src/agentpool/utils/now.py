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
