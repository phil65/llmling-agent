"""Time period parsing for CLI and API interfaces."""

from __future__ import annotations

from datetime import timedelta
import re


# Time units with their patterns
_WEEKS = r"(?P<weeks>[\d.]+)\s*(?:w|wks?|weeks?)"
_DAYS = r"(?P<days>[\d.]+)\s*(?:d|dys?|days?)"
_HOURS = r"(?P<hours>[\d.]+)\s*(?:h|hrs?|hours?)"
_MINS = r"(?P<mins>[\d.]+)\s*(?:m|mins?|minutes?)"
_SECS = r"(?P<secs>[\d.]+)\s*(?:s|secs?|seconds?)"

# Separators between units
_SEPARATORS = r"[,/]"


# Optional patterns with separators
def _OPT(x: str) -> str:  # noqa: N802
    return f"(?:{x})?"


def _OPTSEP(x: str) -> str:  # noqa: N802
    return f"(?:{x}\\s*(?:{_SEPARATORS}\\s*)?)?"


# All supported time formats
_TIME_FORMAT = (
    f"{_OPTSEP(_WEEKS)}{_OPTSEP(_DAYS)}{_OPTSEP(_HOURS)}{_OPTSEP(_MINS)}{_OPT(_SECS)}"
)

# Time unit multipliers in seconds
_MULTIPLIERS = {
    "weeks": 60 * 60 * 24 * 7,
    "days": 60 * 60 * 24,
    "hours": 60 * 60,
    "mins": 60,
    "secs": 1,
}

# Compile patterns
_SIGN_PATTERN = re.compile(r"\s*(?P<sign>[+|-])?\s*(?P<unsigned>.*$)")
_TIME_PATTERN = re.compile(rf"\s*{_TIME_FORMAT}\s*$", re.IGNORECASE)


def parse_time_period(period: str) -> timedelta:
    """Parse a time expression into a timedelta.

    Examples:
        - Simple format: 1h, 2d, 1w
        - Full words: 1 hour, 2 days, 1 week
        - Combined: 1 week 2 days 3 hours
        - With separators: 1h, 30m
        - Signed: -1h, +2d
        - Decimal values: 1.5h

    Args:
        period: Time period string to parse

    Raises:
        ValueError: If the time format is invalid

    Returns:
        Parsed time period as timedelta
    """
    # Handle sign
    sign_match = _SIGN_PATTERN.match(period)
    if not sign_match:
        msg = f"Invalid time format: {period}"
        raise ValueError(msg)

    sign = -1 if sign_match.group("sign") == "-" else 1
    unsigned = sign_match.group("unsigned")

    # Match time pattern
    if match := _TIME_PATTERN.match(unsigned):
        dct = match.groupdict()
        matches = {k: v for k, v in dct.items() if v is not None}
        try:
            secs = sum(_MULTIPLIERS[unit] * float(val) for unit, val in matches.items())
            return timedelta(seconds=sign * secs)
        except (ValueError, KeyError) as e:
            msg = f"Invalid time value in: {period}"
            raise ValueError(msg) from e

    msg = f"Unsupported time format: {period}"
    raise ValueError(msg)
