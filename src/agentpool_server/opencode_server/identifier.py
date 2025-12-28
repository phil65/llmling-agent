"""OpenCode-compatible identifier generation.

Generates IDs that are lexicographically sortable by creation time.
Format: {prefix}_{hex_timestamp}{random_base62}
"""

from __future__ import annotations

import secrets
import time
from typing import Literal


PrefixType = Literal["session", "message", "permission", "user", "part", "pty"]

PREFIXES: dict[PrefixType, str] = {
    "session": "ses",
    "message": "msg",
    "permission": "per",
    "user": "usr",
    "part": "prt",
    "pty": "pty",
}

BASE62_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
ID_LENGTH = 26

# State for monotonic ID generation
_last_timestamp = 0
_counter = 0


def _random_base62(length: int) -> str:
    """Generate random base62 string."""
    return "".join(secrets.choice(BASE62_CHARS) for _ in range(length))


def ascending(prefix: PrefixType, given: str | None = None) -> str:
    """Generate an ascending (chronologically sortable) ID.

    Args:
        prefix: The type prefix for the ID
        given: If provided, validate and return this ID instead of generating

    Returns:
        A sortable ID with the format {prefix}_{hex_timestamp}{random}
    """
    if given is not None:
        expected_prefix = PREFIXES[prefix]
        if not given.startswith(expected_prefix):
            msg = f"ID {given} does not start with {expected_prefix}"
            raise ValueError(msg)
        return given

    return _create(prefix, descending=False)


def _create(prefix: PrefixType, *, descending: bool = False) -> str:
    """Create a new ID with timestamp encoding.

    Args:
        prefix: The type prefix
        descending: If True, invert the timestamp for reverse sorting

    Returns:
        A new ID string
    """
    global _last_timestamp, _counter  # noqa: PLW0603

    current_timestamp = int(time.time() * 1000)  # milliseconds

    if current_timestamp != _last_timestamp:
        _last_timestamp = current_timestamp
        _counter = 0
    _counter += 1

    # Combine timestamp and counter (matches OpenCode's encoding)
    now = current_timestamp * 0x1000 + _counter

    if descending:
        now = ~now & 0xFFFFFFFFFFFF  # Invert for descending order (48 bits)

    # Extract bytes using OpenCode's method (big-endian, 6 bytes from positions 40,32,24,16,8,0)
    time_bytes = bytearray(6)
    for i in range(6):
        time_bytes[i] = (now >> (40 - 8 * i)) & 0xFF

    time_hex = time_bytes.hex()

    # Add random suffix
    random_suffix = _random_base62(ID_LENGTH - 12)

    return f"{PREFIXES[prefix]}_{time_hex}{random_suffix}"
