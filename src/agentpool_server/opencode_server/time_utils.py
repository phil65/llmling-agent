"""Time utilities for OpenCode compatibility."""

import time


def now_ms() -> int:
    """Return current time in milliseconds as integer (OpenCode format)."""
    return int(time.time() * 1000)
