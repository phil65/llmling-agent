"""Rendering utilities for tree-based output.

This module contains helper functions for rendering tree structures and
working with filesystem entries.
"""

from __future__ import annotations

import colorsys
import random
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem


async def is_directory(fs: AsyncFileSystem, path: str, entry_type: str = "") -> bool:
    """Check if path is a directory.

    Args:
        fs: Async filesystem instance
        path: Path to check
        entry_type: Optional pre-known entry type from listing

    Returns:
        True if path is a directory
    """
    if entry_type:
        return entry_type == "directory"

    try:
        info = await fs._info(path)
        return info.get("type") == "directory"  # type: ignore[no-any-return]
    except (OSError, FileNotFoundError):
        return False


def get_random_color() -> str:
    """Generate a random pastel color.

    Returns:
        Hex color string like '#a3b5c7'
    """
    hue = random.random()
    r, g, b = (int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75))
    return f"#{r:02x}{g:02x}{b:02x}"
