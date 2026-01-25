"""OpenCode storage provider.

This package implements the storage backend compatible with OpenCode's
normalized JSON file format.

See ARCHITECTURE.md for detailed documentation of the storage format and
design decisions.
"""

from __future__ import annotations

from agentpool_storage.opencode_provider.provider import (
    OpenCodeSessionMetadata,
    OpenCodeStorageProvider,
)

__all__ = [
    "OpenCodeSessionMetadata",
    "OpenCodeStorageProvider",
]
