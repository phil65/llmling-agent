"""Claude Code storage provider.

This package implements the storage backend compatible with Claude Code's
filesystem-based persistence format.

See ARCHITECTURE.md for detailed documentation of the storage format and
design decisions.
"""

from __future__ import annotations

from agentpool_storage.claude_provider.models import (
    ClaudeApiMessage,
    ClaudeAssistantEntry,
    ClaudeBaseModel,
    ClaudeFileHistoryEntry,
    ClaudeMessageContent,
    ClaudeMessageEntryBase,
    ClaudeProgressData,
    ClaudeProgressEntry,
    ClaudeQueueOperationEntry,
    ClaudeSummaryEntry,
    ClaudeSystemEntry,
    ClaudeUsage,
    ClaudeUserEntry,
    ClaudeUserMessage,
)
from agentpool_storage.claude_provider.provider import ClaudeStorageProvider, SessionMetadata

__all__ = [
    "ClaudeApiMessage",
    "ClaudeAssistantEntry",
    "ClaudeBaseModel",
    "ClaudeFileHistoryEntry",
    "ClaudeMessageContent",
    "ClaudeMessageEntryBase",
    "ClaudeProgressData",
    "ClaudeProgressEntry",
    "ClaudeQueueOperationEntry",
    "ClaudeStorageProvider",
    "ClaudeSummaryEntry",
    "ClaudeSystemEntry",
    "ClaudeUsage",
    "ClaudeUserEntry",
    "ClaudeUserMessage",
    "SessionMetadata",
]
