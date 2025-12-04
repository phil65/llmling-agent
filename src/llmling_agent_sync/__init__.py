"""File sync system for tracking dependencies and triggering reconciliation.

This module provides a system for:
- Defining file dependencies via in-file metadata
  (PEP723-style for Python, frontmatter for Markdown)
- Tracking changes via git
- Triggering LLM-powered reconciliation when dependencies change
"""

from __future__ import annotations

from llmling_agent_sync.git import GitError, GitRepo
from llmling_agent_sync.handlers import AgentReconciler, LoggingHandler
from llmling_agent_sync.manager import InitMode, SyncManager
from llmling_agent_sync.models import FileChange, SyncMetadata, UrlChange
from llmling_agent_sync.resources import ResourceChange, ResourceRegistry, ResourceState
from llmling_agent_sync.parsers import (
    BUILTIN_PARSERS,
    MarkdownSyncParser,
    MetadataParser,
    PythonSyncParser,
    get_parser_for_file,
)

__all__ = [
    # Parsers
    "BUILTIN_PARSERS",
    # Handlers
    "AgentReconciler",
    # Models
    "FileChange",
    # Git
    "GitError",
    "GitRepo",
    # Core
    "InitMode",
    "LoggingHandler",
    "MarkdownSyncParser",
    "MetadataParser",
    "PythonSyncParser",
    # Resources
    "ResourceChange",
    "ResourceRegistry",
    "ResourceState",
    "SyncManager",
    "SyncMetadata",
    "UrlChange",
    "get_parser_for_file",
]
