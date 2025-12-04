"""File sync system for tracking dependencies and triggering reconciliation.

This module provides a system for:
- Defining file dependencies via in-file metadata
  (PEP723-style for Python, frontmatter for Markdown)
- Tracking changes via git
- Tracking external URL resources
- Tracking Python package version changes
- Triggering LLM-powered reconciliation when dependencies change
"""

from __future__ import annotations

from llmling_agent_sync.config import (
    FileSyncConfig,
    PackageSyncConfig,
    SyncConfig,
    UrlSyncConfig,
)
from llmling_agent_sync.git import GitError, GitRepo
from llmling_agent_sync.handlers import AgentReconciler, LoggingHandler
from llmling_agent_sync.manager import InitMode, SyncManager
from llmling_agent_sync.models import FileChange, SyncMetadata, UrlChange
from llmling_agent_sync.packages import (
    PackageChange,
    PackageRegistry,
    PackageState,
    is_significant_bump,
)
from llmling_agent_sync.parsers import (
    BUILTIN_PARSERS,
    MarkdownSyncParser,
    MetadataParser,
    PythonSyncParser,
    get_parser_for_file,
)
from llmling_agent_sync.resources import ResourceChange, ResourceRegistry, ResourceState

__all__ = [
    # Parsers
    "BUILTIN_PARSERS",
    # Handlers
    "AgentReconciler",
    # Models
    "FileChange",
    # Config
    "FileSyncConfig",
    # Git
    "GitError",
    "GitRepo",
    # Core
    "InitMode",
    "LoggingHandler",
    "MarkdownSyncParser",
    "MetadataParser",
    # Packages
    "PackageChange",
    "PackageRegistry",
    "PackageState",
    "PackageSyncConfig",
    "PythonSyncParser",
    # Resources
    "ResourceChange",
    "ResourceRegistry",
    "ResourceState",
    "SyncConfig",
    "SyncManager",
    "SyncMetadata",
    "UrlChange",
    "UrlSyncConfig",
    "get_parser_for_file",
    "is_significant_bump",
]
