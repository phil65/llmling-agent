"""RepoMap module for generating repository structure maps.

This module provides functionality for generating intelligent code structure maps
using tree-sitter and PageRank algorithms. It has been refactored from a single
large file into a modular structure.

Public API:
    RepoMap: Main class for generating repository maps
    Tag: Represents a code tag (definition or reference)

    # Context generation (high-level)
    get_resource_context: Get context for any resource (file or directory)
    generate_directory_context: Generate repository map for directory
    generate_file_context: Generate outline or content for file

    # File outline generation (low-level)
    generate_file_outline: Generate outline for a single file
    get_file_map_from_content: Get file map from content

    # Tag extraction
    get_tags_from_content: Extract tags from file content

    # Language support
    is_language_supported: Check if a file extension is supported
    get_supported_languages: Get set of supported languages
    get_supported_languages_md: Get markdown table of supported languages

    # Utilities
    is_important: Check if a file is important (should be prioritized)
    truncate_with_notice: Truncate large content with notice
    find_src_files: Find all source files in a directory
    get_random_color: Generate random pastel color
"""

from __future__ import annotations

from agentpool.repomap.context import (
    generate_directory_context,
    generate_file_context,
    get_resource_context,
)
from agentpool.repomap.core import RepoMap
from agentpool.repomap.types import FileInfo, RepoMapResult, TokenCounter
from agentpool.repomap.languages import (
    get_supported_languages,
    get_supported_languages_md,
    is_language_supported,
)
from agentpool.repomap.outline import generate_file_outline, get_file_map_from_content
from agentpool.repomap.rendering import get_random_color
from agentpool.repomap.tags import Tag, get_tags_from_content
from agentpool.repomap.utils import find_src_files, is_important, truncate_with_notice

__all__ = [
    "FileInfo",
    "RepoMap",
    "RepoMapResult",
    "Tag",
    "TokenCounter",
    "find_src_files",
    "generate_directory_context",
    "generate_file_context",
    "generate_file_outline",
    "get_file_map_from_content",
    "get_random_color",
    "get_resource_context",
    "get_supported_languages",
    "get_supported_languages_md",
    "get_tags_from_content",
    "is_important",
    "is_language_supported",
    "truncate_with_notice",
]
