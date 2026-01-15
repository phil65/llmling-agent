"""RepoMap module for generating repository structure maps.

This module provides functionality for generating intelligent code structure maps
using tree-sitter and PageRank algorithms. It has been refactored from a single
large file into a modular structure.

Public API:
    RepoMap: Main class for generating repository maps
    Tag: Represents a code tag (definition or reference)
    get_tags_from_content: Extract tags from file content
    generate_file_outline: Generate outline for a single file
    get_file_map_from_content: Get file map from content
    is_language_supported: Check if a file extension is supported
    get_supported_languages: Get set of supported languages
    get_supported_languages_md: Get markdown table of supported languages
    is_important: Check if a file is important (should be prioritized)
    truncate_with_notice: Truncate large content with notice
    find_src_files: Find all source files in a directory
"""

from __future__ import annotations

from agentpool.repomap.core import RepoMap
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
    "RepoMap",
    "Tag",
    "find_src_files",
    "generate_file_outline",
    "get_file_map_from_content",
    "get_random_color",
    "get_supported_languages",
    "get_supported_languages_md",
    "get_tags_from_content",
    "is_important",
    "is_language_supported",
    "truncate_with_notice",
]
