"""Provider for file access tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


def create_file_access_tools() -> list[Tool]:
    """Create tools for file and directory access operations."""
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.read_file,
            source="builtin",
            category="read",
        ),
        Tool.from_callable(
            capability_tools.list_directory,
            source="builtin",
            category="search",
        ),
    ]


class FileAccessTools(StaticResourceProvider):
    """Provider for file access tools."""

    def __init__(self, name: str = "file_access"):
        super().__init__(name=name, tools=create_file_access_tools())
