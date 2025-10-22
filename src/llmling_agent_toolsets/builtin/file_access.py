"""Provider for file access tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider

from .factories import create_file_access_tools


class FileAccessTools(StaticResourceProvider):
    """Provider for file access tools."""

    def __init__(self, name: str = "file_access"):
        super().__init__(name=name, tools=create_file_access_tools())
