"""Provider for code formatting and linting tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider

from .factories import create_code_tools


class CodeTools(StaticResourceProvider):
    """Provider for code formatting and linting tools."""

    def __init__(self, name: str = "code"):
        super().__init__(name=name, tools=create_code_tools())
