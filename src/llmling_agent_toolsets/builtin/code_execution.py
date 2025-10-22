"""Provider for code execution tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider

from .factories import create_code_execution_tools


class CodeExecutionTools(StaticResourceProvider):
    """Provider for code execution tools."""

    def __init__(self, name: str = "code_execution"):
        super().__init__(name=name, tools=create_code_execution_tools())
