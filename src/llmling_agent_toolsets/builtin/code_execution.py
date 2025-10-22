"""Provider for code execution tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


def create_code_execution_tools() -> list[Tool]:
    """Create tools for code execution operations."""
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.execute_python,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.execute_command,
            source="builtin",
            category="execute",
        ),
    ]


class CodeExecutionTools(StaticResourceProvider):
    """Provider for code execution tools."""

    def __init__(self, name: str = "code_execution"):
        super().__init__(name=name, tools=create_code_execution_tools())
