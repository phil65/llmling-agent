"""Provider for user interaction tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


def create_user_interaction_tools() -> list[Tool]:
    """Create tools for user interaction operations."""
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.ask_user,
            source="builtin",
            category="other",
        ),
    ]


class UserInteractionTools(StaticResourceProvider):
    """Provider for user interaction tools."""

    def __init__(self, name: str = "user_interaction"):
        super().__init__(name=name, tools=create_user_interaction_tools())
