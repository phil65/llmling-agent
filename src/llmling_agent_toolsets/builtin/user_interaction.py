"""Provider for user interaction tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider

from .factories import create_user_interaction_tools


class UserInteractionTools(StaticResourceProvider):
    """Provider for user interaction tools."""

    def __init__(self, name: str = "user_interaction"):
        super().__init__(name=name, tools=create_user_interaction_tools())
