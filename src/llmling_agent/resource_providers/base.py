"""Base resource provider interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling import BasePrompt

    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.resources import ResourceInfo


logger = get_logger(__name__)


class ResourceProvider:
    """Base class for resource providers.

    Provides tools, prompts, and other resources to agents.
    Default implementations return empty lists - override as needed.
    """

    def __init__(self, name: str, owner: str | None = None):
        """Initialize the resource provider."""
        self.name = name
        self.owner = owner
        self.log = logger.bind(name=self.name, owner=self.owner)

    async def __aenter__(self) -> Self:
        """Async context entry if required."""
        return self

    async def __aexit__(self, *exc: object):
        """Async context cleanup if required."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    async def get_tools(self) -> list[Tool]:
        """Get available tools. Override to provide tools."""
        return []

    async def get_prompts(self) -> list[BasePrompt]:
        """Get available prompts. Override to provide prompts."""
        return []

    async def get_resources(self) -> list[ResourceInfo]:
        """Get available resources. Override to provide resources."""
        return []

    async def get_formatted_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> ChatMessage[str]:
        """Get a prompt formatted with arguments.

        Args:
            name: Name of the prompt to format
            arguments: Optional arguments for prompt formatting

        Returns:
            Single chat message with merged content

        Raises:
            KeyError: If prompt not found
            ValueError: If formatting fails
        """
        from llmling_agent.messaging.messages import ChatMessage

        prompts = await self.get_prompts()
        prompt = next((p for p in prompts if p.name == name), None)
        if not prompt:
            msg = f"Prompt {name!r} not found"
            raise KeyError(msg)

        messages = await prompt.format(arguments or {})
        if not messages:
            msg = f"Prompt {name!r} produced no messages"
            raise ValueError(msg)

        # Use role from first message (usually system)
        role = messages[0].role
        # Merge all message contents
        content = "\n\n".join(msg.get_text_content() for msg in messages)

        return ChatMessage(
            content=content,
            role=role,  # type: ignore
            name=self.name,
            metadata={
                "prompt_name": name,
                "arguments": arguments or {},  # type: ignore
            },
        )
