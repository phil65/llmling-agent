"""Base resource provider interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from llmling.core.typedefs import MessageContent
from pydantic_ai import BinaryContent, SystemPromptPart, UserPromptPart
from pydantic_ai.messages import ImageUrl

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from types import TracebackType

    from llmling import BasePrompt
    from pydantic_ai import ModelRequestPart

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

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
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

    async def get_request_parts(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> list[ModelRequestPart]:
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
        prompts = await self.get_prompts()
        prompt = next((p for p in prompts if p.name == name), None)
        if not prompt:
            msg = f"Prompt {name!r} not found"
            raise KeyError(msg)

        messages = await prompt.format(arguments or {})
        if not messages:
            msg = f"Prompt {name!r} produced no messages"
            raise ValueError(msg)

        parts: list[ModelRequestPart] = []
        for prompt_msg in messages:
            match prompt_msg.role:
                case "system":
                    parts.append(SystemPromptPart(str(prompt_msg.content)))
                case "user":
                    match prompt_msg.content:
                        case str():
                            parts.append(UserPromptPart(prompt_msg.content))
                        case MessageContent():
                            parts.append(to_pydantic(prompt_msg.content))
                        case list():
                            items = [to_pydantic(i) for i in prompt_msg.content]
                            parts.extend(items)
        return parts


def to_pydantic(content: MessageContent) -> UserPromptPart:
    """Convert MessageContent to Pydantic model."""
    if content.type == "text":
        return UserPromptPart(content.content)
    if content.type == "image_url":
        return UserPromptPart([ImageUrl(content.content)])
    if content.type == "image_base64":
        return UserPromptPart([
            BinaryContent(content.content.encode(), media_type="image/jpeg")
        ])
    msg = "Unsupported content type"
    raise ValueError(msg)
