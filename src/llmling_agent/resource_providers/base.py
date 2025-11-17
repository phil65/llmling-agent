"""Base resource provider interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from types import TracebackType

    from pydantic_ai import ModelRequestPart

    from llmling_agent.prompts.prompts import BasePrompt
    from llmling_agent.skills.skill import Skill
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.resources import ResourceInfo


logger = get_logger(__name__)


class ResourceProvider:
    """Base class for resource providers.

    Provides tools, prompts, and other resources to agents.
    Default implementations return empty lists - override as needed.
    """

    def __init__(self, name: str, owner: str | None = None) -> None:
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
    ) -> None:
        """Async context cleanup if required."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    async def get_tools(self) -> list[Tool]:
        """Get available tools. Override to provide tools."""
        return []

    async def get_tool(self, tool_name: str) -> Tool:
        """Get specific tool."""
        tools = await self.get_tools()
        for tool in tools:
            if tool.name == tool_name:
                return tool
        msg = f"Tool {tool_name!r} not found"
        raise ValueError(msg)

    async def get_prompts(self) -> list[BasePrompt]:
        """Get available prompts. Override to provide prompts."""
        return []

    async def get_resources(self) -> list[ResourceInfo]:
        """Get available resources. Override to provide resources."""
        return []

    async def get_skills(self) -> list[Skill]:
        """Get available skills. Override to provide skills."""
        return []

    async def get_skill_instructions(self, skill_name: str) -> str:
        """Get full instructions for a specific skill.

        Args:
            skill_name: Name of the skill to get instructions for

        Returns:
            The full skill instructions for execution

        Raises:
            KeyError: If skill not found
        """
        msg = f"Skill {skill_name!r} not found"
        raise KeyError(msg)

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

        return [p for prompt_msg in messages for p in prompt_msg.to_pydantic_parts()]
