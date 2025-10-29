"""Task configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling import BasePrompt
from llmling.config.models import ToolConfig
from pydantic import ConfigDict, Field, ImportString
from schemez import Schema

from llmling_agent.tools.base import Tool
from llmling_agent_config.knowledge import Knowledge


if TYPE_CHECKING:
    from llmling_agent.agent import Agent


class Job[TDeps, TResult = str](Schema):
    """A task is a piece of work that can be executed by an agent.

    Requirements:
    - The agent must have compatible dependencies (required_dependency)
    - The agent must produce the specified result type (required_return_type)

    Equipment:
    - The task provides necessary tools for execution (tools)
    - Tools are temporarily available during task execution
    """

    name: str | None = Field(None)
    """Technical identifier (automatically set from config key during registration)"""

    description: str | None = None
    """Human-readable description of what this task does"""

    prompt: str | ImportString[str] | BasePrompt
    """The task instruction/prompt."""

    required_return_type: ImportString[type[TResult]] = Field(
        default="str", validate_default=True
    )  # type: ignore
    """Expected type of the task result."""

    required_dependency: ImportString[type[TDeps]] | None = Field(
        default=None, validate_default=True
    )  # type: ignore
    """Dependencies or context data needed for task execution"""

    requires_vision: bool = False
    """Whether the agent requires vision"""

    knowledge: Knowledge | None = None
    """Optional knowledge sources for this task:
    - Simple file/URL paths
    - Rich resource definitions
    - Prompt templates
    """

    tools: list[ImportString | ToolConfig] = Field(default_factory=list)
    """Tools needed for this task."""

    min_context_tokens: int | None = None
    """Minimum amount of required context size."""

    model_config = ConfigDict(frozen=True)

    async def can_be_executed_by(self, agent: Agent[Any, Any]) -> bool:
        """Check if agent meets all requirements for this task."""
        # Check dependencies
        if self.required_dependency and not isinstance(
            agent.context.data, self.required_dependency
        ):
            return False

        if agent._output_type != self.required_return_type:
            return False

        # Check vision capabilities
        if self.requires_vision:  # noqa: SIM102
            if not await agent.provider.supports_feature("vision"):
                return False

        return True

    @property
    def tool_configs(self) -> list[ToolConfig]:
        """Get all tools as ToolConfig instances."""
        return [
            tool if isinstance(tool, ToolConfig) else ToolConfig(import_path=str(tool))
            for tool in self.tools
        ]

    async def get_prompt(self) -> str:
        if isinstance(self.prompt, BasePrompt):
            messages = await self.prompt.format()
            return "\n\n".join(m.get_text_content() for m in messages)
        return self.prompt

    def get_tools(self) -> list[Tool]:
        """Get all tools as Tool instances."""
        tools: list[Tool] = []
        for tool in self.tools:
            match tool:
                case str():
                    tools.append(Tool.from_callable(tool))
                case ToolConfig(import_path=import_path):
                    tools.append(Tool.from_callable(import_path))
                case Tool():
                    tools.append(tool)
                case _:
                    msg = f"Invalid tool type: {type(tool)}"
                    raise ValueError(msg)
        return tools
