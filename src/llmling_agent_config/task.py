"""Task configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field, ImportString
from schemez import Schema

from llmling_agent.prompts.prompts import BasePrompt
from llmling_agent.tools.base import Tool
from llmling_agent_config.knowledge import Knowledge
from llmling_agent_config.tools import ImportToolConfig


if TYPE_CHECKING:
    from llmling_agent.agents import Agent


class Job[TDeps, TResult = str](Schema):
    """A task is a piece of work that can be executed by an agent.

    Requirements:
    - The agent must have compatible dependencies (required_dependency)
    - The agent must produce the specified result type (required_return_type)

    Equipment:
    - The task provides necessary tools for execution (tools)
    - Tools are temporarily available during task execution
    """

    name: str | None = Field(
        default=None,
        examples=["analyze_code", "generate_summary", "translate_text"],
        title="Task name",
    )
    """Technical identifier (automatically set from config key during registration)"""

    description: str | None = Field(
        default=None,
        examples=[
            "Analyze code for issues",
            "Generate a summary",
            "Translate text to target language",
        ],
        title="Task description",
    )
    """Human-readable description of what this task does"""

    prompt: str | ImportString[str] | BasePrompt = Field(
        examples=["Analyze the following code", "llmling_agent.prompts.prompts:BasePrompt"],
        title="Task prompt",
    )
    """The task instruction/prompt."""

    required_return_type: ImportString[type[TResult]] = Field(
        default="str",
        validate_default=True,
        examples=["str", "dict", "builtins:str"],
        title="Required return type",
    )
    """Expected type of the task result."""

    required_dependency: ImportString[type[TDeps]] | None = Field(
        default=None,
        validate_default=True,
        examples=["llmling_agent.agents.context:AgentContext"],
        title="Required dependencies",
    )
    """Dependencies or context data needed for task execution"""

    requires_vision: bool = Field(default=False, title="Requires vision capabilities")
    """Whether the agent requires vision"""

    knowledge: Knowledge | None = Field(default=None, title="Task knowledge sources")
    """Optional knowledge sources for this task:
    - Simple file/URL paths
    - Rich resource definitions
    - Prompt templates
    """

    tools: list[ImportString[Callable[..., Any]] | ImportToolConfig] = Field(
        default_factory=list,
        examples=[
            [
                {
                    "type": "import",
                    "import_path": "webbrowser:open",
                    "name": "web_browser",
                }
            ],
            ["builtins:print"],
            [
                {
                    "type": "import",
                    "import_path": "builtins:print",
                    "requires_confirmation": True,
                }
            ],
        ],
        title="Task tools",
    )
    """Tools needed for this task."""

    min_context_tokens: int | None = Field(
        default=None,
        examples=[1000, 4000, 8000],
        title="Minimum context tokens",
    )
    """Minimum amount of required context size."""

    model_config = ConfigDict(frozen=True)

    async def can_be_executed_by(self, agent: Agent[Any, Any]) -> bool:
        """Check if agent meets all requirements for this task."""
        # Check dependencies
        if self.required_dependency and not issubclass(
            agent.deps_type or type(None), self.required_dependency
        ):
            return False

        if agent._output_type != self.required_return_type:
            return False

        # Check vision capabilities
        if self.requires_vision:
            from llmling_agent.utils.model_capabilities import supports_vision

            if not await supports_vision(agent.model_name):
                return False

        return True

    @property
    def tool_configs(self) -> list[ImportToolConfig]:
        """Get all tools as ToolConfig instances."""
        return [
            tool if isinstance(tool, ImportToolConfig) else ImportToolConfig(import_path=tool)
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
                case ImportToolConfig() as config:
                    tools.append(config.get_tool())
                case Tool():
                    tools.append(tool)
                case _:
                    raise ValueError(f"Invalid tool type: {type(tool)}")
        return tools
