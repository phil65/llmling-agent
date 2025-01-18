from __future__ import annotations

from llmling import BasePrompt, LLMCallableTool
from llmling.config.models import ToolConfig
from pydantic import BaseModel, ConfigDict, Field, ImportString
from typing_extensions import TypeVar

from llmling_agent.config.knowledge import Knowledge  # noqa: TC001


TResult = TypeVar("TResult", default=str)


class AgentTask[TDeps, TResult](BaseModel):
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

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

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

    def get_tools(self) -> list[LLMCallableTool]:
        """Get all tools as LLMCallableTool instances."""
        tools = []
        for tool in self.tools:
            match tool:
                case str():
                    tools.append(LLMCallableTool.from_callable(tool))
                case ToolConfig():
                    tools.append(LLMCallableTool.from_callable(tool.import_path))
                case LLMCallableTool():
                    tools.append(tool)
                case _:
                    msg = f"Invalid tool type: {type(tool)}"
                    raise ValueError(msg)
        return tools
