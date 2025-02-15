"""Models for tools."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ImportString


if TYPE_CHECKING:
    from llmling_agent.tools.base import Tool


FormatStyle = Literal["simple", "detailed", "markdown"]

SIMPLE_TEMPLATE = """{{ tool_name }}(
    {%- for name, value in args.items() -%}
        {{ name }}={{ value|repr }}{{ "," if not loop.last }}
    {%- endfor -%}
) -> {{ error if error else result }}"""

DEFAULT_TEMPLATE = """Tool Call: {{ tool_name }}
Arguments:
{%- for name, value in args.items() %}
  {{ name }}: {{ value|repr }}
{%- endfor %}
{%- if result %}

Result: {{ result }}
{%- endif %}
{%- if error %}

Error: {{ error }}
{%- endif %}"""

MARKDOWN_TEMPLATE = """### Tool Call: {{ tool_name }}

**Arguments:**
{% for name, value in args.items() %}
- {{ name }}: {{ value|repr }}
{%- endfor %}

{%- if error %}

**Error:** {{ error }}
{%- endif %}
{%- if result %}

**Result:**
```
{{ result }}
```
{%- endif %}

{%- if timing %}
*Execution time: {{ "%.2f"|format(timing) }}s*
{%- endif %}
{%- if agent_tool_name %}
*Agent: {{ agent_tool_name }}*
{%- endif %}"""


TEMPLATES = {
    "simple": SIMPLE_TEMPLATE,
    "detailed": DEFAULT_TEMPLATE,
    "markdown": MARKDOWN_TEMPLATE,
}


class BaseToolConfig(BaseModel):
    """Base configuration for agent tools."""

    type: str = Field(init=False)
    """Type discriminator for tool configs."""

    name: str | None = None
    """Optional override for the tool name."""

    description: str | None = None
    """Optional override for the tool description."""

    enabled: bool = True
    """Whether this tool is initially enabled."""

    requires_confirmation: bool = False
    """Whether tool execution needs confirmation."""

    requires_capability: str | None = None
    """Optional capability needed to use the tool."""

    priority: int = 100
    """Execution priority (lower = higher priority)."""

    cache_enabled: bool = False
    """Whether to enable result caching."""

    metadata: dict[str, str] = Field(default_factory=dict)
    """Additional tool metadata."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    def get_tool(self) -> Tool:
        """Convert config to Tool instance."""
        raise NotImplementedError


class ImportToolConfig(BaseToolConfig):
    """Configuration for importing tools from Python modules."""

    type: Literal["import"] = Field("import", init=False)
    """Import path based tool."""

    import_path: ImportString[Callable[..., Any]]
    """Import path to the tool function."""

    def get_tool(self) -> Tool:
        """Import and create tool from configuration."""
        from llmling_agent.tools.base import Tool

        return Tool.from_callable(
            self.import_path,
            name_override=self.name,
            description_override=self.description,
            enabled=self.enabled,
            requires_confirmation=self.requires_confirmation,
            requires_capability=self.requires_capability,
            priority=self.priority,
            cache_enabled=self.cache_enabled,
            metadata=self.metadata,
        )


class CrewAIToolConfig(BaseToolConfig):
    """Configuration for CrewAI-based tools."""

    type: Literal["crewai"] = Field("crewai", init=False)
    """CrewAI tool configuration."""

    import_path: ImportString
    """Import path to CrewAI tool class."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Tool-specific parameters."""

    def get_tool(self) -> Tool:
        """Import and create CrewAI tool."""
        from llmling_agent.tools.base import Tool

        try:
            return Tool.from_crewai_tool(
                self.import_path(**self.params),
                name_override=self.name,
                description_override=self.description,
                enabled=self.enabled,
                requires_confirmation=self.requires_confirmation,
                requires_capability=self.requires_capability,
                priority=self.priority,
                cache_enabled=self.cache_enabled,
                metadata={"type": "crewai", **self.metadata},
            )
        except ImportError as e:
            msg = "CrewAI not installed. Install with: pip install crewai-tools"
            raise ImportError(msg) from e


class LangChainToolConfig(BaseToolConfig):
    """Configuration for LangChain tools."""

    type: Literal["langchain"] = Field("langchain", init=False)
    """LangChain tool configuration."""

    tool_name: str
    """Name of LangChain tool to use."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Tool-specific parameters."""

    def get_tool(self) -> Tool:
        """Import and create LangChain tool."""
        try:
            from langchain.tools import load_tool

            from llmling_agent.tools.base import Tool

            return Tool.from_langchain_tool(
                load_tool(self.tool_name, **self.params),
                name_override=self.name,
                description_override=self.description,
                enabled=self.enabled,
                requires_confirmation=self.requires_confirmation,
                requires_capability=self.requires_capability,
                priority=self.priority,
                cache_enabled=self.cache_enabled,
                metadata={"type": "langchain", **self.metadata},
            )
        except ImportError as e:
            msg = "LangChain not installed. Install with: pip install langchain"
            raise ImportError(msg) from e


# Union type for tool configs
ToolConfig = Annotated[
    ImportToolConfig | CrewAIToolConfig | LangChainToolConfig,
    Field(discriminator="type"),
]


class ToolCallInfo(BaseModel):
    """Information about an executed tool call."""

    tool_name: str
    """Name of the tool that was called."""

    args: dict[str, Any]
    """Arguments passed to the tool."""

    result: Any
    """Result returned by the tool."""

    agent_name: str
    """Name of the calling agent."""

    tool_call_id: str = Field(default_factory=lambda: str(uuid4()))
    """ID provided by the model (e.g. OpenAI function call ID)."""

    timestamp: datetime = Field(default_factory=datetime.now)
    """When the tool was called."""

    message_id: str | None = None
    """ID of the message that triggered this tool call."""

    context_data: Any | None = None
    """Optional context data that was passed to the agent's run() method."""

    error: str | None = None
    """Error message if the tool call failed."""

    timing: float | None = None
    """Time taken for this specific tool call in seconds."""

    agent_tool_name: str | None = None
    """If this tool is agent-based, the name of that agent."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    def format(
        self,
        style: FormatStyle = "simple",
        *,
        template: str | None = None,
        variables: dict[str, Any] | None = None,
        show_timing: bool = True,
        show_ids: bool = False,
    ) -> str:
        """Format tool call information with configurable style.

        Args:
            style: Predefined style to use:
                - simple: Compact single-line format
                - detailed: Multi-line with all details
                - markdown: Formatted markdown with syntax highlighting
            template: Optional custom template (required if style="custom")
            variables: Additional variables for template rendering
            show_timing: Whether to include execution timing
            show_ids: Whether to include tool_call_id and message_id

        Returns:
            Formatted tool call information

        Raises:
            ValueError: If style is invalid or custom template is missing
        """
        from jinjarope import Environment

        # Select template
        if template:
            template_str = template
        elif style in TEMPLATES:
            template_str = TEMPLATES[style]
        else:
            msg = f"Invalid style: {style}"
            raise ValueError(msg)

        # Prepare template variables
        vars_ = {
            "tool_name": self.tool_name,
            "args": self.args,  # No pre-formatting needed
            "result": self.result,
            "error": self.error,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "timing": self.timing if show_timing else None,
            "agent_tool_name": self.agent_tool_name,
        }

        if show_ids:
            vars_.update({
                "tool_call_id": self.tool_call_id,
                "message_id": self.message_id,
            })

        if variables:
            vars_.update(variables)

        # Render template
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        env.filters["repr"] = repr  # Add repr filter
        template_obj = env.from_string(template_str)
        return template_obj.render(**vars_)
