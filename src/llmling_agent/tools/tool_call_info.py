"""Tool management for LLMling agents."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field
from schemez import Schema

from llmling_agent.utils.now import get_now


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


class ToolCallInfo(Schema):
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

    timestamp: datetime = Field(default_factory=get_now)
    """When the tool was called."""

    message_id: str | None = None
    """ID of the message that triggered this tool call."""

    error: str | None = None
    """Error message if the tool call failed."""

    timing: float | None = None
    """Time taken for this specific tool call in seconds."""

    agent_tool_name: str | None = None
    """If this tool is agent-based, the name of that agent."""

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
