"""Read as markdown tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from agentpool.tool_impls.read_as_markdown.tool import ReadAsMarkdownTool
from agentpool_config.tools import ToolHints


if TYPE_CHECKING:
    from exxec import ExecutionEnvironment

    from agentpool.prompts.conversion_manager import ConversionManager

__all__ = ["ReadAsMarkdownTool", "create_read_as_markdown_tool"]

# Tool metadata defaults
NAME = "read_as_markdown"
DESCRIPTION = """Read a file and convert it to markdown text representation.

Supports:
- Documents (PDF, DOCX) converted to text
- Structured formats (JSON, YAML, TOML) formatted as markdown
- Binary files described in text format
- Automatic format detection and conversion"""
CATEGORY: Literal["read"] = "read"
HINTS = ToolHints(read_only=True, idempotent=True)


def create_read_as_markdown_tool(
    *,
    converter: ConversionManager,
    env: ExecutionEnvironment | None = None,
    cwd: str | None = None,
    name: str = NAME,
    description: str = DESCRIPTION,
    requires_confirmation: bool = False,
) -> ReadAsMarkdownTool:
    """Create a configured ReadAsMarkdownTool instance.

    Args:
        converter: Conversion manager for handling different file formats (required).
        env: Execution environment to use. Falls back to agent.env if not set.
        cwd: Working directory for resolving relative paths.
        name: Tool name override.
        description: Tool description override.
        requires_confirmation: Whether tool execution needs confirmation.

    Returns:
        Configured ReadAsMarkdownTool instance.

    Example:
        # Basic usage
        from agentpool.prompts.conversion_manager import ConversionManager

        converter = ConversionManager()
        read_md = create_read_as_markdown_tool(converter=converter)

        # With specific environment
        read_md = create_read_as_markdown_tool(
            converter=converter,
            env=my_env,
            cwd="/workspace",
        )
    """
    return ReadAsMarkdownTool(
        name=name,
        description=description,
        category=CATEGORY,
        hints=HINTS,
        converter=converter,
        env=env,
        cwd=cwd,
        requires_confirmation=requires_confirmation,
    )
