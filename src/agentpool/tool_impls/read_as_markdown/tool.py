"""Read as markdown tool implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentpool.agents.context import AgentContext  # noqa: TC001
from agentpool.log import get_logger
from agentpool.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from exxec import ExecutionEnvironment

    from agentpool.prompts.conversion_manager import ConversionManager


logger = get_logger(__name__)


@dataclass
class ReadAsMarkdownTool(Tool[str | dict[str, Any]]):
    """Read files and convert them to markdown representation.

    A standalone tool for reading files with automatic conversion to markdown:
    - Documents (PDF, DOCX, etc.) converted to text
    - Structured formats (JSON, YAML, TOML) formatted as markdown
    - Binary files described in text format
    - Requires a ConversionManager for file conversion

    Use create_read_as_markdown_tool() factory for convenient instantiation.
    """

    # Tool-specific configuration
    converter: ConversionManager | None = None
    """Conversion manager for handling different file formats. Required for operation."""

    env: ExecutionEnvironment | None = None
    """Execution environment to use. Falls back to agent.env if not set."""

    cwd: str | None = None
    """Working directory for resolving relative paths."""

    def get_callable(self) -> Callable[..., Awaitable[str | dict[str, Any]]]:
        """Return the read_as_markdown method as the callable."""
        return self._read_as_markdown

    def _resolve_path(self, path: str, ctx: AgentContext) -> str:
        """Resolve a potentially relative path to an absolute path."""
        cwd: str | None = None
        if self.cwd:
            cwd = self.cwd
        elif self.env and self.env.cwd:
            cwd = self.env.cwd
        elif ctx.agent.env and ctx.agent.env.cwd:
            cwd = ctx.agent.env.cwd

        if cwd and not (path.startswith("/") or (len(path) > 1 and path[1] == ":")):
            return str(Path(cwd) / path)
        return path

    async def _read_as_markdown(
        self,
        ctx: AgentContext,
        path: str,
    ) -> str | dict[str, Any]:
        """Read file and convert to markdown text representation.

        Args:
            ctx: Agent context for event emission
            path: Path to read

        Returns:
            File content converted to markdown
        """
        if self.converter is None:
            return {"error": "Converter is required but not configured"}

        path = self._resolve_path(path, ctx)
        msg = f"Reading file as markdown: {path}"
        await ctx.events.tool_call_start(title=msg, kind="read", locations=[path])

        try:
            content = await self.converter.convert_file(path)
            await ctx.events.file_operation("read", path=path, success=True)
            # Emit formatted content for UI display
            from agentpool.agents.events import TextContentItem

            await ctx.events.tool_call_progress(
                title=f"Read as markdown: {path}",
                items=[TextContentItem(text=content)],
                replace_content=True,
            )
        except Exception as e:  # noqa: BLE001
            await ctx.events.file_operation("read", path=path, success=False, error=str(e))
            return f"Error: Failed to convert file {path}: {e}"
        else:
            return content
