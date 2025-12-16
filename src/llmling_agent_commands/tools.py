"""Tool management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slashed import CommandContext, CommandError, CompletionContext  # noqa: TC002
from slashed.completers import CallbackCompleter

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.utils.importing import import_callable
from llmling_agent_commands.base import NodeCommand
from llmling_agent_commands.markdown_utils import format_table


if TYPE_CHECKING:
    from llmling_agent.messaging import MessageNode


logger = get_logger(__name__)


class ListToolsCommand(NodeCommand):
    """Show all available tools and their current status.

    Tools are grouped by source (runtime/agent/builtin).
    ✓ indicates enabled, ✗ indicates disabled.
    """

    name = "list-tools"
    category = "tools"

    @classmethod
    def supports_node(cls, node: MessageNode[Any, Any]) -> bool:
        from llmling_agent import Agent

        return isinstance(node, Agent)

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        source: str | None = None,
    ) -> None:
        """List all available tools.

        Args:
            ctx: Command context
            source: Optional filter by source (runtime/agent/builtin)
        """
        agent = ctx.context.agent

        rows = [
            {
                "Status": "✅" if tool_info.enabled else "❌",
                "Name": tool_info.name,
                "Source": tool_info.source,
            }
            for tool_info in await agent.tools.get_tools()
            if not source or tool_info.source == source
        ]

        headers = ["Status", "Name", "Source"]
        table = format_table(headers, rows)
        await ctx.print(f"## 🔧 Available Tools\n\n{table}")


class ShowToolCommand(NodeCommand):
    """Display detailed information about a specific tool.

    Shows:
    - Source (runtime/agent/builtin)
    - Current status (enabled/disabled)
    - Parameter descriptions
    - Additional metadata

    Example: /tool-info open_browser
    """

    name = "show-tool"
    category = "tools"

    @classmethod
    def supports_node(cls, node: MessageNode[Any, Any]) -> bool:
        from llmling_agent import Agent

        return isinstance(node, Agent)

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        name: str,
    ) -> None:
        """Show detailed information about a tool.

        Args:
            ctx: Command context
            name: Tool name to show
        """
        agent = ctx.context.agent

        try:
            tool_info = await agent.tools.get_tool(name)
            # Start with the standard tool info format
            sections = [tool_info.format_info(indent="")]

            # Add extra metadata section if we have any additional info
            extra_info = []
            if tool_info.requires_confirmation:
                extra_info.append("Requires Confirmation: Yes")
            if tool_info.source != "dynamic":  # Only show if not default
                extra_info.append(f"Source: {tool_info.source}")
            if tool_info.metadata:
                extra_info.append("\nMetadata:")
                extra_info.extend(f"- {k}: {v}" for k, v in tool_info.metadata.items())

            if extra_info:
                sections.extend(extra_info)

            await ctx.print("\n".join(sections))
        except KeyError:
            await ctx.print(f"❌ **Tool** `{name}` **not found**")

    def get_completer(self) -> CallbackCompleter:
        """Get completer for tool names."""
        return CallbackCompleter(get_tool_names)


class EnableToolCommand(NodeCommand):
    """Enable a previously disabled tool.

    Use /list-tools to see available tools.

    Example: /enable-tool open_browser
    """

    name = "enable-tool"
    category = "tools"

    @classmethod
    def supports_node(cls, node: MessageNode[Any, Any]) -> bool:
        from llmling_agent import Agent

        return isinstance(node, Agent)

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        name: str,
    ) -> None:
        """Enable a tool.

        Args:
            ctx: Command context
            name: Tool name to enable
        """
        try:
            await ctx.context.agent.tools.enable_tool(name)
            await ctx.print(f"✅ **Tool** `{name}` **enabled**")
        except ValueError as e:
            msg = f"Failed to enable tool: {e}"
            raise CommandError(msg) from e

    def get_completer(self) -> CallbackCompleter:
        """Get completer for tool names."""
        return CallbackCompleter(get_tool_names)


class DisableToolCommand(NodeCommand):
    """Disable a tool to prevent its use.

    Use /list-tools to see available tools.

    Example: /disable-tool open_browser
    """

    name = "disable-tool"
    category = "tools"

    @classmethod
    def supports_node(cls, node: MessageNode[Any, Any]) -> bool:
        from llmling_agent import Agent

        return isinstance(node, Agent)

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        name: str,
    ) -> None:
        """Disable a tool.

        Args:
            ctx: Command context
            name: Tool name to disable
        """
        try:
            await ctx.context.agent.tools.disable_tool(name)
            await ctx.print(f"❌ **Tool** `{name}` **disabled**")
        except ValueError as e:
            msg = f"Failed to disable tool: {e}"
            raise CommandError(msg) from e

    def get_completer(self) -> CallbackCompleter:
        """Get completer for tool names."""
        return CallbackCompleter(get_tool_names)


class RegisterToolCommand(NodeCommand):
    """Register a new tool from a Python import path.

    Examples:
      /register-tool webbrowser.open
      /register-tool json.dumps --name format_json
      /register-tool os.getcwd --description 'Get current directory'
    """

    name = "register-tool"
    category = "tools"

    @classmethod
    def supports_node(cls, node: MessageNode[Any, Any]) -> bool:
        from llmling_agent import Agent

        return isinstance(node, Agent)

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        import_path: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a new tool from import path or function.

        Args:
            ctx: Command context
            import_path: Python import path to the function
            name: Optional custom name for the tool
            description: Optional custom description
        """
        try:
            callable_func = import_callable(import_path)
            # Register with ToolManager
            tool_info = ctx.context.agent.tools.register_tool(
                callable_func,
                name_override=name,
                description_override=description,
                enabled=True,
                source="dynamic",
                metadata={"import_path": import_path, "registered_via": "register-tool"},
            )

            # Show the registered tool info
            tool_info.format_info()
            await ctx.print(
                f"✅ **Tool registered successfully:**\n`{tool_info.name}`"
                f" - {tool_info.description or '*No description*'}"
            )

        except Exception as e:
            msg = f"Failed to register tool: {e}"
            raise CommandError(msg) from e


async def get_tool_names(ctx: CompletionContext[AgentContext]) -> list[str]:
    manager = ctx.command_context.context.agent.tools
    return list({t.name for t in await manager.get_tools()})
