"""Tool management commands."""

from __future__ import annotations

from typing import Any

from llmling.utils.importing import import_callable
from slashed import (  # noqa: TC002
    CommandContext,
    CommandError,
    CompletionContext,
    SlashedCommand,
)
from slashed.completers import CallbackCompleter

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent_commands.markdown_utils import format_table


logger = get_logger(__name__)

CODE_TEMPLATE = '''\
def my_tool(text: str) -> str:
    """A new tool.

    Args:
        text: Input text

    Returns:
        Tool result
    """
    return f"You said: {text}"
'''

TOOL_INFO_HELP = """\
Display detailed information about a specific tool:
- Source (runtime/agent/builtin)
- Current status (enabled/disabled)
- Priority and capabilities
- Parameter descriptions
- Additional metadata

Example: /tool-info open_browser
"""

WRITE_TOOL_HELP = """\
Opens an interactive Python editor to create new tools.
- ESC + Enter or Alt + Enter to save and exit
- Functions will be available as tools immediately

Example template:
def my_tool(text: str) -> str:
    '''A new tool'''
    return f'You said: {text}'
"""

REGISTER_TOOL_HELP = """\
Register a new tool from a Python import path.
Examples:
  /register-tool webbrowser.open
  /register-tool json.dumps --name format_json
  /register-tool os.getcwd --description 'Get current directory'
"""

ENABLE_TOOL_HELP = """\
Enable a previously disabled tool.
Use /list-tools to see available tools.

Example: /enable-tool open_browser
"""

DISABLE_TOOL_HELP = """\
Disable a tool to prevent its use.
Use /list-tools to see available tools.

Example: /disable-tool open_browser
"""

LIST_TOOLS_HELP = """\
Show all available tools and their current status.
Tools are grouped by source (runtime/agent/builtin).
✓ indicates enabled, ✗ indicates disabled.
"""


class ListToolsCommand(SlashedCommand):
    """Show all available tools and their current status.

    Tools are grouped by source (runtime/agent/builtin).
    ✓ indicates enabled, ✗ indicates disabled.
    """

    name = "list-tools"
    category = "tools"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        source: str | None = None,
    ):
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
        await ctx.output.print(f"## 🔧 Available Tools\n\n{table}")


class ShowToolCommand(SlashedCommand):
    """Display detailed information about a specific tool.

    Shows:
    - Source (runtime/agent/builtin)
    - Current status (enabled/disabled)
    - Priority and capabilities
    - Parameter descriptions
    - Additional metadata

    Example: /tool-info open_browser
    """

    name = "show-tool"
    category = "tools"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        name: str,
    ):
        """Show detailed information about a tool.

        Args:
            ctx: Command context
            name: Tool name to show
        """
        agent = ctx.context.agent

        try:
            tool_info = await agent.tools.get_tool(name)
            assert tool_info, f"Tool {name} not found"
            # Start with the standard tool info format
            sections = [tool_info.format_info(indent="")]

            # Add extra metadata section if we have any additional info
            extra_info = []
            if tool_info.requires_capability:
                extra_info.append(f"Required Capability: {tool_info.requires_capability}")
            if tool_info.requires_confirmation:
                extra_info.append("Requires Confirmation: Yes")
            if tool_info.source != "runtime":  # Only show if not default
                extra_info.append(f"Source: {tool_info.source}")
            if tool_info.priority != 100:  # Only show if not default  # noqa: PLR2004
                extra_info.append(f"Priority: {tool_info.priority}")
            if tool_info.metadata:
                extra_info.append("\nMetadata:")
                extra_info.extend(f"- {k}: {v}" for k, v in tool_info.metadata.items())

            if extra_info:
                sections.extend(extra_info)

            await ctx.output.print("\n".join(sections))
        except KeyError:
            await ctx.output.print(f"❌ **Tool** `{name}` **not found**")

    def get_completer(self):
        """Get completer for tool names."""
        return CallbackCompleter(get_tool_names)


class EnableToolCommand(SlashedCommand):
    """Enable a previously disabled tool.

    Use /list-tools to see available tools.

    Example: /enable-tool open_browser
    """

    name = "enable-tool"
    category = "tools"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        name: str,
    ):
        """Enable a tool.

        Args:
            ctx: Command context
            name: Tool name to enable
        """
        try:
            ctx.context.agent.tools.enable_tool(name)
            await ctx.output.print(f"✅ **Tool** `{name}` **enabled**")
        except ValueError as e:
            msg = f"Failed to enable tool: {e}"
            raise CommandError(msg) from e

    def get_completer(self):
        """Get completer for tool names."""
        return CallbackCompleter(get_tool_names)


class DisableToolCommand(SlashedCommand):
    """Disable a tool to prevent its use.

    Use /list-tools to see available tools.

    Example: /disable-tool open_browser
    """

    name = "disable-tool"
    category = "tools"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        name: str,
    ):
        """Disable a tool.

        Args:
            ctx: Command context
            name: Tool name to disable
        """
        try:
            ctx.context.agent.tools.disable_tool(name)
            await ctx.output.print(f"❌ **Tool** `{name}` **disabled**")
        except ValueError as e:
            msg = f"Failed to disable tool: {e}"
            raise CommandError(msg) from e

    def get_completer(self):
        """Get completer for tool names."""
        return CallbackCompleter(get_tool_names)


class RegisterToolCommand(SlashedCommand):
    """Register a new tool from a Python import path.

    Examples:
      /register-tool webbrowser.open
      /register-tool json.dumps --name format_json
      /register-tool os.getcwd --description 'Get current directory'
    """

    name = "register-tool"
    category = "tools"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        import_path: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ):
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
            await ctx.output.print(
                f"✅ **Tool registered successfully:**\n`{tool_info.name}`"
                f" - {tool_info.description or '*No description*'}"
            )

        except Exception as e:
            msg = f"Failed to register tool: {e}"
            raise CommandError(msg) from e


class WriteToolCommand(SlashedCommand):
    """Opens an interactive Python editor to create new tools.

    - ESC + Enter or Alt + Enter to save and exit
    - Functions will be available as tools immediately

    Example template:
    def my_tool(text: str) -> str:
        '''A new tool'''
        return f'You said: {text}'
    """

    name = "write-tool"
    category = "tools"

    async def execute_command(self, ctx: CommandContext[AgentContext]):
        """Write and register a new tool interactively.

        Args:
            ctx: Command context
        """
        from prompt_toolkit import PromptSession
        from prompt_toolkit.lexers import PygmentsLexer
        from prompt_toolkit.styles import style_from_pygments_cls
        from pygments.lexers.python import PythonLexer
        from pygments.styles import get_style_by_name

        # Create editing session with syntax highlighting
        session: PromptSession[str] = PromptSession(
            lexer=PygmentsLexer(PythonLexer),
            multiline=True,
            style=style_from_pygments_cls(get_style_by_name("monokai")),
            include_default_pygments_style=False,
            mouse_support=True,
        )
        msg = "\nEnter tool code (ESC + Enter or Alt + Enter to save):\n\n"
        code = await session.prompt_async(msg, default=CODE_TEMPLATE)
        try:
            # Execute code in a namespace
            namespace: dict[str, Any] = {}
            exec(code, namespace)

            # Find all callable non-private functions
            tools = [
                v
                for v in namespace.values()
                if callable(v)
                and not v.__name__.startswith("_")
                and v.__code__.co_filename == "<string>"
            ]

            if not tools:
                await ctx.output.print("⚠️ **No tools found in code**")
                return

            # Register all tools with ctx parameter added
            for func in tools:
                tool_info = ctx.context.agent.tools.register_tool(
                    func, source="dynamic", metadata={"created_by": "write-tool"}
                )
                tool_info.format_info()
                await ctx.output.print(
                    f"🎉 **Tool** `{tool_info.name}` **registered!**"
                    f"\n{tool_info.description or '*No description*'}"
                )

        except Exception as e:
            msg = f"Error creating tools: {e}"
            raise CommandError(msg) from e


async def get_tool_names(ctx: CompletionContext[AgentContext]) -> list[str]:
    return list(await ctx.command_context.context.agent.tools.get_tool_names())
