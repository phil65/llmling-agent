"""Command utilities."""

from __future__ import annotations

import importlib.util
import webbrowser

from anyenv.text_sharing import TextSharerStr, Visibility  # noqa: TC002
from slashed import CommandContext, CommandError  # noqa: TC002

from llmling_agent.messaging.context import NodeContext  # noqa: TC001
from llmling_agent_commands.base import NodeCommand


class CopyClipboardCommand(NodeCommand):
    """Copy messages from conversation history to system clipboard.

    Allows copying a configurable number of messages with options for:
    - Number of messages to include
    - Including/excluding system messages
    - Token limit for context size
    - Custom format templates

    Requires clipman package to be installed.
    """

    name = "copy-clipboard"
    category = "utils"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext],
        *,
        num_messages: int = 1,
        include_system: bool = False,
        max_tokens: int | None = None,
        format_template: str | None = None,
    ) -> None:
        """Copy messages to clipboard.

        Args:
            ctx: Command context
            num_messages: Number of messages to copy (default: 1)
            include_system: Include system messages
            max_tokens: Only include messages up to token limit
            format_template: Custom format template
        """
        try:
            import clipman  # type: ignore[import-untyped]
        except ImportError as e:
            msg = "clipman package required for clipboard operations"
            raise CommandError(msg) from e

        content = await ctx.context.agent.conversation.format_history(
            num_messages=num_messages,
            include_system=include_system,
            max_tokens=max_tokens,
            format_template=format_template,
        )

        if not content.strip():
            await ctx.print("ℹ️ **No messages found to copy**")  #  noqa: RUF001
            return

        try:
            clipman.init()
            clipman.copy(content)
            await ctx.print("📋 **Messages copied to clipboard**")
        except Exception as e:
            msg = f"Failed to copy to clipboard: {e}"
            raise CommandError(msg) from e

    @classmethod
    def condition(cls) -> bool:
        """Check if clipman is available."""
        return importlib.util.find_spec("clipman") is not None


class EditAgentFileCommand(NodeCommand):
    """Open the agent's configuration file in your default editor.

    This file contains:
    - Agent settings and capabilities
    - System promptss
    - Model configuration
    - Environment references
    - Role definitions

    Note: Changes to the configuration file require reloading the agent.
    """

    name = "open-agent-file"
    category = "utils"

    async def execute_command(self, ctx: CommandContext[NodeContext]) -> None:
        """Open agent's configuration file."""
        agent = ctx.context.agent
        if not agent.context:
            msg = "No agent context available"
            raise CommandError(msg)

        config = agent.context.config
        if not config.config_file_path:
            msg = "No configuration file path available"
            raise CommandError(msg)

        try:
            webbrowser.open(config.config_file_path)
            msg = f"🌐 **Opening agent configuration:** `{config.config_file_path}`"
            await ctx.print(msg)
        except Exception as e:
            msg = f"Failed to open configuration file: {e}"
            raise CommandError(msg) from e


class ShareHistoryCommand(NodeCommand):
    """Share message history using various text sharing providers.

    Supports multiple providers:
    - gist: GitHub Gist (requires GITHUB_TOKEN or GH_TOKEN)
    - pastebin: Pastebin (requires PASTEBIN_API_KEY)
    - paste_rs: paste.rs (no authentication required)
    - opencode: OpenCode (no authentication required)

    The shared content can be conversation history or custom text.
    """

    name = "share-text"
    category = "utils"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext],
        *,
        provider: TextSharerStr = "paste_rs",
        num_messages: int = 1,
        include_system: bool = False,
        max_tokens: int | None = None,
        format_template: str | None = None,
        title: str | None = None,
        syntax: str = "markdown",
        visibility: Visibility = "unlisted",
    ) -> None:
        """Share text content via a text sharing provider.

        Args:
            ctx: Command context
            provider: Text sharing provider to use
            num_messages: Number of messages from history to share (ignored if custom_content)
            include_system: Include system messages in history
            max_tokens: Token limit for conversation history
            format_template: Custom format template for history
            title: Title/filename for the shared content
            syntax: Syntax highlighting (e.g., "python", "markdown")
            visibility: Visibility level
        """
        from anyenv.text_sharing import get_sharer

        content = await ctx.context.agent.conversation.format_history(
            num_messages=num_messages,
            include_system=include_system,
            max_tokens=max_tokens,
            format_template=format_template,
        )

        if not content.strip():
            await ctx.print("ℹ️ **No content to share**")  # noqa: RUF001
            return

        try:
            # Get the appropriate sharer
            sharer = get_sharer(provider)
            # Share the content
            result = await sharer.share(content, title=title, syntax=syntax, visibility=visibility)
            # Format success message
            provider_name = sharer.name
            msg_parts = [f"🔗 **Content shared via {provider_name}:**", f"• URL: {result.url}"]
            if result.raw_url:
                msg_parts.append(f"• Raw: {result.raw_url}")
            if result.delete_url:
                msg_parts.append(f"• Delete: {result.delete_url}")

            await ctx.print("\n".join(msg_parts))

        except Exception as e:
            msg = f"Failed to share content via {provider}: {e}"
            raise CommandError(msg) from e

    @classmethod
    def condition(cls) -> bool:
        """Check if anyenv text_sharing is available."""
        return importlib.util.find_spec("anyenv.text_sharing") is not None
