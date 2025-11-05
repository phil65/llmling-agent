"""Command bridge for converting slashed commands to ACP format."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from acp.schema import AvailableCommand
from llmling_agent.log import get_logger
from llmling_agent_acp.commands.acp_commands import ACPCommandContext


if TYPE_CHECKING:
    from collections.abc import Callable

    from slashed import CommandContext, CommandStore

    from llmling_agent.agent.context import AgentContext
    from llmling_agent.mcp_server.manager import Prompt
    from llmling_agent_acp.session import ACPSession

logger = get_logger(__name__)
SLASH_PATTERN = re.compile(r"^/([\w-]+)(?:\s+(.*))?$")
ACP_COMMANDS = {"list-sessions", "load-session", "save-session", "delete-session"}


class ACPCommandBridge:
    """Converts slashed commands to ACP AvailableCommand format."""

    def __init__(self, command_store: CommandStore) -> None:
        """Initialize with existing command store.

        Args:
            command_store: The slashed CommandStore containing available commands
        """
        self.command_store = command_store
        self._update_callbacks: list[Callable[[], None]] = []

    def get_acp_commands(self, context: AgentContext[Any]) -> list[AvailableCommand]:
        """Convert all slashed commands to ACP format.

        Args:
            context: Optional agent context to filter commands

        Returns:
            List of ACP AvailableCommand objects
        """
        return [
            AvailableCommand.create(
                name=cmd.name,
                description=cmd.description,
                input_hint=cmd.usage,
            )
            for cmd in self.command_store.list_commands()
        ]

    async def execute_slash_command(self, command_text: str, session: ACPSession) -> None:
        """Execute any slash command with unified handling.

        Args:
            command_text: Full command text (including slash)
            session: ACP session context
        """
        if match := SLASH_PATTERN.match(command_text.strip()):
            command_name = match.group(1)
            args = match.group(2) or ""
        else:
            logger.warning("Invalid slash command", command=command_text)
            return

        # Single execution path for ALL commands
        if command_name in ACP_COMMANDS:
            # Use ACP context for ACP commands
            acp_ctx = ACPCommandContext(session)
            cmd_ctx: CommandContext = self.command_store.create_context(
                data=acp_ctx,
                output_writer=session.notifications.send_agent_text,
            )
        else:
            # Use regular agent context for other commands (including MCP prompts)
            cmd_ctx = self.command_store.create_context(
                data=session.agent.context,
                output_writer=session.notifications.send_agent_text,
            )

        command_str = f"{command_name} {args}".strip()
        try:
            await self.command_store.execute_command(command_str, cmd_ctx)
        except Exception as e:
            logger.exception("Command execution failed")
            await session.notifications.send_agent_text(f"❌ Command error: {e}")

    def register_update_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for command updates.

        Args:
            callback: Function to call when commands are updated
        """
        self._update_callbacks.append(callback)

    def register_mcp_prompts(self, prompts: list[Prompt], session: ACPSession) -> None:
        """Register MCP prompts as regular slashed commands.

        Args:
            prompts: List of Prompt instances from MCP servers
            session: ACP session for command execution context
        """
        from llmling_agent_acp.mcp_command_factory import create_mcp_command

        for prompt in prompts:
            command = create_mcp_command(prompt, session)
            self.command_store.register_command(command)

        self._notify_command_update()

    def _notify_command_update(self) -> None:
        """Notify all registered callbacks about command updates."""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Command update callback failed")
