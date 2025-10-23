"""Command bridge for converting slashed commands to ACP format."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from acp.schema import AvailableCommand, AvailableCommandInput, CommandInputHint
from llmling_agent.log import get_logger
from llmling_agent_acp.mcp_commands import MCPPromptCommand


if TYPE_CHECKING:
    from slashed import CommandStore


if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.types import Prompt as MCPPrompt
    from slashed import BaseCommand, CommandStore

    from llmling_agent.agent.context import AgentContext
    from llmling_agent_acp.session import ACPSession

logger = get_logger(__name__)
SLASH_PATTERN = re.compile(r"^/([\w-]+)(?:\s+(.*))?$")


class ACPOutputWriter:
    """OutputWriter that immediately sends updates to ACP session."""

    def __init__(self, session: ACPSession) -> None:
        """Initialize with ACP session for immediate updates."""
        self.session = session

    async def print(self, message: str = "", **kwargs: Any) -> None:
        """Send message immediately as session updates."""
        await self.session.notifications.send_agent_text(message)


class ACPCommandBridge:
    """Converts slashed commands to ACP AvailableCommand format."""

    def __init__(self, command_store: CommandStore) -> None:
        """Initialize with existing command store.

        Args:
            command_store: The slashed CommandStore containing available commands
        """
        self.command_store = command_store
        self._update_callbacks: list[Callable[[], None]] = []
        self._mcp_prompt_commands: dict[str, MCPPromptCommand] = {}

    def to_available_commands(self, context: AgentContext[Any]) -> list[AvailableCommand]:
        """Convert slashed commands to ACP format.

        Args:
            context: Optional agent context to filter commands

        Returns:
            List of ACP AvailableCommand objects
        """
        commands = [  # Add regular slashed commands
            _convert_command(cmd)
            for cmd in self.command_store.list_commands()
            if _convert_command(cmd) is not None
        ]

        commands.extend([  # Add MCP prompt commands
            mcp_cmd.to_available_command()
            for mcp_cmd in self._mcp_prompt_commands.values()
        ])

        return commands

    async def execute_slash_command(self, command_text: str, session: ACPSession) -> None:
        """Execute slash command and send results immediately as ACP notifications.

        Args:
            command_text: Full command text (including slash)
            session: ACP session context
        """
        if match := SLASH_PATTERN.match(command_text.strip()):
            command_name = match.group(1)
            args = match.group(2) or ""
            command_name, args = command_name, args.strip()
        else:
            logger.warning("Invalid slash command: %s", command_text)
            return

        # Check if it's an MCP prompt command first
        if command_name in self._mcp_prompt_commands:
            mcp_cmd = self._mcp_prompt_commands[command_name]
            await mcp_cmd.execute(args, session)
        # Create output writer that sends directly to session
        output_writer = ACPOutputWriter(session)

        # Check if it's an ACP-specific command
        acp_commands = {"list-sessions", "load-session", "save-session", "delete-session"}

        if command_name in acp_commands:
            # Use ACP context for ACP commands
            from llmling_agent_acp.acp_commands import ACPCommandContext

            acp_context = ACPCommandContext(session)
            cmd_context: Any = self.command_store.create_context(
                data=acp_context,
                output_writer=output_writer,
            )
        else:
            # Use regular agent context for other commands
            cmd_context = self.command_store.create_context(
                data=session.agent.context,
                output_writer=output_writer,
            )

        command_str = f"{command_name} {args}".strip()
        try:
            await self.command_store.execute_command(command_str, cmd_context)
        except Exception as e:
            logger.exception("Command execution failed")
            await session.notifications.send_agent_text(f"Command error: {e}")

    def register_update_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for command updates.

        Args:
            callback: Function to call when commands are updated
        """
        self._update_callbacks.append(callback)

    def add_mcp_prompt_commands(self, mcp_prompts: list[MCPPrompt]) -> None:
        """Add MCP prompts as slash commands.

        Args:
            mcp_prompts: List of MCP prompt objects from MCP servers
        """
        self._mcp_prompt_commands = {p.name: MCPPromptCommand(p) for p in mcp_prompts}
        self._notify_command_update()  # Notify about command updates

    def _notify_command_update(self) -> None:
        """Notify all registered callbacks about command updates."""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Command update callback failed")


def is_slash_command(text: str) -> bool:
    """Check if text starts with a slash command.

    Args:
        text: Text to check

    Returns:
        True if text is a slash command
    """
    return bool(SLASH_PATTERN.match(text.strip()))


def _convert_command(command: BaseCommand) -> AvailableCommand:
    """Convert a single slashed command to ACP format.

    Args:
        command: Slashed command to convert

    Returns:
        ACP AvailableCommand
    """
    description = command.description
    spec = (
        AvailableCommandInput(root=CommandInputHint(hint=command.usage))
        if command.usage
        else None
    )
    return AvailableCommand(name=command.name, description=description, input=spec)
