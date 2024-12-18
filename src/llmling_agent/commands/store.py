"""Command store implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.commands.base import CommandContext, parse_command
from llmling_agent.commands.exceptions import CommandError
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.commands.base import BaseCommand


logger = get_logger(__name__)


class CommandStore:
    """Central store for command management."""

    def __init__(self) -> None:
        """Initialize an empty command store."""
        self._commands: dict[str, BaseCommand] = {}

    def register_command(self, command: BaseCommand) -> None:
        """Register a new command.

        Args:
            command: Command to register

        Raises:
            ValueError: If command with same name exists
        """
        if command.name in self._commands:
            msg = f"Command '{command.name}' already registered"
            raise ValueError(msg)

        self._commands[command.name] = command
        logger.debug("Registered command: %s", command.name)

    def unregister_command(self, name: str) -> None:
        """Remove a command.

        Args:
            name: Name of command to remove
        """
        if name in self._commands:
            del self._commands[name]
            logger.debug("Unregistered command: %s", name)

    def get_command(self, name: str) -> BaseCommand | None:
        """Get command by name.

        Args:
            name: Name of command to get

        Returns:
            Command if found, None otherwise
        """
        return self._commands.get(name)

    def list_commands(
        self,
        category: str | None = None,
    ) -> list[BaseCommand]:
        """List all commands, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of commands
        """
        if category:
            return [cmd for cmd in self._commands.values() if cmd.category == category]
        return list(self._commands.values())

    def get_categories(self) -> list[str]:
        """Get list of available command categories.

        Returns:
            Sorted list of unique categories
        """
        return sorted({cmd.category for cmd in self._commands.values()})

    def get_commands_by_category(self) -> dict[str, list[BaseCommand]]:
        """Get commands grouped by category.

        Returns:
            Dict mapping categories to lists of commands
        """
        result: dict[str, list[BaseCommand]] = {}
        for cmd in self._commands.values():
            result.setdefault(cmd.category, []).append(cmd)
        return result

    async def execute_command(
        self,
        command_str: str,
        ctx: CommandContext,
    ) -> None:
        """Execute a command from string input.

        Args:
            command_str: Full command string (without leading slash)
            ctx: Command execution context

        Raises:
            CommandError: If command parsing or execution fails
        """
        try:
            # Parse the command string
            parsed = parse_command(command_str)

            # Get the command
            command = self.get_command(parsed.name)
            if not command:
                msg = f"Unknown command: {parsed.name}"
                raise CommandError(msg)  # noqa: TRY301

            # Execute it
            logger.debug(
                "Executing command: %s (args=%s, kwargs=%s)",
                parsed.name,
                parsed.args.args,
                parsed.args.kwargs,
            )
            await command.execute(ctx, parsed.args.args, parsed.args.kwargs)

        except CommandError:
            raise
        except Exception as e:
            msg = f"Command execution failed: {e}"
            raise CommandError(msg) from e

    def register_builtin_commands(self) -> None:
        """Register default system commands."""
        from llmling_agent.commands.builtin import get_builtin_commands

        for command in get_builtin_commands():
            self.register_command(command)
