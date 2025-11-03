"""MCP prompt commands for ACP slash command integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from acp.schema import AvailableCommand, AvailableCommandInput, CommandInputHint
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.mcp_server.manager import Prompt
    from llmling_agent_acp.session import ACPSession


logger = get_logger(__name__)


class PromptCommand:
    """Wrapper for Prompt instances as slash commands."""

    def __init__(self, prompt: Prompt) -> None:
        """Initialize with Prompt instance.

        Args:
            prompt: Prompt instance from ToolManager
        """
        self.prompt = prompt
        self.name = prompt.name
        self.description = prompt.description or f"MCP prompt: {prompt.name}"

    def to_available_command(self) -> AvailableCommand:
        """Convert to ACP AvailableCommand format.

        Returns:
            ACP AvailableCommand object
        """
        # Create input spec from prompt arguments
        spec = None
        if self.prompt.arguments:
            arg_names = [arg["name"] for arg in self.prompt.arguments]
            hint = f"Arguments: {', '.join(arg_names)}"
            spec = AvailableCommandInput(root=CommandInputHint(hint=hint))
        name = self.name  # Use prompt name directly
        return AvailableCommand(name=name, description=self.description, input=spec)

    async def execute(self, args: str, session: ACPSession) -> None:
        """Execute prompt command.

        Args:
            args: Command arguments string
            session: ACP session context
        """
        arguments = self._parse_arguments(args) if args.strip() else None

        try:
            # Get components from the prompt
            components = await self.prompt.get_components(arguments)

            # Stage the components for later use
            session.add_staged_parts(components)

            # Convert components to text output for display
            content_parts = []
            for part in components:
                if isinstance(part.content, str):
                    content_parts.append(part.content)
                else:
                    # Handle sequence of UserContent - convert to string
                    content_parts.append(str(part.content))
            output = "\n".join(content_parts)

            # Add argument info if provided
            if arguments:
                arg_info = ", ".join(f"{k}={v}" for k, v in arguments.items())
                display_output = (
                    f"Prompt {self.prompt.name!r} with args ({arg_info}):\n\n{output}"
                )
            else:
                display_output = f"Prompt {self.prompt.name!r}:\n\n{output}"

            # Send confirmation + preview to user
            staged_count = session.get_staged_parts_count()
            confirmation = (
                f"✅ Prompt '{self.prompt.name}' added to context "
                f"({staged_count} total parts staged)\n\n{display_output}"
            )
            await session.notifications.send_agent_text(confirmation)

        except Exception as e:
            error_msg = f"❌ Prompt execution failed: {e}"
            logger.exception("Prompt execution error")
            await session.notifications.send_agent_text(error_msg)

    def _parse_arguments(self, args_str: str) -> dict[str, str]:
        """Parse argument string to dictionary.

        Args:
            args_str: Raw argument string

        Returns:
            Dictionary of argument name to value
        """
        # Simple parsing - split on spaces and match to prompt arguments
        if not self.prompt.arguments:
            return {}

        args_list = args_str.strip().split()
        arguments = {}

        # Map positional arguments to prompt argument names
        for i, arg_value in enumerate(args_list):
            if i < len(self.prompt.arguments):
                arg_name = self.prompt.arguments[i]["name"]
                arguments[arg_name] = arg_value

        return arguments
