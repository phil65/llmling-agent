"""MCP prompt commands for ACP slash command integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import TextContent

from acp.schema import AvailableCommand, AvailableCommandInput, CommandInputHint
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from mcp.types import Prompt as MCPPrompt

    from llmling_agent_acp.session import ACPSession


logger = get_logger(__name__)


class MCPPromptCommand:
    """Wrapper for MCP prompts as slash commands."""

    def __init__(self, mcp_prompt: MCPPrompt) -> None:
        """Initialize with MCP prompt.

        Args:
            mcp_prompt: MCP prompt object from server
        """
        self.mcp_prompt = mcp_prompt
        self.name = mcp_prompt.name
        self.description = mcp_prompt.description or f"MCP prompt: {mcp_prompt.name}"

    def to_available_command(self) -> AvailableCommand:
        """Convert to ACP AvailableCommand format.

        Returns:
            ACP AvailableCommand object
        """
        # Create input spec from MCP prompt arguments
        spec = None
        if self.mcp_prompt.arguments:
            arg_names = [arg.name for arg in self.mcp_prompt.arguments]
            hint = f"Arguments: {', '.join(arg_names)}"
            spec = AvailableCommandInput(root=CommandInputHint(hint=hint))
        name = f"mcp-{self.name}"  # Prefix to avoid conflicts
        return AvailableCommand(name=name, description=self.description, input=spec)

    async def execute(self, args: str, session: ACPSession) -> None:
        """Execute MCP prompt command.

        Args:
            args: Command arguments string
            session: ACP session context

        Yields:
            SessionNotification objects with prompt results
        """
        arguments = self._parse_arguments(args) if args.strip() else None
        assert session.agent.mcp, "No MCP manager available"
        # Find appropriate MCP client (use first available for now)
        if not session.agent.mcp.clients:
            error_msg = "No MCP clients connected"
            await session.notifications.send_agent_text(error_msg)
            return
        # Execute prompt via first available MCP client
        client = next(iter(session.agent.mcp.clients.values()))

        try:
            # Try with arguments first, fallback to no arguments
            try:
                result = await client.get_prompt(self.mcp_prompt.name, arguments)
            except Exception as e:
                if arguments:
                    msg = "MCP prompt with arguments failed, trying without"
                    logger.warning(msg, error=e)
                    result = await client.get_prompt(self.mcp_prompt.name)
                else:
                    raise

            content_parts = [  # Convert prompt result to text
                message.content.text
                for message in result.messages
                if isinstance(message.content, TextContent)
            ]
            output = "\n".join(content_parts)
            # Add argument info if provided
            if arguments:
                arg_info = ", ".join(f"{k}={v}" for k, v in arguments.items())
                output = (
                    f"Prompt {self.mcp_prompt.name!r} with args ({arg_info}):\n\n{output}"
                )

            await session.notifications.send_agent_text(output)

        except Exception as e:
            error_msg = f"MCP prompt execution failed: {e}"
            logger.exception("MCP prompt execution error")
            await session.notifications.send_agent_text(error_msg)

    def _parse_arguments(self, args_str: str) -> dict[str, str]:
        """Parse argument string to dictionary.

        Args:
            args_str: Raw argument string

        Returns:
            Dictionary of argument name to value
        """
        # Simple parsing - split on spaces and match to prompt arguments
        if not self.mcp_prompt.arguments:
            return {}

        args_list = args_str.strip().split()
        arguments = {}

        # Map positional arguments to prompt argument names
        for i, arg_value in enumerate(args_list):
            if i < len(self.mcp_prompt.arguments):
                arg_name = self.mcp_prompt.arguments[i].name
                arguments[arg_name] = arg_value

        return arguments
