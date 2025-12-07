"""Terminal management slash commands for ACP sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from slashed import CommandContext  # noqa: TC002

from acp.schema import TerminalToolCallContent
from llmling_agent.log import get_logger
from llmling_agent.messaging.context import NodeContext  # noqa: TC001
from llmling_agent_commands.base import NodeCommand
from llmling_agent_server.acp_server.session import ACPSession  # noqa: TC001


if TYPE_CHECKING:
    from acp.schema import ToolCallStatus


logger = get_logger(__name__)


class TerminalOutputCommand(NodeCommand):
    """Get current output from a terminal.

    Retrieves and displays the current output from the specified terminal,
    showing it as an embedded terminal in a tool call for rich display.
    """

    name = "terminal-output"
    category = "terminal"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        terminal_id: str,
    ) -> None:
        """Get current output from a terminal.

        Args:
            ctx: Command context with ACP session
            terminal_id: Terminal identifier to get output from
        """
        session = ctx.context.data
        assert session

        try:
            # Check terminal capabilities
            if not (
                session.client_capabilities
                and session.client_capabilities.terminal
                and session.acp_agent.terminal_access
            ):
                await session.notifications.send_agent_text("❌ **Terminal access not available**")
                return

            # Generate a tool call ID for this operation
            tool_call_id = f"terminal-output-{uuid.uuid4().hex[:8]}"

            # Get terminal output first to check status
            output_response = await session.requests.terminal_output(terminal_id)

            # Determine status and title
            if output_response.exit_status:
                exit_code = output_response.exit_status.exit_code or 0
                status: ToolCallStatus = "completed" if exit_code == 0 else "failed"
                title = f"Terminal {terminal_id} - Exit Code: {exit_code}"
            else:
                status = "in_progress"
                title = f"Terminal {terminal_id} - Running"

            # Send tool call with embedded terminal content
            await session.notifications.tool_call_start(
                tool_call_id=tool_call_id,
                title=title,
                kind="read",
                content=[TerminalToolCallContent(terminal_id=terminal_id)],
            )

            # Send progress update with status
            await session.notifications.terminal_progress(
                tool_call_id=tool_call_id,
                terminal_id=terminal_id,
                status=status,
                title=title,
            )

        except Exception as e:
            logger.exception("Failed to get terminal output", terminal_id=terminal_id)
            # Send error notification asynchronously to avoid blocking
            session.acp_agent.tasks.create_task(
                session._send_error_notification(f"❌ **Failed to get terminal output:** {e}"),
                name=f"terminal_output_error_{session.session_id}",
            )


class TerminalKillCommand(NodeCommand):
    """Kill a running terminal.

    Terminates the specified terminal without releasing its resources.
    The terminal remains valid for output retrieval after killing.
    """

    name = "terminal-kill"
    category = "terminal"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        terminal_id: str,
    ) -> None:
        """Kill a running terminal.

        Args:
            ctx: Command context with ACP session
            terminal_id: Terminal identifier to kill
        """
        session = ctx.context.data
        assert session

        try:
            # Check terminal capabilities
            if not (
                session.client_capabilities
                and session.client_capabilities.terminal
                and session.acp_agent.terminal_access
            ):
                await session.notifications.send_agent_text("❌ **Terminal access not available**")
                return

            # Generate a tool call ID for this operation
            tool_call_id = f"terminal-kill-{uuid.uuid4().hex[:8]}"

            # Start tool call
            await session.notifications.tool_call_start(
                tool_call_id=tool_call_id,
                title=f"Killing terminal {terminal_id}",
                kind="execute",
                content=[TerminalToolCallContent(terminal_id=terminal_id)],
            )

            # Kill the terminal
            await session.requests.kill_terminal(terminal_id)

            # Send completion with final terminal state
            await session.notifications.terminal_progress(
                tool_call_id=tool_call_id,
                terminal_id=terminal_id,
                status="completed",
                title=f"Terminal {terminal_id} killed",
            )

        except Exception as e:
            logger.exception("Failed to kill terminal", terminal_id=terminal_id)
            # Send failure notification if we have a tool call ID
            try:
                tool_call_id = f"terminal-kill-{uuid.uuid4().hex[:8]}"
                await session.notifications.tool_call_start(
                    tool_call_id=tool_call_id,
                    title=f"Failed to kill terminal {terminal_id}",
                    kind="execute",
                )
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="failed",
                    title=f"Error: {e}",
                )
            except Exception:  # noqa: BLE001
                # Send error notification asynchronously to avoid blocking
                session.acp_agent.tasks.create_task(
                    session._send_error_notification(f"❌ **Failed to kill terminal:** {e}"),
                    name=f"terminal_kill_error_{session.session_id}",
                )


class TerminalCreateCommand(NodeCommand):
    """Create a new terminal and run a command.

    Creates a new terminal session and starts the specified command,
    showing live output as it runs through an embedded terminal display.
    """

    name = "terminal-create"
    category = "terminal"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        command: str,
        *args: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Create a new terminal and run a command.

        Args:
            ctx: Command context with ACP session
            command: Command to execute
            args: Command arguments
            cwd: Working directory (defaults to session cwd)
            env: Environment variables
        """
        session = ctx.context.data
        assert session

        try:
            # Check terminal capabilities
            if not (
                session.client_capabilities
                and session.client_capabilities.terminal
                and session.acp_agent.terminal_access
            ):
                await session.notifications.send_agent_text("❌ **Terminal access not available**")
                return

            # Generate a tool call ID for this operation
            tool_call_id = f"terminal-create-{uuid.uuid4().hex[:8]}"

            # Create terminal
            create_response = await session.requests.create_terminal(
                command=command,
                args=list(args),
                cwd=cwd or session.cwd,
                env=env or {},
            )
            terminal_id = create_response.terminal_id

            # Display command being run with embedded terminal
            cmd_display = f"{command} {' '.join(args)}"

            await session.notifications.tool_call_start(
                tool_call_id=tool_call_id,
                title=f"Running: {cmd_display}",
                kind="execute",
                content=[TerminalToolCallContent(terminal_id=terminal_id)],
            )

            # Send initial progress update
            await session.notifications.terminal_progress(
                tool_call_id=tool_call_id,
                terminal_id=terminal_id,
                status="in_progress",
                title=f"Terminal {terminal_id} created",
            )

        except Exception as e:
            logger.exception("Failed to create terminal")
            # Send failure notification
            try:
                tool_call_id = f"terminal-create-{uuid.uuid4().hex[:8]}"
                await session.notifications.tool_call_start(
                    tool_call_id=tool_call_id,
                    title="Failed to create terminal",
                    kind="execute",
                )
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="failed",
                    title=f"Error: {e}",
                )
            except Exception:  # noqa: BLE001
                # Send error notification asynchronously to avoid blocking
                session.acp_agent.tasks.create_task(
                    session._send_error_notification(f"❌ **Failed to create terminal:** {e}"),
                    name=f"terminal_create_error_{session.session_id}",
                )


def get_terminal_commands() -> list[type[NodeCommand]]:
    """Get all terminal management slash commands."""
    return [
        TerminalOutputCommand,
        TerminalKillCommand,
        TerminalCreateCommand,
    ]
