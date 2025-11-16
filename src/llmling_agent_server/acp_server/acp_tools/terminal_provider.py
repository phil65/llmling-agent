"""ACP terminal provider for shell command execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext  # noqa: TC002

from acp.schema import TerminalToolCallContent
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling_agent_server.acp_server.session import ACPSession


class ACPTerminalProvider(ResourceProvider):
    """Provides ACP terminal-related tools for command execution.

    This provider creates session-aware tools for executing shell commands
    via the ACP client. All tools have the session ID baked in at creation time,
    eliminating the need for parameter injection.
    """

    def __init__(self, session: ACPSession) -> None:
        """Initialize terminal provider.

        Args:
            session: Session for all tools created by this provider
        """
        super().__init__(name=f"acp_terminal_{session.session_id}")
        self.agent = session.acp_agent
        self.session_id = session.session_id
        self.session = session
        self.client_capabilities = session.client_capabilities
        self.cwd = session.cwd

    async def get_tools(self) -> list[Tool]:
        """Get all terminal tools with session_id baked in."""
        if (
            self.client_capabilities
            and self.client_capabilities.terminal
            and self.agent.terminal_access
        ):
            return [
                Tool.from_callable(
                    self.run_command, source="terminal", category="execute"
                ),
                Tool.from_callable(
                    self.get_command_output, source="terminal", category="read"
                ),
                Tool.from_callable(
                    self.create_terminal, source="terminal", category="execute"
                ),
                Tool.from_callable(
                    self.wait_for_terminal_exit, source="terminal", category="execute"
                ),
                Tool.from_callable(
                    self.kill_terminal, source="terminal", category="execute"
                ),
                Tool.from_callable(
                    self.release_terminal, source="terminal", category="execute"
                ),
            ]

        return []

    async def run_command(  # noqa: D417
        self,
        ctx: RunContext[Any],
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        output_char_limit: int | None = None,
        timeout_seconds: int | None = None,
    ) -> str:
        r"""Execute a shell command with live terminal output visible to the user.

        This tool shows real-time command output in the UI as it happens,
        perfect for long-running commands like builds, tests, or installations.

        Args:
            command: The command to execute (e.g., 'npm', 'python', 'make')
            args: Command arguments as list (e.g., ['test', '--coverage'])
            cwd: Working directory path
            env: Environment variables as key-value pairs
            output_char_limit: Maximum output characters to retain
            timeout_seconds: Maximum time to wait for command completion

        Returns:
            Command completion status and any final output
        """
        # Send initial tool call notification
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        try:
            create_response = await self.session.requests.create_terminal(
                command=command,
                args=args or [],
                cwd=cwd,
                env=env or {},
                output_byte_limit=output_char_limit,
            )
            terminal_id = create_response.terminal_id
            # Send initial notification with embedded terminal for live output
            cmd_display = f"{command} {' '.join(args or [])}"
            await self.session.notifications.tool_call_start(
                tool_call_id=ctx.tool_call_id,
                title=f"Running: {cmd_display}",
                kind="execute",
                content=[TerminalToolCallContent(terminal_id=terminal_id)],
            )

            # Wait for completion (with optional timeout)
            if timeout_seconds:
                try:
                    exit_result = await asyncio.wait_for(
                        self.session.requests.wait_for_terminal_exit(terminal_id),
                        timeout=timeout_seconds,
                    )
                except TimeoutError:
                    # Kill the command on timeout
                    await self.session.requests.kill_terminal(terminal_id)

                    # Get output before releasing
                    output_response = await self.session.requests.terminal_output(
                        terminal_id
                    )

                    # Send timeout notification
                    await self.session.notifications.terminal_progress(
                        tool_call_id=ctx.tool_call_id,
                        terminal_id=terminal_id,
                        status="failed",
                        title=f"Command timed out after {timeout_seconds}s",
                    )

                    # Release terminal
                    await self.session.requests.release_terminal(terminal_id)

                    return (
                        f"Command timed out after {timeout_seconds} seconds. "
                        f"Output: \n{output_response.output}"
                    )
            else:
                exit_result = await self.session.requests.wait_for_terminal_exit(
                    terminal_id
                )

            output_response = await self.session.requests.terminal_output(terminal_id)
            # Send completion notification (terminal remains embedded for viewing)
            exit_code = exit_result.exit_code or 0
            await self.session.notifications.terminal_progress(
                tool_call_id=ctx.tool_call_id,
                terminal_id=terminal_id,
                status="completed" if (exit_result.exit_code or 0) == 0 else "failed",
                title=f"Command completed (exit code: {exit_code})",
            )

            # Release terminal (output remains visible in UI)
            await self.session.requests.release_terminal(terminal_id)
            result = f"Command completed with exit code {exit_code}:\n"
            result += f"Output:\n{output_response.output}"
            if output_response.truncated:
                result += " (output was truncated)"
        except Exception as e:  # noqa: BLE001
            try:  # Send error notification
                await self.session.notifications.tool_call_progress(
                    tool_call_id=ctx.tool_call_id,
                    status="failed",
                    title=f"Command failed: {e}",
                )
            except Exception:  # noqa: BLE001
                self.log.warning("Failed to send error update")

            return f"Error executing command: {e}"
        else:
            return result

    async def get_command_output(self, ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
        """Get output from a terminal that was created with create_terminal.

        Use this to check output from a terminal created separately.
        Do NOT use this for simple command execution - use run_command instead.

        Args:
            terminal_id: The terminal ID returned by create_terminal

        Returns:
            Current output and status (running/completed/failed)
        """
        # Send initial pending notification
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        try:
            await self.session.notifications.tool_call_start(
                tool_call_id=ctx.tool_call_id,
                title=f"Getting output from terminal {terminal_id}",
                kind="read",
            )
        except Exception as e:  # noqa: BLE001
            self.log.warning("Failed to send pending update", error=e)

        try:
            output_response = await self.session.requests.terminal_output(terminal_id)
            # Determine status for both UI and agent
            exit_code = None
            signal = None
            status_text = "Running"
            if output_response.exit_status:
                exit_code = output_response.exit_status.exit_code
                signal = output_response.exit_status.signal
                if exit_code is not None:
                    status_text = f"Exited with code {exit_code}"
                elif signal:
                    status_text = f"Terminated by signal {signal}"

            # Send completion update with output as content
            await self.session.notifications.tool_call_progress(
                tool_call_id=ctx.tool_call_id,
                status="completed",
                title=f"Terminal {terminal_id} output retrieved",
                content=[output_response.output],
                raw_output=f"Retrieved output from terminal {terminal_id}",
            )

            # Return structured result for agent
            result = f"""Terminal {terminal_id} Output:
{output_response.output}

Status: {status_text}"""

            if output_response.truncated:
                result += "\n(output truncated)"
        except Exception as e:  # noqa: BLE001
            # Send failed update
            try:
                await self.session.notifications.tool_call_progress(
                    tool_call_id=ctx.tool_call_id,
                    status="failed",
                    raw_output=f"Error: {e}",
                )
            except Exception:  # noqa: BLE001
                self.log.warning("Failed to send failed update")
            return f"Error getting command output: {e}"
        else:
            return result

    async def create_terminal(  # noqa: D417
        self,
        ctx: RunContext[Any],
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        output_byte_limit: int = 1048576,
    ) -> str:
        """Create a terminal for advanced terminal management (ADVANCED USE ONLY).

        WARNING: This only creates a terminal and returns its ID. It does NOT wait for
        completion or get output. You must use get_command_output,
        wait_for_terminal_exit, and release_terminal separately.

        For simple command execution, use run_command instead - it's much easier.

        Args:
            command: The command to execute
            args: Command arguments
            cwd: Working directory
            env: Environment variables
            output_byte_limit: Maximum output bytes to retain

        Returns:
            Terminal creation details with ID for management
        """
        # Send initial pending notification
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        cmd_display = f"{command} {' '.join(args or [])}"
        try:
            await self.session.notifications.tool_call_start(
                tool_call_id=ctx.tool_call_id,
                title=f"Creating terminal: {cmd_display}",
                kind="execute",
            )
        except Exception as e:  # noqa: BLE001
            self.log.warning("Failed to send pending update", error=e)

        try:
            create_response = await self.session.requests.create_terminal(
                command=command,
                args=args or [],
                cwd=cwd,
                env=env or {},
                output_byte_limit=output_byte_limit,
            )
            terminal_id = create_response.terminal_id

            # Send completion update
            await self.session.notifications.tool_call_progress(
                tool_call_id=ctx.tool_call_id,
                status="completed",
                title=f"Terminal {terminal_id} created",
                raw_output=f"Created terminal: {terminal_id}",
            )
        except Exception as e:  # noqa: BLE001
            try:  # Send failed update
                await self.session.notifications.tool_call_progress(
                    tool_call_id=ctx.tool_call_id,
                    status="failed",
                    raw_output=f"Error: {e}",
                )
            except Exception:  # noqa: BLE001
                self.log.warning("Failed to send failed update")
            return f"Error creating terminal: {e}"
        else:
            return f"""Created terminal: {terminal_id}
Command: {cmd_display}
Working Directory: {cwd or "default"}
Output Limit: {output_byte_limit} bytes
Status: Running

Use this terminal_id with other terminal tools:
- wait_for_terminal_exit({terminal_id})
- get_command_output({terminal_id})
- kill_terminal({terminal_id})
- release_terminal({terminal_id})"""

    async def wait_for_terminal_exit(self, ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
        """Wait for a terminal to finish (ADVANCED USE ONLY).

        Only use this with terminals created by create_terminal.
        For simple command execution, use run_command instead.

        Args:
            terminal_id: The terminal ID from create_terminal

        Returns:
            Exit status information and final output
        """
        # Send initial notification with embedded terminal for live view
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        try:
            await self.session.notifications.tool_call_start(
                tool_call_id=ctx.tool_call_id,
                title=f"Waiting for terminal {terminal_id} to complete",
                kind="execute",
                content=[TerminalToolCallContent(terminal_id=terminal_id)],
            )
        except Exception as e:  # noqa: BLE001
            self.log.warning("Failed to send pending update", error=e)

        try:
            exit_response = await self.session.requests.wait_for_terminal_exit(
                terminal_id
            )
            output_response = await self.session.requests.terminal_output(terminal_id)
            # Send completion notification (terminal remains embedded for viewing)
            exit_code = exit_response.exit_code or 0
            await self.session.notifications.terminal_progress(
                tool_call_id=ctx.tool_call_id,
                terminal_id=terminal_id,
                status="completed" if exit_code == 0 else "failed",
                title=f"Terminal completed (exit code: {exit_code})",
            )

            # Return structured result for agent
            result = f"""Terminal {terminal_id} completed
Exit Code: {exit_code}
Status: {"SUCCESS" if exit_code == 0 else "FAILED"}"""
            if exit_response.signal:
                result += f"\nTerminated by signal: {exit_response.signal}"
            result += f"\n\nFinal Output:\n{output_response.output}"
            if output_response.truncated:
                result += "\n(output truncated)"
        except Exception as e:  # noqa: BLE001
            try:  # Send failed update
                await self.session.notifications.tool_call_progress(
                    tool_call_id=ctx.tool_call_id,
                    status="failed",
                    raw_output=f"Error: {e}",
                )
            except Exception:  # noqa: BLE001
                self.log.warning("Failed to send failed update")
            return f"Error waiting for terminal exit: {e}"
        else:
            return result

    async def kill_terminal(self, ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
        """Kill a running terminal (ADVANCED USE ONLY).

        Only use this with terminals created by create_terminal.
        For simple command execution, use run_command instead.

        Args:
            terminal_id: The terminal ID to kill

        Returns:
            Termination confirmation message
        """
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        await self.session.notifications.tool_call_start(
            tool_call_id=ctx.tool_call_id,
            title=f"Killing terminal {terminal_id}",
            kind="execute",
        )

        try:
            await self.session.requests.kill_terminal(terminal_id)
            # Send completion update
            await self.session.notifications.tool_call_progress(
                tool_call_id=ctx.tool_call_id,
                status="completed",
                title=f"Terminal {terminal_id} killed",
                raw_output=f"Killed terminal: {terminal_id}",
            )
        except Exception as e:  # noqa: BLE001
            try:  # Send failed update
                await self.session.notifications.tool_call_progress(
                    tool_call_id=ctx.tool_call_id,
                    status="failed",
                    raw_output=f"Error: {e}",
                )
            except Exception:  # noqa: BLE001
                self.log.warning("Failed to send failed update")
            return f"Error killing terminal: {e}"
        else:
            return f"Terminal {terminal_id} killed successfully"

    async def release_terminal(self, ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
        """Release a terminal session (ADVANCED USE ONLY).

        This cleans up a terminal created by create_terminal.
        Usually you don't need to call this directly as other tools
        handle cleanup automatically.

        Args:
            terminal_id: The terminal ID from create_terminal

        Returns:
            Release confirmation
        """
        # Send initial pending notification
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        try:
            await self.session.notifications.tool_call_start(
                tool_call_id=ctx.tool_call_id,
                title=f"Releasing terminal {terminal_id}",
                kind="execute",
            )
        except Exception as e:  # noqa: BLE001
            self.log.warning("Failed to send pending update", error=e)

        try:
            await self.session.requests.release_terminal(terminal_id)
            # Send completion update
            await self.session.notifications.tool_call_progress(
                tool_call_id=ctx.tool_call_id,
                status="completed",
                title=f"Terminal {terminal_id} released",
                raw_output=f"Terminal {terminal_id} released successfully",
            )
        except Exception as e:  # noqa: BLE001
            try:  # Send failed update
                await self.session.notifications.tool_call_progress(
                    tool_call_id=ctx.tool_call_id,
                    status="failed",
                    raw_output=f"Error: {e}",
                )
            except Exception:  # noqa: BLE001
                self.log.warning("Failed to send failed update")
            return f"Error releasing terminal: {e}"
        else:
            return f"Terminal {terminal_id} released successfully"
