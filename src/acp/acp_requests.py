"""ACP request utilities for simplified client interactions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from acp.schema import (
    CreateTerminalRequest,
    EnvVariable,
    KillTerminalCommandRequest,
    PermissionOption,
    ReadTextFileRequest,
    ReleaseTerminalRequest,
    RequestPermissionRequest,
    TerminalOutputRequest,
    ToolCall,
    WaitForTerminalExitRequest,
    WriteTextFileRequest,
)
from acp.terminal_handle import TerminalHandle
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from acp import (
        Client,
        RequestPermissionResponse,
        TerminalOutputResponse,
        WaitForTerminalExitResponse,
    )

logger = get_logger(__name__)


RULES_FILE_NAMES = [
    ".rules",
    "CLAUDE.md",
    "AGENT.md",
    "AGENTS.md",
    "GEMINI.md",
    ".cursorrules",
    ".windsurfrules",
    ".clinerules",
    ".github/copilot-instructions.md",
]


class ACPRequests:
    """Clean API for creating and sending ACP session requests.

    Provides convenient methods for common request patterns,
    handling both creation and sending in a single call.
    """

    def __init__(self, client: Client, session_id: str) -> None:
        """Initialize requests helper.

        Args:
            client: ACP client
            session_id: Session ID
        """
        self.client = client
        self.id = session_id

    async def read_text_file(
        self,
        path: str,
        *,
        limit: int | None = None,
        line: int | None = None,
    ) -> str:
        """Read text content from a file.

        Args:
            path: File path to read
            limit: Maximum number of lines to read
            line: Line number to start reading from (1-based)

        Returns:
            File content as string
        """
        request = ReadTextFileRequest(
            session_id=self.id,
            path=path,
            limit=limit,
            line=line,
        )
        response = await self.client.read_text_file(request)
        return response.content

    async def read_agent_rules(self, cwd: str) -> str | None:
        """Check common agent rule files and return content of first match.

        Args:
            cwd: Current working directory

        Returns:
            Agent rules as string or None if not found
        """
        for filename in RULES_FILE_NAMES:
            if info := await self.read_text_file(f"{cwd}/{filename}"):
                return info
        return None

    async def write_text_file(self, path: str, content: str) -> None:
        """Write text content to a file.

        Args:
            path: File path to write
            content: Text content to write
        """
        request = WriteTextFileRequest(session_id=self.id, path=path, content=content)
        await self.client.write_text_file(request)

    async def create_terminal(
        self,
        command: str,
        *,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        output_byte_limit: int | None = None,
    ) -> TerminalHandle:
        """Create a new terminal session.

        Args:
            command: Command to run in terminal
            args: Command arguments
            cwd: Working directory for terminal
            env: Environment variables for terminal
            output_byte_limit: Maximum bytes to capture from output

        Returns:
            Terminal handle for managing the terminal session
        """
        request = CreateTerminalRequest(
            session_id=self.id,
            command=command,
            args=args,
            cwd=cwd,
            env=[EnvVariable(name=k, value=v) for k, v in (env or {}).items()],
            output_byte_limit=output_byte_limit,
        )
        response = await self.client.create_terminal(request)
        return TerminalHandle(terminal_id=response.terminal_id, requests=self)

    async def terminal_output(self, terminal_id: str) -> TerminalOutputResponse:
        """Get output from a terminal session.

        Args:
            terminal_id: Terminal identifier

        Returns:
            Terminal output response
        """
        request = TerminalOutputRequest(session_id=self.id, terminal_id=terminal_id)
        return await self.client.terminal_output(request)

    async def wait_for_terminal_exit(
        self,
        terminal_id: str,
    ) -> WaitForTerminalExitResponse:
        """Wait for a terminal to exit.

        Args:
            terminal_id: Terminal identifier

        Returns:
            Terminal exit response with exit_code
        """
        request = WaitForTerminalExitRequest(session_id=self.id, terminal_id=terminal_id)
        return await self.client.wait_for_terminal_exit(request)

    async def kill_terminal(self, terminal_id: str) -> None:
        """Kill a terminal session.

        Args:
            terminal_id: Terminal identifier to kill
        """
        request = KillTerminalCommandRequest(session_id=self.id, terminal_id=terminal_id)
        await self.client.kill_terminal(request)

    async def release_terminal(self, terminal_id: str) -> None:
        """Release a terminal session.

        Args:
            terminal_id: Terminal identifier to release
        """
        request = ReleaseTerminalRequest(session_id=self.id, terminal_id=terminal_id)
        await self.client.release_terminal(request)

    async def run_command(
        self,
        command: str,
        *,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        output_byte_limit: int | None = None,
        timeout_seconds: int | None = None,
    ) -> tuple[str, int | None]:
        """Execute a shell command and return output and exit code.

        This is a high-level convenience method that creates a terminal,
        runs the command, waits for completion, and cleans up.

        Args:
            command: Command to execute
            args: Command arguments
            cwd: Working directory for command execution
            env: Environment variables for command execution
            output_byte_limit: Maximum bytes to capture from output
            timeout_seconds: Command timeout in seconds

        Returns:
            Tuple of (output, exit_code)
        """
        terminal_handle = await self.create_terminal(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            output_byte_limit=output_byte_limit,
        )
        terminal_id = terminal_handle.terminal_id

        try:
            if timeout_seconds:  # Wait for completion (with optional timeout)
                try:
                    exit_result = await asyncio.wait_for(
                        self.wait_for_terminal_exit(terminal_id),
                        timeout=timeout_seconds,
                    )
                except TimeoutError:  # Kill on timeout and get partial output
                    await self.kill_terminal(terminal_id)
                    output_response = await self.terminal_output(terminal_id)
                    return output_response.output, None
            else:
                exit_result = await self.wait_for_terminal_exit(terminal_id)

            output_response = await self.terminal_output(terminal_id)
            return output_response.output, exit_result.exit_code

        finally:  # Always release terminal
            await self.release_terminal(terminal_id)

    async def request_permission(
        self,
        tool_call_id: str,
        *,
        title: str | None = None,
        options: list[PermissionOption] | None = None,
    ) -> RequestPermissionResponse:
        """Request permission from user before executing a tool call.

        Args:
            tool_call_id: Unique identifier for the tool call
            title: Human-readable description of the operation
            options: Available permission options (defaults to allow/reject once)

        Returns:
            Permission response with user's decision
        """
        if options is None:
            options = [
                PermissionOption(
                    option_id="allow-once",
                    name="Allow once",
                    kind="allow_once",
                ),
                PermissionOption(
                    option_id="reject-once",
                    name="Reject",
                    kind="reject_once",
                ),
            ]

        tool_call = ToolCall(tool_call_id=tool_call_id, title=title)
        request = RequestPermissionRequest(
            session_id=self.id,
            tool_call=tool_call,
            options=options,
        )
        return await self.client.request_permission(request)
