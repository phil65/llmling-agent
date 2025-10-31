"""Headless ACP client implementation with real filesystem and terminal operations.

This module provides a headless client implementation that performs actual
filesystem operations and uses ProcessManager for real terminal execution,
making it ideal for testing and standalone usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import uuid

from anyenv import ProcessManager

from acp.client import Client
from acp.schema import (
    AllowedOutcome,
    CreateTerminalRequest,
    CreateTerminalResponse,
    DeniedOutcome,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    ReadTextFileRequest,
    ReadTextFileResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    WriteTextFileRequest,
    WriteTextFileResponse,
)
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from acp.schema import (
        AgentRequest,
        ClientResponse,
        SessionNotification,
    )

logger = get_logger(__name__)


class HeadlessACPClient(Client):
    """Headless ACP client with real filesystem and terminal operations.

    This client implementation:
    - Performs real filesystem operations
    - Uses ProcessManager for actual terminal/command execution
    - Automatically grants permissions for testing
    - Suitable for testing and standalone usage
    """

    def __init__(
        self,
        *,
        working_dir: Path | str | None = None,
        allow_file_operations: bool = True,
        auto_grant_permissions: bool = True,
    ) -> None:
        """Initialize headless ACP client.

        Args:
            working_dir: Default working directory for operations
            allow_file_operations: Whether to allow file read/write operations
            auto_grant_permissions: Whether to automatically grant all permissions
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.allow_file_operations = allow_file_operations
        self.auto_grant_permissions = auto_grant_permissions

        # Process management for terminals
        self.process_manager = ProcessManager()
        self.terminals: dict[str, str] = {}  # terminal_id -> process_id

        # Tracking for testing/debugging
        self.notifications: list[SessionNotification] = []
        self.permission_requests: list[RequestPermissionRequest] = []

    async def handle_request(self, request: AgentRequest) -> ClientResponse:  # noqa: PLR0911
        """Handle any agent request and return appropriate response."""
        match request:
            case WriteTextFileRequest():
                return await self._write_text_file(request)
            case ReadTextFileRequest():
                return await self._read_text_file(request)
            case RequestPermissionRequest():
                return await self._request_permission(request)
            case CreateTerminalRequest():
                return await self._create_terminal(request)
            case TerminalOutputRequest():
                return await self._terminal_output(request)
            case ReleaseTerminalRequest():
                return await self._release_terminal(request)
            case WaitForTerminalExitRequest():
                return await self._wait_for_terminal_exit(request)
            case KillTerminalCommandRequest():
                return await self._kill_terminal(request)
            case _:
                msg = f"Unknown request type: {type(request)}"
                raise ValueError(msg)

    async def handle_notification(self, notification: SessionNotification) -> None:
        """Handle session update notification."""
        logger.debug(
            "Session update for %s: %s",
            notification.session_id,
            type(notification.update).__name__,
        )
        self.notifications.append(notification)

    async def _request_permission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        """Handle permission requests.

        Args:
            params: Permission request parameters

        Returns:
            Permission response - grants if auto_grant_permissions is True
        """
        self.permission_requests.append(params)

        tool_name = params.tool_call.title or "operation"
        logger.info("Permission requested for %s", tool_name)

        if self.auto_grant_permissions and params.options:
            # Grant permission using first available option
            option_id = params.options[0].option_id
            logger.debug("Auto-granting permission for %s", tool_name)
            return RequestPermissionResponse(outcome=AllowedOutcome(option_id=option_id))

        logger.debug("Denying permission for %s", tool_name)
        return RequestPermissionResponse(outcome=DeniedOutcome())

    async def _read_text_file(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        """Read text from file.

        Args:
            params: File read request parameters

        Returns:
            File content response

        Raises:
            RuntimeError: If file operations not allowed
            FileNotFoundError: If file doesn't exist
        """
        if not self.allow_file_operations:
            msg = "File operations not allowed"
            raise RuntimeError(msg)

        try:
            path = Path(params.path)

            if not path.exists():
                msg = f"File not found: {params.path}"
                raise FileNotFoundError(msg)  # noqa: TRY301

            content = path.read_text(encoding="utf-8")

            # Apply line filtering if requested
            if params.line is not None or params.limit is not None:
                lines = content.splitlines(keepends=True)
                start_line = (params.line - 1) if params.line else 0
                end_line = start_line + params.limit if params.limit else len(lines)
                content = "".join(lines[start_line:end_line])

            logger.debug("Read file %s (%d chars)", params.path, len(content))
            return ReadTextFileResponse(content=content)

        except Exception:
            logger.exception("Failed to read file %s", params.path)
            raise

    async def _write_text_file(
        self, params: WriteTextFileRequest
    ) -> WriteTextFileResponse:
        """Write text to file.

        Args:
            params: File write request parameters

        Returns:
            Empty write response

        Raises:
            RuntimeError: If file operations not allowed
        """
        if not self.allow_file_operations:
            msg = "File operations not allowed"
            raise RuntimeError(msg)

        try:
            path = Path(params.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(params.content, encoding="utf-8")

            logger.debug("Wrote file %s (%d chars)", params.path, len(params.content))
            return WriteTextFileResponse()

        except Exception:
            logger.exception("Failed to write file %s", params.path)
            raise

    async def _create_terminal(
        self, params: CreateTerminalRequest
    ) -> CreateTerminalResponse:
        """Create a new terminal session using ProcessManager.

        Args:
            params: Terminal creation parameters

        Returns:
            Terminal creation response with terminal_id
        """
        try:
            process_id = await self.process_manager.start_process(
                command=params.command,
                args=list(params.args) if params.args else None,
                cwd=params.cwd or str(self.working_dir),
                env={var.name: var.value for var in (params.env or [])},
                output_limit=params.output_byte_limit,
            )
            terminal_id = f"term_{uuid.uuid4().hex[:8]}"
            self.terminals[terminal_id] = process_id
            logger.info(
                "Created terminal %s for command: %s",
                terminal_id,
                f"{params.command} {' '.join(params.args or [])}",
            )

            return CreateTerminalResponse(terminal_id=terminal_id)

        except Exception:
            logger.exception("Failed to create terminal for command: %s", params.command)
            raise

    async def _terminal_output(
        self, params: TerminalOutputRequest
    ) -> TerminalOutputResponse:
        """Get output from terminal.

        Args:
            params: Terminal output request parameters

        Returns:
            Terminal output response

        Raises:
            ValueError: If terminal not found
        """
        terminal_id = params.terminal_id

        if terminal_id not in self.terminals:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        try:
            process_id = self.terminals[terminal_id]
            output = await self.process_manager.get_output(process_id)
            return TerminalOutputResponse(
                output=output.combined, truncated=output.truncated
            )
        except Exception:
            logger.exception("Failed to get output for terminal %s", terminal_id)
            raise

    async def _wait_for_terminal_exit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        """Wait for terminal process to exit.

        Args:
            params: Terminal wait parameters

        Returns:
            Terminal exit response with exit code

        Raises:
            ValueError: If terminal not found
        """
        terminal_id = params.terminal_id

        if terminal_id not in self.terminals:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        try:
            process_id = self.terminals[terminal_id]
            exit_code = await self.process_manager.wait_for_exit(process_id)
            logger.debug("Terminal %s exited with code %d", terminal_id, exit_code)
            return WaitForTerminalExitResponse(exit_code=exit_code)
        except Exception:
            logger.exception("Failed to wait for terminal %s", terminal_id)
            raise

    async def _kill_terminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse:
        """Kill terminal process.

        Args:
            params: Terminal kill parameters

        Returns:
            Empty kill response

        Raises:
            ValueError: If terminal not found
        """
        terminal_id = params.terminal_id

        if terminal_id not in self.terminals:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        try:
            process_id = self.terminals[terminal_id]
            await self.process_manager.kill_process(process_id)

            logger.info("Killed terminal %s", terminal_id)
            return KillTerminalCommandResponse()

        except Exception:
            logger.exception("Failed to kill terminal %s", terminal_id)
            raise

    async def _release_terminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse:
        """Release terminal resources.

        Args:
            params: Terminal release parameters

        Returns:
            Empty release response

        Raises:
            ValueError: If terminal not found
        """
        terminal_id = params.terminal_id

        if terminal_id not in self.terminals:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        try:
            process_id = self.terminals[terminal_id]
            await self.process_manager.release_process(process_id)

            # Remove from our tracking
            del self.terminals[terminal_id]

            logger.info("Released terminal %s", terminal_id)
            return ReleaseTerminalResponse()

        except Exception:
            logger.exception("Failed to release terminal %s", terminal_id)
            raise

    async def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up headless client resources")

        # Clean up all terminals
        for terminal_id in list(self.terminals.keys()):
            try:
                process_id = self.terminals[terminal_id]
                await self.process_manager.release_process(process_id)
                del self.terminals[terminal_id]
            except Exception:
                logger.exception("Error cleaning up terminal %s", terminal_id)

        # Clean up process manager
        await self.process_manager.cleanup()

        logger.info("Headless client cleanup completed")

    # Testing/debugging helpers

    def get_session_updates(self) -> list[SessionNotification]:
        """Get all received session updates."""
        return self.notifications.copy()

    def clear_session_updates(self) -> None:
        """Clear all stored session updates."""
        self.notifications.clear()

    def get_permission_requests(self) -> list[RequestPermissionRequest]:
        """Get all permission requests."""
        return self.permission_requests.copy()

    def clear_permission_requests(self) -> None:
        """Clear all stored permission requests."""
        self.permission_requests.clear()

    def list_active_terminals(self) -> list[str]:
        """List all active terminal IDs."""
        return list(self.terminals.keys())
