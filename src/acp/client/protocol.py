from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from acp.schema import (
        CreateTerminalRequest,
        CreateTerminalResponse,
        KillTerminalCommandRequest,
        KillTerminalCommandResponse,
        ReadTextFileRequest,
        ReadTextFileResponse,
        ReleaseTerminalRequest,
        ReleaseTerminalResponse,
        RequestPermissionRequest,
        RequestPermissionResponse,
        SessionNotification,
        TerminalOutputRequest,
        TerminalOutputResponse,
        WaitForTerminalExitRequest,
        WaitForTerminalExitResponse,
        WriteTextFileRequest,
        WriteTextFileResponse,
    )


class BaseClient(Protocol):
    """Base client interface for ACP - always required."""

    async def request_permission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse: ...

    async def session_update(self, params: SessionNotification) -> None: ...


class FsCapability(Protocol):
    """Client capability for filesystem operations."""

    async def write_text_file(
        self, params: WriteTextFileRequest
    ) -> WriteTextFileResponse | None: ...

    async def read_text_file(
        self, params: ReadTextFileRequest
    ) -> ReadTextFileResponse: ...


class TerminalCapability(Protocol):
    """Client capability for terminal operations."""

    async def create_terminal(
        self, params: CreateTerminalRequest
    ) -> CreateTerminalResponse: ...

    async def terminal_output(
        self, params: TerminalOutputRequest
    ) -> TerminalOutputResponse: ...

    async def release_terminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse | None: ...

    async def wait_for_terminal_exit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse: ...

    async def kill_terminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse | None: ...


class ExtensibilityCapability(Protocol):
    """Client capability for extension methods."""

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None: ...


class Client(BaseClient, FsCapability, TerminalCapability, ExtensibilityCapability):
    """High-level client interface for interacting with an ACP server.

    Includes all client capabilities.
    New implementations should inherit from specific capability protocols instead.
    """
