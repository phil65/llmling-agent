"""Terminal handle implementation. NOTE: not integrated yet."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from acp.acp_requests import ACPRequests
    from acp.schema import TerminalOutputResponse, WaitForTerminalExitResponse


class TerminalHandle:
    """Handle for a terminal session."""

    def __init__(self, terminal_id: str, requests: ACPRequests) -> None:
        self.terminal_id = terminal_id
        self._requests = requests

    async def current_output(self) -> TerminalOutputResponse:
        return await self._requests.terminal_output(self.terminal_id)

    async def wait_for_exit(self) -> WaitForTerminalExitResponse:
        return await self._requests.wait_for_terminal_exit(self.terminal_id)

    async def kill(self) -> None:
        return await self._requests.kill_terminal(self.terminal_id)

    async def release(self) -> None:
        return await self._requests.release_terminal(self.terminal_id)
