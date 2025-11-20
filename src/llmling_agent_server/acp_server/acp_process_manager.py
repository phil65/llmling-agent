"""ACP-compatible process manager that implements the ProcessManager protocol."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from anyenv.process_manager import ProcessManagerProtocol, ProcessOutput


if TYPE_CHECKING:
    from pathlib import Path

    from llmling_agent_server.acp_server.session import ACPSession


@dataclass
class ACPRunningProcess:
    """Represents a running process via ACP terminal."""

    process_id: str
    terminal_id: str
    command: str
    args: list[str]
    cwd: str | None
    env: dict[str, str]
    created_at: datetime
    output_limit: int | None = None
    exit_code: int | None = None

    async def is_running(self) -> bool:
        """Check if process is still running."""
        return self.exit_code is None

    async def wait(self) -> int:
        """Wait for process to complete and return exit code."""
        # This would be implemented by the caller
        msg = "Use ACPProcessManager.wait_for_exit"
        raise NotImplementedError(msg)

    async def kill(self) -> None:
        """Terminate the process."""
        # This would be implemented by the caller
        msg = "Use ACPProcessManager.kill_process"
        raise NotImplementedError(msg)


class ACPProcessManager(ProcessManagerProtocol):
    """Process manager that uses ACP terminal API for process execution."""

    def __init__(self, session: ACPSession) -> None:
        """Initialize with ACP session.

        Args:
            session: ACP session for terminal operations
        """
        self.session = session
        self._processes: dict[str, ACPRunningProcess] = {}
        self._terminal_to_process: dict[str, str] = {}

    @property
    def processes(self) -> dict[str, ACPRunningProcess]:
        """Get the running processes."""
        return self._processes

    async def start_process(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        output_limit: int | None = None,
    ) -> str:
        """Start a background process via ACP terminal.

        Args:
            command: Command to execute
            args: Command arguments
            cwd: Working directory
            env: Environment variables
            output_limit: Maximum bytes of output to retain

        Returns:
            Process ID for tracking

        Raises:
            Exception: If terminal creation fails
        """
        args = args or []
        env = env or {}

        # Create terminal via ACP
        response = await self.session.requests.create_terminal(
            command=command,
            args=args,
            cwd=str(cwd) if cwd else None,
            env=env,
            output_byte_limit=output_limit,
        )

        terminal_id = response.terminal_id
        process_id = terminal_id  # Use terminal_id directly as process_id

        # Create tracking object
        process = ACPRunningProcess(
            process_id=process_id,
            terminal_id=terminal_id,
            command=command,
            args=args,
            cwd=str(cwd) if cwd else None,
            env=env,
            created_at=datetime.now(),
            output_limit=output_limit,
        )

        self._processes[process_id] = process
        self._terminal_to_process[terminal_id] = process_id

        return process_id

    async def get_output(self, process_id: str) -> ProcessOutput:
        """Get current process output.

        Args:
            process_id: Process identifier

        Returns:
            Current process output

        Raises:
            ValueError: If process ID not found
        """
        if process_id not in self._processes:
            msg = f"Process {process_id} not found"
            raise ValueError(msg)

        process = self._processes[process_id]

        try:
            # Get output from ACP terminal
            response = await self.session.requests.terminal_output(process.terminal_id)

            # ACP combines stdout/stderr, so we put it all in combined
            # and split roughly for stdout (since ACP doesn't separate them)
            combined_output = response.output

            return ProcessOutput(
                stdout=combined_output,  # ACP doesn't separate stdout/stderr
                stderr="",
                combined=combined_output,
                truncated=response.truncated,
                exit_code=process.exit_code,
                signal=None,
            )
        except Exception as e:  # noqa: BLE001
            # Return empty output on error
            return ProcessOutput(
                stdout="",
                stderr=f"Error getting output: {e}",
                combined=f"Error getting output: {e}",
                truncated=False,
                exit_code=process.exit_code,
                signal=None,
            )

    async def wait_for_exit(self, process_id: str) -> int:
        """Wait for process to complete and return exit code.

        Args:
            process_id: Process identifier

        Returns:
            Exit code

        Raises:
            ValueError: If process ID not found
        """
        if process_id not in self._processes:
            msg = f"Process {process_id} not found"
            raise ValueError(msg)

        process = self._processes[process_id]

        if process.exit_code is not None:
            return process.exit_code

        try:
            # Wait for terminal exit via ACP
            response = await self.session.requests.wait_for_terminal_exit(process.terminal_id)
            exit_code = response.exit_code or 0
            process.exit_code = exit_code
        except Exception:  # noqa: BLE001
            # Mark as failed and return error code
            process.exit_code = -1
            return -1
        else:
            return exit_code

    async def kill_process(self, process_id: str) -> None:
        """Terminate a process.

        Args:
            process_id: Process identifier

        Raises:
            ValueError: If process ID not found
        """
        if process_id not in self._processes:
            msg = f"Process {process_id} not found"
            raise ValueError(msg)

        process = self._processes[process_id]

        if process.exit_code is not None:
            return  # Already finished

        try:
            await self.session.requests.kill_terminal(process.terminal_id)
            process.exit_code = -2  # Killed
        except Exception:  # noqa: BLE001
            # Best effort - mark as killed anyway
            process.exit_code = -2

    async def release_process(self, process_id: str) -> None:
        """Release resources for a process.

        Args:
            process_id: Process identifier

        Raises:
            ValueError: If process ID not found
        """
        if process_id not in self._processes:
            msg = f"Process {process_id} not found"
            raise ValueError(msg)

        process = self._processes[process_id]

        with contextlib.suppress(Exception):
            await self.session.requests.release_terminal(process.terminal_id)

        # Remove from tracking
        del self._processes[process_id]
        if process.terminal_id in self._terminal_to_process:
            del self._terminal_to_process[process.terminal_id]

    async def list_processes(self) -> list[str]:
        """List all active process IDs.

        Returns:
            List of process IDs
        """
        return list(self._processes.keys())

    async def get_process_info(self, process_id: str) -> dict[str, Any]:
        """Get process information.

        Args:
            process_id: Process identifier

        Returns:
            Process information dict

        Raises:
            ValueError: If process ID not found
        """
        if process_id not in self._processes:
            msg = f"Process {process_id} not found"
            raise ValueError(msg)

        process = self._processes[process_id]

        return {
            "process_id": process.process_id,
            "terminal_id": process.terminal_id,
            "command": process.command,
            "args": process.args,
            "cwd": process.cwd,
            "created_at": process.created_at.isoformat(),
            "is_running": process.exit_code is None,
            "exit_code": process.exit_code,
        }

    async def cleanup(self) -> None:
        """Clean up all processes and release resources."""
        # Copy list to avoid modification during iteration
        process_ids = list(self._processes.keys())

        for process_id in process_ids:
            with contextlib.suppress(Exception):
                await self.release_process(process_id)
