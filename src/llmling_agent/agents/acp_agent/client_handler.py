"""ACP Agent - MessageNode wrapping an external ACP subprocess."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any
import uuid

from acp.client.protocol import Client
from acp.schema import (
    CreateTerminalResponse,
    KillTerminalCommandResponse,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalOutputResponse,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from llmling_agent.log import get_logger
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from exxec import ExecutionEnvironment

    from acp.schema import (
        CreateTerminalRequest,
        KillTerminalCommandRequest,
        ReadTextFileRequest,
        ReleaseTerminalRequest,
        RequestPermissionRequest,
        SessionNotification,
        TerminalOutputRequest,
        WaitForTerminalExitRequest,
        WriteTextFileRequest,
    )
    from llmling_agent.agents.acp_agent import ACPAgent
    from llmling_agent.agents.acp_agent.session_state import ACPSessionState
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.nodes import ToolConfirmationMode

logger = get_logger(__name__)


class ACPClientHandler(Client):
    """Client handler that collects session updates and handles agent requests.

    This implements the full ACP Client protocol including:
    - Session update collection (text chunks, thoughts, tool calls)
    - Filesystem operations (read/write files) via ExecutionEnvironment
    - Terminal operations (create, output, kill, release) via ProcessManager
    - Permission request handling via InputProvider

    The handler accumulates session updates in an ACPSessionState instance,
    allowing the ACPAgent to build the final response from streamed chunks.

    Uses ExecutionEnvironment for all file and process operations, enabling
    swappable backends (local, Docker, E2B, SSH, etc.).

    The handler holds a reference to the parent ACPAgent, delegating env access
    to ensure the env stays in sync when reassigned externally.
    """

    def __init__(
        self,
        agent: ACPAgent[Any],
        state: ACPSessionState,
        input_provider: InputProvider | None = None,
    ) -> None:
        self._agent = agent
        self.state = state
        self._input_provider = input_provider
        self._update_event = asyncio.Event()
        # Map ACP terminal IDs to process manager IDs
        self._terminal_to_process: dict[str, str] = {}
        # Copy tool confirmation mode from agent (can be updated via set_tool_confirmation_mode)
        self.tool_confirmation_mode: ToolConfirmationMode = agent.tool_confirmation_mode

    @property
    def env(self) -> ExecutionEnvironment:
        """Get execution environment from parent agent."""
        return self._agent.env

    @property
    def allow_file_operations(self) -> bool:
        return self._agent.config.allow_file_operations

    @property
    def allow_terminal(self) -> bool:
        return self._agent.config.allow_terminal

    async def session_update(self, params: SessionNotification[Any]) -> None:
        """Handle session update notifications from the agent."""
        from llmling_agent.agents.acp_agent.acp_converters import acp_to_native_event

        update = params.update
        # Convert to native event and queue it
        if native_event := acp_to_native_event(update):
            self.state.events.append(native_event)
        self._update_event.set()

    async def request_permission(  # noqa: PLR0911
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        """Handle permission requests via InputProvider."""
        name = params.tool_call.title or "operation"
        logger.info("Permission requested", tool_name=name)

        # Check tool_confirmation_mode FIRST, before any forwarding
        # This ensures "bypass permissions" mode works even for nested ACP agents
        if self.tool_confirmation_mode == "never" and params.options:
            option_id = params.options[0].option_id
            logger.debug("Auto-granting permission (tool_confirmation_mode=never)", tool_name=name)
            return RequestPermissionResponse.allowed(option_id)

        # Try callback second (forwards to parent session for nested ACP agents)
        if self._agent.acp_permission_callback:
            # return RequestPermissionResponse.allowed(option_id=params.options[0].option_id) # "acceptEdits"  # noqa: E501
            try:
                logger.debug("Forwarding permission via callback", tool_name=name)
                response = await self._agent.acp_permission_callback(params)
                logger.debug(
                    "Permission response received", tool_name=name, outcome=response.outcome.outcome
                )
            except Exception:
                logger.exception("Failed to forward permission via callback")
                # Fall through to old logic
            else:
                return response

        if self._input_provider:
            ctx = self._agent.context  # Use the agent's existing NodeContext
            # Create a dummy tool representation from ACP params
            tool = Tool(callable=lambda: None, name=params.tool_call.tool_call_id, description=name)
            # Extract arguments - ACP doesn't expose them in ToolCall
            try:
                result = await self._input_provider.get_tool_confirmation(ctx, tool=tool, args={})
                # Map confirmation result to ACP response
                if result == "allow":
                    option_id = params.options[0].option_id if params.options else "allow"
                    return RequestPermissionResponse.allowed(option_id)
                if result == "skip":
                    return RequestPermissionResponse.denied()
                return RequestPermissionResponse.denied()  # abort_run

            except Exception:
                logger.exception("Failed to get permission via input provider")
                return RequestPermissionResponse.denied()

        logger.debug("Denying permission (no input provider)", tool_name=name)
        return RequestPermissionResponse.denied()

    async def read_text_file(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        """Read text from file via ExecutionEnvironment filesystem."""
        if not self.allow_file_operations:
            raise RuntimeError("File operations not allowed")

        fs = self.env.get_fs()
        try:
            content_bytes = await fs._cat_file(params.path)
            content = content_bytes.decode("utf-8")
            # Apply line filtering if requested
            if params.line is not None or params.limit is not None:
                lines = content.splitlines(keepends=True)
                start_line = (params.line - 1) if params.line else 0
                end_line = start_line + params.limit if params.limit else len(lines)
                content = "".join(lines[start_line:end_line])

            logger.debug("Read file", path=params.path, num_chars=len(content))
            return ReadTextFileResponse(content=content)

        except FileNotFoundError:
            logger.exception("File not found", path=params.path)
            raise
        except Exception:
            logger.exception("Failed to read file", path=params.path)
            raise

    async def write_text_file(self, params: WriteTextFileRequest) -> WriteTextFileResponse:
        """Write text to file via ExecutionEnvironment filesystem."""
        if not self.allow_file_operations:
            raise RuntimeError("File operations not allowed")
        fs = self.env.get_fs()
        content_bytes = params.content.encode("utf-8")
        parent = str(Path(params.path).parent)
        try:
            if parent and parent != ".":  # Ensure parent directory exists
                await fs._makedirs(parent, exist_ok=True)
            await fs._pipe_file(params.path, content_bytes)
            logger.debug("Wrote file", path=params.path, num_chars=len(params.content))
            return WriteTextFileResponse()
        except Exception:
            logger.exception("Failed to write file", path=params.path)
            raise

    async def create_terminal(self, params: CreateTerminalRequest) -> CreateTerminalResponse:
        """Create a new terminal session via ProcessManager."""
        if not self.allow_terminal:
            raise RuntimeError("Terminal operations not allowed")
        try:
            process_id = await self.env.process_manager.start_process(
                command=params.command,
                args=list(params.args) if params.args else None,
                cwd=params.cwd,
                env={var.name: var.value for var in params.env or []},
            )
        except Exception:
            logger.exception("Failed to create terminal", command=params.command)
            raise
        else:
            terminal_id = f"term_{uuid.uuid4().hex[:8]}"
            self._terminal_to_process[terminal_id] = process_id
            msg = "Created terminal"
            logger.info(msg, terminal_id=terminal_id, process_id=process_id, cmd=params.command)
            return CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(self, params: TerminalOutputRequest) -> TerminalOutputResponse:
        """Get output from terminal via ProcessManager."""
        if not self.allow_terminal:
            raise RuntimeError("Terminal operations not allowed")

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            raise ValueError(f"Terminal {terminal_id} not found")

        process_id = self._terminal_to_process[terminal_id]
        proc_output = await self.env.process_manager.get_output(process_id)
        output = proc_output.combined or proc_output.stdout or ""
        return TerminalOutputResponse(output=output, truncated=proc_output.truncated)

    async def wait_for_terminal_exit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        """Wait for terminal process to exit via ProcessManager."""
        if not self.allow_terminal:
            raise RuntimeError("Terminal operations not allowed")

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            raise ValueError(f"Terminal {terminal_id} not found")

        process_id = self._terminal_to_process[terminal_id]
        exit_code = await self.env.process_manager.wait_for_exit(process_id)
        logger.debug("Terminal exited", terminal_id=terminal_id, exit_code=exit_code)
        return WaitForTerminalExitResponse(exit_code=exit_code)

    async def kill_terminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse:
        """Kill terminal process via ProcessManager."""
        if not self.allow_terminal:
            raise RuntimeError("Terminal operations not allowed")

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            raise ValueError(f"Terminal {terminal_id} not found")

        process_id = self._terminal_to_process[terminal_id]
        await self.env.process_manager.kill_process(process_id)
        logger.info("Killed terminal", terminal_id=terminal_id)
        return KillTerminalCommandResponse()

    async def release_terminal(self, params: ReleaseTerminalRequest) -> ReleaseTerminalResponse:
        """Release terminal resources via ProcessManager."""
        if not self.allow_terminal:
            raise RuntimeError("Terminal operations not allowed")

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            raise ValueError(f"Terminal {terminal_id} not found")
        process_id = self._terminal_to_process[terminal_id]
        await self.env.process_manager.release_process(process_id)
        del self._terminal_to_process[terminal_id]
        logger.info("Released terminal", terminal_id=terminal_id)
        return ReleaseTerminalResponse()

    async def cleanup(self) -> None:
        """Clean up all resources."""
        for terminal_id, process_id in list(self._terminal_to_process.items()):
            try:
                await self.env.process_manager.release_process(process_id)
            except Exception:
                logger.exception("Error cleaning up terminal", terminal_id=terminal_id)

        self._terminal_to_process.clear()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods."""
        logger.debug("Extension method called", method=method)
        return {"ok": True, "method": method}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications."""
        logger.debug("Extension notification", method=method)


if __name__ == "__main__":
    from llmling_agent.agents.acp_agent import ACPAgent

    async def main() -> None:
        """Demo: Basic call to an ACP agent."""
        args = ["run", "llmling-agent", "serve-acp", "--model-provider", "openai"]
        cwd = str(Path.cwd())
        async with ACPAgent(command="uv", args=args, cwd=cwd, event_handlers=["detailed"]) as agent:
            print("Response (streaming): ", end="", flush=True)
            async for chunk in agent.run_stream("Say hello briefly."):
                print(chunk, end="", flush=True)

    asyncio.run(main())
