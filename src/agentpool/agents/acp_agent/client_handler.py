"""ACP Agent - MessageNode wrapping an external ACP subprocess."""

from __future__ import annotations

import asyncio
from asyncio import Event
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from pydantic_ai import PartDeltaEvent, TextPartDelta

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
from agentpool.agents.exceptions import OperationNotAllowedError, ResourceNotFoundError
from agentpool.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from exxec import ExecutionEnvironment
    from slashed import Command

    from acp.schema import (
        AvailableCommand,
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
    from agentpool.agents.acp_agent import ACPAgent
    from agentpool.agents.acp_agent.session_state import ACPSessionState
    from agentpool.ui.base import InputProvider

logger = get_logger(__name__)


class TimeoutableEvent(Event):
    """Event that can be waited on with a timeout."""

    async def wait_with_timeout(self, timeout: float | None = None) -> bool:
        if timeout:
            return await asyncio.wait_for(super().wait(), timeout)
        return await super().wait()


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
        self._update_event = TimeoutableEvent()
        # Map ACP terminal IDs to process manager IDs (for local execution only)
        self._terminal_to_process: dict[str, str] = {}
        # Copy auto_approve from agent (can be updated via set_auto_approve)
        self.auto_approve: bool = agent.auto_approve

    @property
    def env(self) -> ExecutionEnvironment:
        """Get execution environment for subprocess requests.

        Uses the agent's client_env which handles subprocess file/terminal
        operations. Falls back to agent's main env if not explicitly configured.
        """
        return self._agent.client_env

    @property
    def allow_file_operations(self) -> bool:
        if (caps := self._agent._init_request.client_capabilities) and caps.fs:
            return bool(caps.fs.read_text_file or caps.fs.write_text_file)
        return False

    @property
    def allow_terminal(self) -> bool:
        caps = self._agent._init_request.client_capabilities
        return bool(caps and caps.terminal)

    async def session_update(self, params: SessionNotification[Any]) -> None:
        """Handle session update notifications from the agent.

        Some updates are state changes (mode, model, config) that should update
        session state. Others are stream data (text chunks, tool calls) that
        should be queued for consumption.

        Raw updates are stored as the single source of truth. Conversion to
        native events happens lazily during streaming consumption.
        """
        from tokonomics.model_discovery.model_info import ModelInfo

        from acp.schema import (
            AvailableCommandsUpdate,
            ConfigOptionUpdate,
            CurrentModelUpdate,
            CurrentModeUpdate,
        )
        from agentpool.agents.modes import ConfigOptionChanged, ModeInfo

        # Handle state updates - these modify session state, not stream data
        match params.update:
            case CurrentModeUpdate(current_mode_id=mode_id):
                if self.state.modes:
                    self.state.modes.current_mode_id = mode_id
                    # Find ModeInfo and emit signal
                    if acp_mode := next(
                        (m for m in self.state.modes.available_modes if m.id == mode_id), None
                    ):
                        mode_info = ModeInfo(
                            id=acp_mode.id,
                            name=acp_mode.name,
                            description=acp_mode.description or "",
                            category_id="mode",  # Old modes API is for operational modes
                        )
                        await self._agent.state_updated.emit(mode_info)
                self.state.current_mode_id = mode_id
                logger.debug("Mode updated", mode_id=mode_id)
                self._update_event.set()
                return

            case CurrentModelUpdate(current_model_id=model_id):
                self.state.current_model_id = model_id
                if state := self.state.models:
                    state.current_model_id = model_id
                    # Find ModelInfo and emit signal
                    if m := next(
                        (m for m in state.available_models if m.model_id == model_id), None
                    ):
                        info = ModelInfo(id=m.model_id, name=m.name, description=m.description)
                        await self._agent.state_updated.emit(info)
                logger.debug("Model updated", model_id=model_id)
                self._update_event.set()
                return

            case ConfigOptionUpdate(config_id=config_id, value_id=value_id):
                # Update the config option in state
                for config_opt in self.state.config_options:
                    if config_opt.id == config_id:
                        config_opt.current_value = value_id
                        break
                change = ConfigOptionChanged(config_id=config_id, value_id=value_id)
                await self._agent.state_updated.emit(change)
                logger.debug("Config option updated", config_id=config_id, value_id=value_id)
                self._update_event.set()
                return

            case AvailableCommandsUpdate() as update:
                self.state.available_commands = update
                # Populate command store with remote commands
                self._populate_command_store(update.available_commands)
                # Emit to parent session - remote commands will be merged with local ones.
                # The "way back" works because session.split_commands() only extracts
                # LOCAL commands; remote commands pass through to the agent prompt.
                await self._agent.state_updated.emit(update)
                logger.debug("Available commands updated", count=len(update.available_commands))
                self._update_event.set()
                return

        # TODO: AgentPlanUpdate handling is complex and needs design work.
        # Options:
        # 1. Update pool.todos - requires merging with existing todos
        # 2. Pass through to UI - but then todos aren't centrally managed
        # 3. Switch to agent-owned todos instead of pool-owned
        # For now, AgentPlanUpdate falls through to stream data.

        # Store raw update - conversion happens lazily during consumption
        self.state.add_update(params.update)
        self._update_event.set()

    async def request_permission(  # noqa: PLR0911
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        """Handle permission requests via InputProvider."""
        name = params.tool_call.title or "operation"
        logger.info("Permission requested", tool_name=name)
        # Check auto_approve FIRST, before any forwarding
        # This ensures "bypass permissions" mode works even for nested ACP agents
        if self.auto_approve and params.options:
            option_id = params.options[0].option_id
            logger.debug("Auto-granting permission (auto_approve=True)", tool_name=name)
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
            else:  # Fall through to old logic
                return response

        if self._input_provider:
            tc = params.tool_call
            args = tc.raw_input if isinstance(tc.raw_input, dict) else {}
            ctx = self._agent.get_context(
                tool_call_id=tc.tool_call_id,
                tool_name=tc.title,
                tool_input=args,
            )
            try:
                result = await self._input_provider.get_tool_confirmation(
                    ctx,
                    tool_name=tc.tool_call_id,
                    tool_description=name,
                    args=args,
                )
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
            raise OperationNotAllowedError("File operations")
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

        except (FileNotFoundError, KeyError):
            # Match Zed behavior: return empty string for non-existent files
            logger.debug("File not found, returning empty string", path=params.path)
            return ReadTextFileResponse(content="")
        except Exception:
            logger.exception("Failed to read file", path=params.path)
            raise

    async def write_text_file(self, params: WriteTextFileRequest) -> WriteTextFileResponse:
        """Write text to file via ExecutionEnvironment filesystem."""
        if not self.allow_file_operations:
            raise OperationNotAllowedError("File operations")
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
        """Create a new terminal session via the configured ExecutionEnvironment.

        The ProcessManager implementation determines where the terminal runs
        (local, Docker, E2B, SSH, or forwarded to parent ACP client like Zed).

        The terminal_id returned by process_manager.start_process() is used directly.
        """
        if not self.allow_terminal:
            raise OperationNotAllowedError("Terminal operations")

        try:
            terminal_id = await self.env.process_manager.start_process(
                command=params.command,
                args=list(params.args) if params.args else None,
                cwd=params.cwd,
                env={var.name: var.value for var in params.env or []},
            )
        except Exception:
            logger.exception("Failed to create terminal", command=params.command)
            raise
        else:
            # Use the ID from process_manager directly - for ACPProcessManager this
            # is already the parent's terminal ID (e.g., Zed's)
            self._terminal_to_process[terminal_id] = terminal_id
            logger.info("Created terminal", terminal_id=terminal_id, command=params.command)
            return CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(self, params: TerminalOutputRequest) -> TerminalOutputResponse:
        """Get output from terminal via ProcessManager."""
        if not self.allow_terminal:
            raise OperationNotAllowedError("Terminal operations")

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            raise ResourceNotFoundError("Terminal", terminal_id)

        proc_output = await self.env.process_manager.get_output(terminal_id)
        output = proc_output.combined or proc_output.stdout or ""
        return TerminalOutputResponse(output=output, truncated=proc_output.truncated)

    async def wait_for_terminal_exit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        """Wait for terminal process to exit via ProcessManager."""
        if not self.allow_terminal:
            raise OperationNotAllowedError("Terminal operations")

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            raise ResourceNotFoundError("Terminal", terminal_id)

        exit_code = await self.env.process_manager.wait_for_exit(terminal_id)
        logger.debug("Terminal exited", terminal_id=terminal_id, exit_code=exit_code)
        return WaitForTerminalExitResponse(exit_code=exit_code)

    async def kill_terminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse:
        """Kill terminal process via ProcessManager."""
        if not self.allow_terminal:
            raise OperationNotAllowedError("Terminal operations")

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            raise ResourceNotFoundError("Terminal", terminal_id)

        await self.env.process_manager.kill_process(terminal_id)
        logger.info("Killed terminal", terminal_id=terminal_id)
        return KillTerminalCommandResponse()

    async def release_terminal(self, params: ReleaseTerminalRequest) -> ReleaseTerminalResponse:
        """Release terminal resources via ProcessManager."""
        if not self.allow_terminal:
            raise OperationNotAllowedError("Terminal operations")

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            raise ResourceNotFoundError("Terminal", terminal_id)

        await self.env.process_manager.release_process(terminal_id)
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

    def _populate_command_store(self, commands: Sequence[AvailableCommand]) -> None:
        """Populate the agent's command store with remote ACP commands.

        Args:
            commands: List of AvailableCommand objects from the remote agent
        """
        store = self._agent.command_store

        for cmd in commands:
            command = self._create_acp_command(cmd)
            # Unregister if already exists (in case of update)
            if store.get_command(cmd.name):
                store.unregister_command(cmd.name)
            store.register_command(command)

        logger.debug("Populated command store", command_count=len(store.list_commands()))

    def _create_acp_command(self, cmd: AvailableCommand) -> Command:
        """Create a slashed Command from an ACP AvailableCommand.

        The command, when executed, sends a prompt with the slash command
        to the remote ACP agent.

        Args:
            cmd: AvailableCommand from remote agent

        Returns:
            A slashed Command that sends the command to the remote agent
        """
        from slashed import Command

        async def run_cmd(ctx: Any, args: list[str], kwargs: dict[str, str]) -> None:
            """Execute the remote ACP slash command."""
            # Build command string
            args_str = " ".join(args) if args else ""
            if kwargs:
                kwargs_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
                args_str = f"{args_str} {kwargs_str}".strip()
            # Execute via agent run_stream - the slash command goes as a prompt
            async for event in self._agent.run_stream(f"/{cmd.name} {args_str}".strip()):
                # Extract text from PartDeltaEvent with TextPartDelta
                if isinstance(event, PartDeltaEvent):
                    delta = event.delta
                    if isinstance(delta, TextPartDelta):
                        await ctx.print(delta.content_delta)

        return Command.from_raw(
            run_cmd,
            name=cmd.name,
            description=cmd.description,
            category="remote",
            usage=cmd.input.root.hint if cmd.input else None,
        )


if __name__ == "__main__":
    from agentpool.agents.acp_agent import ACPAgent

    async def main() -> None:
        """Demo: Basic call to an ACP agent."""
        args = ["run", "agentpool", "serve-acp"]
        cwd = str(Path.cwd())
        async with ACPAgent(command="uv", args=args, cwd=cwd, event_handlers=["detailed"]) as agent:
            print("Response (streaming): ", end="", flush=True)
            async for chunk in agent.run_stream("Say hello briefly."):
                print(chunk, end="", flush=True)

    anyio.run(main)
