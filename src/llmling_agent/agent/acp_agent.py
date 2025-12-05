"""ACP Agent - MessageNode wrapping an external ACP subprocess.

This module provides an agent implementation that communicates with external
ACP (Agent Client Protocol) servers via stdio, enabling integration of any
ACP-compatible agent into the llmling-agent pool.

The ACPAgent class acts as an ACP client, spawning an ACP server subprocess
and communicating with it via JSON-RPC over stdio. This allows:
- Integration of external ACP-compatible agents (like claude-code-acp)
- Composition with native llmling agents via connections, teams, etc.
- Full ACP protocol support including file operations and terminals

Example:
    ```python
    config = ACPAgentConfig(
        command="claude-code-acp",
        name="claude_coder",
        cwd="/path/to/project",
    )
    async with ACPAgent(config) as agent:
        result = await agent.run("Write a hello world program")
        print(result.content)
    ```
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field as dataclass_field
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, overload
import uuid

from acp.client.connection import ClientSideConnection
from acp.client.protocol import Client
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    CreateTerminalResponse,
    DeniedOutcome,
    InitializeRequest,
    KillTerminalCommandResponse,
    NewSessionRequest,
    PromptRequest,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from llmling_agent.log import get_logger
from llmling_agent.messaging import ChatMessage
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent.models.acp_agents import ACPAgentConfig
from llmling_agent.talk.stats import MessageStats


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

    from anyenv.code_execution import ExecutionEnvironment
    from evented.configs import EventConfig
    from tokonomics.model_discovery import ProviderType

    from acp.agent.protocol import Agent as ACPAgentProtocol
    from acp.schema import (
        CreateTerminalRequest,
        InitializeResponse,
        KillTerminalCommandRequest,
        PromptResponse,
        ReadTextFileRequest,
        ReleaseTerminalRequest,
        RequestPermissionRequest,
        SessionNotification,
        TerminalOutputRequest,
        WaitForTerminalExitRequest,
        WriteTextFileRequest,
    )
    from llmling_agent.agent.events import RichAgentStreamEvent
    from llmling_agent.common_types import PromptCompatible
    from llmling_agent.delegation import AgentPool
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.models.acp_agents import BaseACPAgentConfig


logger = get_logger(__name__)

PROTOCOL_VERSION = 1


@dataclass
class ACPSessionState:
    """Tracks state of an ACP session."""

    session_id: str
    """The session ID from the ACP server."""

    text_chunks: list[str] = dataclass_field(default_factory=list)
    """Accumulated text chunks."""

    thought_chunks: list[str] = dataclass_field(default_factory=list)
    """Accumulated thought/reasoning chunks."""

    tool_calls: list[dict[str, Any]] = dataclass_field(default_factory=list)
    """Tool call records."""

    events: list[Any] = dataclass_field(default_factory=list)
    """Queue of native events converted from ACP updates."""

    is_complete: bool = False
    """Whether the prompt processing is complete."""

    stop_reason: str | None = None
    """Reason processing stopped."""

    current_model_id: str | None = None
    """Current model ID from session state."""


class ACPClientHandler(Client):
    """Client handler that collects session updates and handles agent requests.

    This implements the full ACP Client protocol including:
    - Session update collection (text chunks, thoughts, tool calls)
    - Filesystem operations (read/write files) via ExecutionEnvironment
    - Terminal operations (create, output, kill, release) via ProcessManager
    - Permission request handling

    The handler accumulates session updates in an ACPSessionState instance,
    allowing the ACPAgent to build the final response from streamed chunks.

    Uses ExecutionEnvironment for all file and process operations, enabling
    swappable backends (local, Docker, E2B, SSH, etc.).
    """

    def __init__(
        self,
        state: ACPSessionState,
        *,
        env: ExecutionEnvironment | None = None,
        allow_file_operations: bool = True,
        allow_terminal: bool = True,
        auto_grant_permissions: bool = True,
    ) -> None:
        self.state = state
        self._env = env
        self.allow_file_operations = allow_file_operations
        self.allow_terminal = allow_terminal
        self.auto_grant_permissions = auto_grant_permissions
        self._update_event = asyncio.Event()

        # Map ACP terminal IDs to process manager IDs
        self._terminal_to_process: dict[str, str] = {}

    @property
    def env(self) -> ExecutionEnvironment:
        """Get execution environment, creating default if needed."""
        if self._env is None:
            from anyenv.code_execution import LocalExecutionEnvironment

            self._env = LocalExecutionEnvironment()
        return self._env

    async def session_update(self, params: SessionNotification[Any]) -> None:
        """Handle session update notifications from the agent."""
        from llmling_agent.agent.acp_converters import acp_to_native_event

        update = params.update

        # Convert to native event and queue it
        if native_event := acp_to_native_event(update):
            self.state.events.append(native_event)

        # Also maintain text chunk accumulation for simple access
        match update:
            case AgentMessageChunk(content=TextContentBlock(text=text)):
                self.state.text_chunks.append(text)
            case AgentThoughtChunk(content=TextContentBlock(text=text)):
                self.state.thought_chunks.append(text)
            case ToolCallStart() as tc:
                self.state.tool_calls.append({
                    "id": tc.tool_call_id,
                    "title": tc.title,
                    "kind": tc.kind,
                    "status": tc.status,
                    "input": tc.raw_input,
                    "output": tc.raw_output,
                })
            case ToolCallProgress() as tc:
                # Update existing tool call
                for tool in self.state.tool_calls:
                    if tool["id"] == tc.tool_call_id:
                        if tc.status:
                            tool["status"] = tc.status
                        if tc.raw_output:
                            tool["output"] = tc.raw_output
                        break
            case UserMessageChunk():
                pass  # Echo of user message, ignore
            case _:
                logger.debug("Unhandled session update", update_type=type(update).__name__)
        self._update_event.set()

    async def request_permission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        """Handle permission requests."""
        tool_name = params.tool_call.title or "operation"
        logger.info("Permission requested", tool_name=tool_name)

        if self.auto_grant_permissions and params.options:
            option_id = params.options[0].option_id
            logger.debug("Auto-granting permission", tool_name=tool_name)
            return RequestPermissionResponse(outcome=AllowedOutcome(option_id=option_id))

        logger.debug("Denying permission", tool_name=tool_name)
        return RequestPermissionResponse(outcome=DeniedOutcome())

    async def read_text_file(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        """Read text from file via ExecutionEnvironment filesystem."""
        if not self.allow_file_operations:
            msg = "File operations not allowed"
            raise RuntimeError(msg)

        try:
            fs = self.env.get_fs()
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
            msg = "File operations not allowed"
            raise RuntimeError(msg)

        try:
            fs = self.env.get_fs()
            content_bytes = params.content.encode("utf-8")

            # Ensure parent directory exists
            parent = str(Path(params.path).parent)
            if parent and parent != ".":
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
            msg = "Terminal operations not allowed"
            raise RuntimeError(msg)

        try:
            terminal_id = f"term_{uuid.uuid4().hex[:8]}"

            # Build environment dict from ACP env vars
            env_dict: dict[str, str] | None = None
            if params.env:
                env_dict = {var.name: var.value for var in params.env}

            # Start process via ProcessManager
            process_id = await self.env.process_manager.start_process(
                command=params.command,
                args=list(params.args) if params.args else None,
                cwd=params.cwd,
                env=env_dict,
            )

            # Map terminal ID to process ID
            self._terminal_to_process[terminal_id] = process_id

            logger.info(
                "Created terminal",
                terminal_id=terminal_id,
                process_id=process_id,
                command=params.command,
            )
            return CreateTerminalResponse(terminal_id=terminal_id)

        except Exception:
            logger.exception("Failed to create terminal", command=params.command)
            raise

    async def terminal_output(self, params: TerminalOutputRequest) -> TerminalOutputResponse:
        """Get output from terminal via ProcessManager."""
        if not self.allow_terminal:
            msg = "Terminal operations not allowed"
            raise RuntimeError(msg)

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        process_id = self._terminal_to_process[terminal_id]
        output = await self.env.process_manager.get_output(process_id)

        return TerminalOutputResponse(
            output=output.combined or output.stdout or "",
            truncated=output.truncated,
        )

    async def wait_for_terminal_exit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        """Wait for terminal process to exit via ProcessManager."""
        if not self.allow_terminal:
            msg = "Terminal operations not allowed"
            raise RuntimeError(msg)

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        process_id = self._terminal_to_process[terminal_id]
        exit_code = await self.env.process_manager.wait_for_exit(process_id)

        logger.debug("Terminal exited", terminal_id=terminal_id, exit_code=exit_code)
        return WaitForTerminalExitResponse(exit_code=exit_code)

    async def kill_terminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse | None:
        """Kill terminal process via ProcessManager."""
        if not self.allow_terminal:
            msg = "Terminal operations not allowed"
            raise RuntimeError(msg)

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        process_id = self._terminal_to_process[terminal_id]
        await self.env.process_manager.kill_process(process_id)

        logger.info("Killed terminal", terminal_id=terminal_id)
        return KillTerminalCommandResponse()

    async def release_terminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse | None:
        """Release terminal resources via ProcessManager."""
        if not self.allow_terminal:
            msg = "Terminal operations not allowed"
            raise RuntimeError(msg)

        terminal_id = params.terminal_id
        if terminal_id not in self._terminal_to_process:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

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


class ACPAgent[TDeps = None](MessageNode[TDeps, str]):
    """MessageNode that wraps an external ACP agent subprocess.

    This allows integrating any ACP-compatible agent into the llmling-agent
    pool, enabling composition with native agents via connections, teams, etc.

    The agent manages:
    - Subprocess lifecycle (spawn on enter, terminate on exit)
    - ACP protocol initialization and session creation
    - Prompt execution with session update collection
    - Client-side operations (filesystem, terminals, permissions)

    Supports both blocking `run()` and streaming `run_iter()` execution modes.

    Example with config:
        ```python
        config = ClaudeACPAgentConfig(cwd="/project", model="sonnet")
        agent = ACPAgent(config, agent_pool=pool)
        ```

    Example with kwargs:
        ```python
        agent = ACPAgent(
            command="claude-code-acp",
            cwd="/project",
            providers=["anthropic"],
        )
        ```
    """

    @overload
    def __init__(
        self,
        *,
        config: BaseACPAgentConfig,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        command: str,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        allow_file_operations: bool = True,
        allow_terminal: bool = True,
        auto_grant_permissions: bool = True,
        providers: list[ProviderType] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        config: BaseACPAgentConfig | None = None,
        command: str | None = None,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        allow_file_operations: bool = True,
        allow_terminal: bool = True,
        auto_grant_permissions: bool = True,
        providers: list[ProviderType] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
    ) -> None:
        # Build config from kwargs if not provided
        if config is None:
            if command is None:
                msg = "Either config or command must be provided"
                raise ValueError(msg)
            config = ACPAgentConfig(
                name=name,
                description=description,
                display_name=display_name,
                command=command,
                args=args or [],
                cwd=cwd,
                env=env or {},
                allow_file_operations=allow_file_operations,
                allow_terminal=allow_terminal,
                auto_grant_permissions=auto_grant_permissions,
                providers=list(providers) if providers else [],
            )
        super().__init__(
            name=config.name or config.get_command(),
            description=config.description,
            display_name=config.display_name,
            mcp_servers=config.mcp_servers,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs or list(config.triggers),
        )
        self.config = config
        self._env: ExecutionEnvironment | None = None
        self._process: Process | None = None
        self._connection: ClientSideConnection | None = None
        self._client_handler: ACPClientHandler | None = None
        self._init_response: InitializeResponse | None = None
        self._session_id: str | None = None
        self._state: ACPSessionState | None = None
        self._message_count = 0
        self._total_tokens = 0

    @property
    def context(self) -> NodeContext:
        """Get node context."""
        from llmling_agent.messaging.context import NodeContext
        from llmling_agent.models.manifest import AgentsManifest
        from llmling_agent_config.nodes import NodeConfig

        return NodeContext(
            node_name=self.name,
            pool=self.agent_pool,
            config=NodeConfig(name=self.name, description=self.description),
            definition=AgentsManifest(),
        )

    async def __aenter__(self) -> Self:
        """Start subprocess and initialize ACP connection."""
        await super().__aenter__()
        await self._start_process()
        await self._initialize()
        await self._create_session()
        # Small delay to let subprocess fully initialize before accepting prompts
        await asyncio.sleep(0.3)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up subprocess and connection."""
        await self._cleanup()
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _start_process(self) -> None:
        """Start the ACP server subprocess."""
        env = {**os.environ, **self.config.env}
        cmd = [self.config.get_command(), *self.config.get_args()]
        self.log.info("Starting ACP subprocess", command=cmd)
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self.config.cwd,
            limit=10 * 1024 * 1024,  # 10MB,
        )

        if not self._process.stdin or not self._process.stdout:
            msg = "Failed to create subprocess pipes"
            raise RuntimeError(msg)

    async def _initialize(self) -> None:
        """Initialize the ACP connection."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            msg = "Process not started"
            raise RuntimeError(msg)

        env = self._env or self.config.get_execution_environment()
        self._state = ACPSessionState(session_id="")
        self._client_handler = ACPClientHandler(
            self._state,
            env=env,
            allow_file_operations=self.config.allow_file_operations,
            allow_terminal=self.config.allow_terminal,
            auto_grant_permissions=self.config.auto_grant_permissions,
        )

        def client_factory(agent: ACPAgentProtocol) -> Client:
            return self._client_handler  # type: ignore[return-value]

        self._connection = ClientSideConnection(
            to_client=client_factory,
            input_stream=self._process.stdin,
            output_stream=self._process.stdout,
        )
        init_request = InitializeRequest.create(
            title="LLMling Agent",
            version="0.1.0",
            name="llmling-agent",
            protocol_version=PROTOCOL_VERSION,
            terminal=self.config.allow_terminal,
            read_text_file=self.config.allow_file_operations,
            write_text_file=self.config.allow_file_operations,
        )
        self._init_response = await self._connection.initialize(init_request)
        self.log.info("ACP connection initialized", agent_info=self._init_response.agent_info)

    async def _create_session(self) -> None:
        """Create a new ACP session."""
        if not self._connection:
            msg = "Connection not initialized"
            raise RuntimeError(msg)

        cwd = self.config.cwd or str(Path.cwd())
        session_request = NewSessionRequest(cwd=cwd, mcp_servers=[])
        response = await self._connection.new_session(session_request)
        self._session_id = response.session_id
        if self._state:
            self._state.session_id = self._session_id
            # Store model info from session response
            if response.models:
                self._state.current_model_id = response.models.current_model_id
        model = self._state.current_model_id if self._state else None
        self.log.info("ACP session created", session_id=self._session_id, model=model)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._client_handler:
            try:
                await self._client_handler.cleanup()
            except Exception:
                self.log.exception("Error cleaning up client handler")
            self._client_handler = None

        if self._connection:
            try:
                await self._connection.close()
            except Exception:
                self.log.exception("Error closing ACP connection")
            self._connection = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception:
                self.log.exception("Error terminating ACP process")
            self._process = None

    async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[str]:
        """Execute prompt against ACP agent.

        Sends the prompt to the ACP server and waits for completion.
        Session updates (text chunks, tool calls, etc.) are collected
        and the final text content is returned as a ChatMessage.

        Args:
            *prompts: Prompts to send (will be joined with spaces)
            **kwargs: Additional arguments (unused)

        Returns:
            ChatMessage containing the agent's aggregated text response
        """
        if not self._connection or not self._session_id or not self._state:
            msg = "Agent not initialized - use async context manager"
            raise RuntimeError(msg)

        # Reset state for new prompt
        self._state.text_chunks.clear()
        self._state.thought_chunks.clear()
        self._state.tool_calls.clear()
        self._state.is_complete = False
        self._state.stop_reason = None
        prompt_text = " ".join(str(p) for p in prompts)
        content_blocks = [TextContentBlock(text=prompt_text)]
        prompt_request = PromptRequest(session_id=self._session_id, prompt=content_blocks)
        self.log.debug("Sending prompt to ACP agent", prompt=prompt_text[:100])
        # The prompt call blocks until completion, session updates come via notifications
        response: PromptResponse = await self._connection.prompt(prompt_request)
        self._state.is_complete = True
        self._state.stop_reason = response.stop_reason
        self._message_count += 1
        message = ChatMessage[str](
            content="".join(self._state.text_chunks),
            role="assistant",
            name=self.name,
            message_id=str(uuid.uuid4()),
            conversation_id=self.conversation_id,
            model_name=self._get_model_name(),
            cost_info=None,
        )
        self.message_sent.emit(message)
        return message

    async def run_stream(
        self, *prompts: Any, **kwargs: Any
    ) -> AsyncIterator[RichAgentStreamEvent[str]]:
        """Stream native events as they arrive from ACP agent.

        Yields the same event types as native agents, enabling uniform
        handling regardless of whether the agent is native or ACP-based.

        Args:
            *prompts: Prompts to send (will be joined with spaces)
            **kwargs: Additional arguments (unused)

        Yields:
            RichAgentStreamEvent instances converted from ACP session updates
        """
        from llmling_agent.agent.events import StreamCompleteEvent

        if not self._connection or not self._session_id or not self._state:
            msg = "Agent not initialized - use async context manager"
            raise RuntimeError(msg)

        # Reset state
        self._state.text_chunks.clear()
        self._state.thought_chunks.clear()
        self._state.tool_calls.clear()
        self._state.events.clear()
        self._state.is_complete = False
        self._state.stop_reason = None
        prompt_text = " ".join(str(p) for p in prompts)
        content_blocks = [TextContentBlock(text=prompt_text)]
        prompt_request = PromptRequest(session_id=self._session_id, prompt=content_blocks)
        self.log.debug("Starting streaming prompt", prompt=prompt_text[:100])
        # Run prompt in background
        prompt_task = asyncio.create_task(self._connection.prompt(prompt_request))
        last_idx = 0
        while not prompt_task.done():
            # Wait for new events
            if self._client_handler:
                try:
                    await asyncio.wait_for(self._client_handler._update_event.wait(), timeout=0.05)
                    self._client_handler._update_event.clear()
                except TimeoutError:
                    pass

            # Yield new native events
            while last_idx < len(self._state.events):
                yield self._state.events[last_idx]
                last_idx += 1

        # Yield remaining events after completion
        while last_idx < len(self._state.events):
            yield self._state.events[last_idx]
            last_idx += 1

        # Ensure we catch any exceptions from the prompt task
        response = await prompt_task
        self._state.stop_reason = response.stop_reason
        self._state.is_complete = True
        self._message_count += 1

        # Emit final StreamCompleteEvent with aggregated message
        message = ChatMessage[str](
            content="".join(self._state.text_chunks),
            role="assistant",
            name=self.name,
            message_id=str(uuid.uuid4()),
            conversation_id=self.conversation_id,
            model_name=self._get_model_name(),
        )
        yield StreamCompleteEvent(message=message)
        self.message_sent.emit(message)

    async def run_iter(
        self,
        *prompt_groups: Sequence[PromptCompatible],
    ) -> AsyncIterator[ChatMessage[str]]:
        """Run agent sequentially on multiple prompt groups.

        Args:
            prompt_groups: Groups of prompts to process sequentially

        Yields:
            Response messages in sequence
        """
        for prompts in prompt_groups:
            response = await self.run(*prompts)
            yield response

    def _get_model_name(self) -> str:
        """Get model name from session state or agent info."""
        # Prefer current model from session state
        if self._state and self._state.current_model_id:
            return self._state.current_model_id
        # Fall back to agent info name
        if self._init_response and self._init_response.agent_info:
            return self._init_response.agent_info.name
        return self.config.get_command()

    async def get_stats(self) -> MessageStats:
        """Get message statistics."""
        return MessageStats()


if __name__ == "__main__":

    async def main() -> None:
        """Demo: Basic call to an ACP agent."""
        async with ACPAgent(
            command="uv",
            args=["run", "llmling-agent", "serve-acp", "--model-provider", "openai"],
            name="llmling_acp",
            description="LLMling Agent via ACP",
            cwd=str(Path.cwd()),
        ) as agent:
            print(f"Connected to: {agent._get_model_name()}")
            print(f"Session ID: {agent._session_id}")
            print("-" * 50)
            prompt = "Say hello briefly."
            print(f"Prompt: {prompt}")
            print("-" * 50)

            print("Response (streaming): ", end="", flush=True)
            async for chunk in agent.run_stream(prompt):
                print(chunk, end="", flush=True)
            print()  # newline after streaming
            # Show tool calls if any
            if agent._state and agent._state.tool_calls:
                print("-" * 50)
                print("Tool calls:")
                for tc in agent._state.tool_calls:
                    print(f"  - {tc['title']} ({tc['status']})")

    asyncio.run(main())
