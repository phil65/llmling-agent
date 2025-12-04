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
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
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
from llmling_agent.talk.stats import MessageStats
from llmling_agent.utils.tasks import TaskManager
from llmling_agent_config.nodes import NodeConfig


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator
    from types import TracebackType

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
    from llmling_agent.messaging.context import NodeContext


logger = get_logger(__name__)

PROTOCOL_VERSION = 1


class ACPAgentConfig(NodeConfig):
    """Configuration for an external ACP agent."""

    command: str
    """Command to spawn the ACP server."""

    args: list[str] = field(default_factory=list)
    """Arguments to pass to the command."""

    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set."""

    cwd: str | None = None
    """Working directory for the session."""

    allow_file_operations: bool = True
    """Whether to allow file read/write operations."""

    allow_terminal: bool = True
    """Whether to allow terminal operations."""

    auto_grant_permissions: bool = True
    """Whether to automatically grant all permission requests."""


@dataclass
class ACPSessionState:
    """Tracks state of an ACP session."""

    session_id: str
    """The session ID from the ACP server."""

    text_chunks: list[str] = field(default_factory=list)
    """Accumulated text chunks."""

    thought_chunks: list[str] = field(default_factory=list)
    """Accumulated thought/reasoning chunks."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Tool call records."""

    events: list[Any] = field(default_factory=list)
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
    - Filesystem operations (read/write files)
    - Terminal operations (create, output, kill, release)
    - Permission request handling

    The handler accumulates session updates in an ACPSessionState instance,
    allowing the ACPAgent to build the final response from streamed chunks.
    """

    def __init__(
        self,
        state: ACPSessionState,
        *,
        working_dir: Path | None = None,
        allow_file_operations: bool = True,
        allow_terminal: bool = True,
        auto_grant_permissions: bool = True,
    ) -> None:
        self.state = state
        self.working_dir = working_dir or Path.cwd()
        self.allow_file_operations = allow_file_operations
        self.allow_terminal = allow_terminal
        self.auto_grant_permissions = auto_grant_permissions
        self._update_event = asyncio.Event()
        self._tasks = TaskManager()

        # Terminal tracking
        self._terminals: dict[str, asyncio.subprocess.Process] = {}
        self._terminal_outputs: dict[str, str] = {}

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
        """Read text from file."""
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

            logger.debug("Read file", path=params.path, num_chars=len(content))
            return ReadTextFileResponse(content=content)

        except Exception:
            logger.exception("Failed to read file", path=params.path)
            raise

    async def write_text_file(self, params: WriteTextFileRequest) -> WriteTextFileResponse:
        """Write text to file."""
        if not self.allow_file_operations:
            msg = "File operations not allowed"
            raise RuntimeError(msg)

        try:
            path = Path(params.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(params.content, encoding="utf-8")

            logger.debug("Wrote file", path=params.path, num_chars=len(params.content))
            return WriteTextFileResponse()

        except Exception:
            logger.exception("Failed to write file", path=params.path)
            raise

    async def create_terminal(self, params: CreateTerminalRequest) -> CreateTerminalResponse:
        """Create a new terminal session."""
        if not self.allow_terminal:
            msg = "Terminal operations not allowed"
            raise RuntimeError(msg)

        try:
            terminal_id = f"term_{uuid.uuid4().hex[:8]}"
            # Build command
            cmd = [params.command, *(params.args or [])]
            # Build environment
            env = dict(os.environ)
            if params.env:
                for var in params.env:
                    env[var.name] = var.value

            # Start process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=params.cwd or str(self.working_dir),
                env=env,
            )

            self._terminals[terminal_id] = process
            self._terminal_outputs[terminal_id] = ""

            # Start output reader task
            self._tasks.create_task(
                self._read_terminal_output(terminal_id, process),
                name=f"terminal_output_{terminal_id}",
            )

            logger.info("Created terminal", terminal_id=terminal_id, command=cmd)
            return CreateTerminalResponse(terminal_id=terminal_id)

        except Exception:
            logger.exception("Failed to create terminal", command=params.command)
            raise

    async def _read_terminal_output(
        self, terminal_id: str, process: asyncio.subprocess.Process
    ) -> None:
        """Read output from terminal process."""
        if not process.stdout:
            return

        try:
            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    break
                self._terminal_outputs[terminal_id] += chunk.decode("utf-8", errors="replace")
        except Exception:
            logger.exception("Error reading terminal output", terminal_id=terminal_id)

    async def terminal_output(self, params: TerminalOutputRequest) -> TerminalOutputResponse:
        """Get output from terminal."""
        if not self.allow_terminal:
            msg = "Terminal operations not allowed"
            raise RuntimeError(msg)

        terminal_id = params.terminal_id
        if terminal_id not in self._terminals:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        output = self._terminal_outputs.get(terminal_id, "")
        return TerminalOutputResponse(output=output, truncated=False)

    async def wait_for_terminal_exit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        """Wait for terminal process to exit."""
        if not self.allow_terminal:
            msg = "Terminal operations not allowed"
            raise RuntimeError(msg)

        terminal_id = params.terminal_id
        if terminal_id not in self._terminals:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        process = self._terminals[terminal_id]
        exit_code = await process.wait()
        logger.debug("Terminal exited", terminal_id=terminal_id, exit_code=exit_code)
        return WaitForTerminalExitResponse(exit_code=exit_code)

    async def kill_terminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse | None:
        """Kill terminal process."""
        if not self.allow_terminal:
            msg = "Terminal operations not allowed"
            raise RuntimeError(msg)

        terminal_id = params.terminal_id
        if terminal_id not in self._terminals:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        process = self._terminals[terminal_id]
        process.kill()
        await process.wait()

        logger.info("Killed terminal", terminal_id=terminal_id)
        return KillTerminalCommandResponse()

    async def release_terminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse | None:
        """Release terminal resources."""
        if not self.allow_terminal:
            msg = "Terminal operations not allowed"
            raise RuntimeError(msg)

        terminal_id = params.terminal_id
        if terminal_id not in self._terminals:
            msg = f"Terminal {terminal_id} not found"
            raise ValueError(msg)

        process = self._terminals[terminal_id]
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                process.kill()
                await process.wait()

        del self._terminals[terminal_id]
        self._terminal_outputs.pop(terminal_id, None)

        logger.info("Released terminal", terminal_id=terminal_id)
        return ReleaseTerminalResponse()

    async def cleanup(self) -> None:
        """Clean up all resources."""
        # Clean up task manager first
        await self._tasks.cleanup_tasks()

        for terminal_id in list(self._terminals.keys()):
            try:
                process = self._terminals[terminal_id]
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
            except Exception:
                logger.exception("Error cleaning up terminal", terminal_id=terminal_id)

        self._terminals.clear()
        self._terminal_outputs.clear()

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
    """

    def __init__(self, config: ACPAgentConfig, **kwargs: Any) -> None:
        """Initialize ACP agent wrapper.

        Args:
            config: Configuration for the ACP agent
            **kwargs: Additional arguments passed to MessageNode
        """
        super().__init__(
            name=config.name or config.command,
            description=config.description,
            **kwargs,
        )
        self.config = config
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
        cmd = [self.config.command, *self.config.args]
        self.log.info("Starting ACP subprocess", command=cmd)

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self.config.cwd,
        )

        if not self._process.stdin or not self._process.stdout:
            msg = "Failed to create subprocess pipes"
            raise RuntimeError(msg)

    async def _initialize(self) -> None:
        """Initialize the ACP connection."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            msg = "Process not started"
            raise RuntimeError(msg)

        self._state = ACPSessionState(session_id="")
        working_dir = Path(self.config.cwd) if self.config.cwd else Path.cwd()
        self._client_handler = ACPClientHandler(
            self._state,
            working_dir=working_dir,
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
        self.log.info(
            "ACP session created",
            session_id=self._session_id,
            model=self._state.current_model_id if self._state else None,
        )

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

    async def run_iter(self, *prompts: Any, **kwargs: Any) -> AsyncIterator[ChatMessage[str]]:
        """Yield ChatMessage instances for text chunks during execution.

        Args:
            *prompts: Prompts to send (will be joined with spaces)
            **kwargs: Additional arguments (unused)

        Yields:
            ChatMessage for each text chunk from the ACP agent
        """
        from pydantic_ai import PartDeltaEvent
        from pydantic_ai.messages import TextPartDelta

        async for event in self.run_stream(*prompts, **kwargs):
            # Only yield ChatMessages for text content
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                yield ChatMessage(
                    content=event.delta.content_delta,
                    role="assistant",
                    name=self.name,
                    message_id=str(uuid.uuid4()),
                    conversation_id=self.conversation_id,
                    model_name=self._get_model_name(),
                )

    def _get_model_name(self) -> str:
        """Get model name from session state or agent info."""
        # Prefer current model from session state
        if self._state and self._state.current_model_id:
            return self._state.current_model_id
        # Fall back to agent info name
        if self._init_response and self._init_response.agent_info:
            return self._init_response.agent_info.name
        return self.config.command

    async def get_stats(self) -> MessageStats:
        """Get message statistics."""
        return MessageStats()


if __name__ == "__main__":

    async def main() -> None:
        """Demo: Basic call to an ACP agent.

        Usage:
            python -m llmling_agent.agent.acp_agent "Your prompt here"
            python -m llmling_agent.agent.acp_agent --claude "Your prompt here"
            python -m llmling_agent.agent.acp_agent --stream "Your prompt here"

        Or with default prompt:
            python -m llmling_agent.agent.acp_agent
        """
        import sys

        # Check for flags
        use_claude = "--claude" in sys.argv
        use_stream = "--stream" in sys.argv
        if use_claude:
            sys.argv.remove("--claude")
        if use_stream:
            sys.argv.remove("--stream")

        if use_claude:
            # Use claude-code-acp (must be installed: npm install -g @anthropics/claude-code-acp)
            config = ACPAgentConfig(
                command="claude-code-acp",
                args=[],
                name="claude_code",
                description="Claude Code via ACP",
                cwd=str(Path.cwd()),
                allow_file_operations=True,
                allow_terminal=True,
                auto_grant_permissions=True,
            )
        else:
            # Use llmling-agent serve-acp with openai provider
            config = ACPAgentConfig(
                command="uv",
                args=["run", "llmling-agent", "serve-acp", "--model-provider", "openai"],
                name="llmling_acp",
                description="LLMling Agent via ACP",
                cwd=str(Path.cwd()),
                allow_file_operations=True,
                allow_terminal=True,
                auto_grant_permissions=True,
            )

        print(f"Starting ACP agent: {config.command} {' '.join(config.args)}")

        try:
            async with ACPAgent(config) as agent:
                print(f"Connected to: {agent._get_model_name()}")
                print(f"Session ID: {agent._session_id}")
                print("-" * 50)
                prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Say hello briefly."
                print(f"Prompt: {prompt}")
                print("-" * 50)

                if use_stream:
                    print("Response (streaming): ", end="", flush=True)
                    async for chunk in agent.run_stream(prompt):
                        print(chunk, end="", flush=True)
                    print()  # newline after streaming
                else:
                    result = await agent.run(prompt)
                    print(f"Response: {result.content}")
                # Show tool calls if any
                if agent._state and agent._state.tool_calls:
                    print("-" * 50)
                    print("Tool calls:")
                    for tc in agent._state.tool_calls:
                        print(f"  - {tc['title']} ({tc['status']})")

        except FileNotFoundError:
            print(f"Error: Command '{config.command}' not found.")
            print("Make sure uv is installed and in PATH.")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            raise

    asyncio.run(main())
