"""AG-UI remote agent implementation.

This module provides a MessageNode adapter that connects to remote AG-UI protocol servers,
enabling remote agent execution with streaming support.

Supports client-side tool execution: tools can be defined locally and sent to the
remote AG-UI agent. When the agent requests tool execution, the tools are executed
locally and results sent back.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar, Self
from uuid import uuid4

from anyenv.processes import hard_kill
import anyio
import httpx
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agentpool.agents.agui_agent.helpers import execute_tool_calls, parse_sse_stream
from agentpool.agents.base_agent import BaseAgent
from agentpool.agents.events import RunStartedEvent, StreamCompleteEvent
from agentpool.agents.events.processors import FileTracker
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.tools import ToolManager
from agentpool.utils.token_breakdown import calculate_usage_from_parts


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

    from ag_ui.core import Message, ToolMessage
    from anyenv import MultiEventHandler
    from evented_config import EventConfig
    from pydantic_ai import UserContent
    from slashed import BaseCommand
    from tokonomics.model_discovery.model_info import ModelInfo

    from agentpool.agents.context import AgentContext
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory
    from agentpool.common_types import (
        BuiltinEventHandlerType,
        IndividualEventHandler,
        ToolType,
    )
    from agentpool.delegation import AgentPool
    from agentpool.hooks import AgentHooks
    from agentpool.messaging import MessageHistory
    from agentpool.models.agui_agents import AGUIAgentConfig
    from agentpool.sessions import SessionData, SessionInfo
    from agentpool.tools import Tool
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig
    from agentpool_config.nodes import ToolConfirmationMode


logger = get_logger(__name__)


def get_client(headers: dict[str, str], timeout: float) -> httpx.AsyncClient:
    headers = {**headers, "Accept": "text/event-stream", "Content-Type": "application/json"}
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout), headers=headers)


class AGUIAgent[TDeps = None](BaseAgent[TDeps, str]):
    """MessageNode that wraps a remote AG-UI protocol server.

    Connects to AG-UI compatible endpoints via HTTP/SSE and provides the same
    interface as native agents, enabling composition with other nodes via
    connections, teams, etc.

    The agent manages:
    - HTTP client lifecycle (create on enter, close on exit)
    - AG-UI protocol communication via SSE streams
    - Event conversion to native agentpool events
    - Message accumulation and final response generation
    - Client-side tool execution (tools defined locally, executed when requested)
    - Subscriber system for event hooks
    - Chunk transformation for compatibility with different server modes

    Client-Side Tools:
        Tools can be registered with this agent and sent to the remote AG-UI server.
        When the server requests a tool call, the tool is executed locally and the
        result is sent back. This enables human-in-the-loop workflows and local
        capability exposure to remote agents.
        ```
    """

    AGENT_TYPE: ClassVar = "agui"

    def __init__(
        self,
        endpoint: str,
        *,
        name: str = "agui-agent",
        description: str | None = None,
        display_name: str | None = None,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
        startup_command: str | None = None,
        startup_delay: float = 2.0,
        tools: Sequence[ToolType] | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        input_provider: InputProvider | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        tool_confirmation_mode: ToolConfirmationMode = "per_tool",
        commands: Sequence[BaseCommand] | None = None,
        hooks: AgentHooks | None = None,
    ) -> None:
        """Initialize AG-UI agent client.

        Args:
            endpoint: HTTP endpoint for the AG-UI agent
            name: Agent name for identification
            description: Agent description
            display_name: Human-readable display name
            timeout: Request timeout in seconds
            headers: Additional HTTP headers
            startup_command: Optional shell command to start server automatically.
                           Useful for testing - server lifecycle is managed by the agent.
                           Example: "ag ui agent config.yml"
            startup_delay: Seconds to wait after starting server before connecting (default: 2.0)
            tools: Tools to expose to the remote agent (executed locally when called)
            mcp_servers: MCP servers to connect
            agent_pool: Agent pool for multi-agent coordination
            enable_logging: Whether to enable database logging
            input_provider: Provider for user input/confirmations
            event_configs: Event trigger configurations
            event_handlers: Sequence of event handlers to register
            tool_confirmation_mode: Tool confirmation mode
            commands: Slash commands
            hooks: Agent hooks for pre/post tool execution
        """
        super().__init__(
            name=name,
            description=description,
            display_name=display_name,
            mcp_servers=mcp_servers,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs,
            tool_confirmation_mode=tool_confirmation_mode,
            event_handlers=event_handlers,
            input_provider=input_provider,
            commands=commands,
            hooks=hooks,
        )

        # AG-UI specific configuration
        self.endpoint = endpoint
        self.timeout = timeout
        self.headers = headers or {}
        # Startup command configuration
        self._startup_command = startup_command
        self._startup_delay = startup_delay
        self._startup_process: Process | None = None
        # Client state
        self._client: httpx.AsyncClient | None = None
        self._thread_id: str | None = None
        # Override tools with provided tools
        self.tools = ToolManager(tools)

    @classmethod
    def from_config(
        cls,
        config: AGUIAgentConfig,
        *,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
    ) -> Self:
        """Create an AGUIAgent from a config object.

        This is the preferred way to instantiate an AGUIAgent from configuration.

        Args:
            config: AG-UI agent configuration
            event_handlers: Optional event handlers (merged with config handlers)
            input_provider: Optional input provider for user interactions
            agent_pool: Optional agent pool for coordination

        Returns:
            Configured AGUIAgent instance
        """
        # Merge config-level handlers with provided handlers
        config_handlers = config.get_event_handlers()
        merged_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] = [
            *config_handlers,
            *(event_handlers or []),
        ]
        return cls(
            endpoint=config.endpoint,
            name=config.name or "agui-agent",
            description=config.description,
            display_name=config.display_name,
            event_handlers=merged_handlers or None,
            timeout=config.timeout,
            input_provider=input_provider,
            headers=config.headers,
            startup_command=config.startup_command,
            startup_delay=config.startup_delay,
            tools=[tool_config.get_tool() for tool_config in config.tools],
            mcp_servers=config.mcp_servers,
            tool_confirmation_mode=config.requires_tool_confirmation,
            agent_pool=agent_pool,
            hooks=config.hooks.get_agent_hooks() if config.hooks else None,
        )

    def get_context(
        self,
        data: Any = None,
        input_provider: InputProvider | None = None,
    ) -> AgentContext:
        """Create a new context for this agent.

        Args:
            data: Optional custom data to attach to the context
            input_provider: Optional input provider override

        Returns:
            A new AgentContext instance
        """
        from agentpool.agents.context import AgentContext
        from agentpool.models.agui_agents import AGUIAgentConfig
        from agentpool.models.manifest import AgentsManifest

        effective_provider = input_provider or self._input_provider
        cfg = AGUIAgentConfig(  # type: ignore[call-arg]
            name=self.name,
            description=self.description,
            display_name=self.display_name,
            endpoint=self.endpoint,
            timeout=self.timeout,
            headers=self.headers,
            input_provider=effective_provider,
            startup_command=self._startup_command,
            startup_delay=self._startup_delay,
        )
        defn = self.agent_pool.manifest if self.agent_pool else AgentsManifest()
        return AgentContext(
            node=self,
            pool=self.agent_pool,
            config=cfg,
            definition=defn,
            input_provider=effective_provider,
            data=data,
            model_name=self.model_name,
        )

    async def __aenter__(self) -> Self:
        """Enter async context - initialize client and base resources."""
        await super().__aenter__()
        self._client = get_client(self.headers, self.timeout)
        self._thread_id = self.conversation_id
        if self._startup_command:  # Start server if startup command is provided
            await self._start_server()
        self.log.debug("AG-UI client initialized", endpoint=self.endpoint)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context - cleanup client and base resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._thread_id = None
        if self._startup_process:  # Stop server if we started it
            await self._stop_server()
        self.log.debug("AG-UI client closed")
        await super().__aexit__(exc_type, exc_val, exc_tb)

    def register_tool(self, tool: ToolType) -> Tool:
        """Register a tool for client-side execution.

        Args:
            tool: Tool instance or callable to register

        Returns:
            Registered Tool instance
        """
        return self.tools.register_tool(tool)

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set the tool confirmation mode for this agent.

        Args:
            mode: Tool confirmation mode:
                - "always": Always require confirmation for all tools
                - "never": Never require confirmation
                - "per_tool": Use individual tool settings
        """
        self.tool_confirmation_mode = mode
        self.log.info("Tool confirmation mode changed", mode=mode)

    async def _interrupt(self) -> None:
        """Cancel the current stream task."""
        if self._current_stream_task and not self._current_stream_task.done():
            self._current_stream_task.cancel()

    async def _start_server(self) -> None:
        """Start the AG-UI server subprocess."""
        if not self._startup_command:
            return

        self.log.info("Starting AG-UI server", command=self._startup_command)
        self._startup_process = await asyncio.create_subprocess_shell(
            self._startup_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # Create new process group
        )
        self.log.debug("Waiting for server startup", delay=self._startup_delay)
        await anyio.sleep(self._startup_delay)
        # Check if process is still running
        if self._startup_process.returncode is not None:
            stderr = ""
            if self._startup_process.stderr:
                stderr = (await self._startup_process.stderr.read()).decode()
            msg = f"Startup process exited with code {self._startup_process.returncode}: {stderr}"
            raise RuntimeError(msg)

        self.log.info("AG-UI server started")

    async def _stop_server(self) -> None:
        """Stop the AG-UI server subprocess."""
        if not self._startup_process:
            return

        self.log.info("Stopping AG-UI server")
        try:
            await hard_kill(self._startup_process)  # Use cross-platform hard kill helper
        except Exception:  # Log but don't fail if kill has issues
            self.log.exception("Error during process termination")
        finally:
            self._startup_process = None
            self.log.info("AG-UI server stopped")

    async def _stream_events(  # noqa: PLR0915
        self,
        prompts: list[UserContent],
        *,
        user_msg: ChatMessage[Any],
        message_history: MessageHistory,
        effective_parent_id: str | None,
        message_id: str | None = None,
        conversation_id: str | None = None,
        parent_id: str | None = None,
        input_provider: InputProvider | None = None,
        deps: TDeps | None = None,
        event_handlers: MultiEventHandler[IndividualEventHandler],
        wait_for_connections: bool | None = None,
        store_history: bool = True,
    ) -> AsyncIterator[RichAgentStreamEvent[str]]:
        from ag_ui.core import (
            RunAgentInput,
            TextMessageChunkEvent,
            TextMessageContentEvent,
            ThinkingTextMessageContentEvent,
            ToolCallArgsEvent as AGUIToolCallArgsEvent,
            ToolCallEndEvent as AGUIToolCallEndEvent,
            ToolCallStartEvent as AGUIToolCallStartEvent,
            UserMessage,
        )

        from agentpool.agents.agui_agent.agui_converters import (
            agui_to_native_event,
            to_agui_input_content,
            to_agui_tool,
        )
        from agentpool.agents.agui_agent.chunk_transformer import ChunkTransformer
        from agentpool.agents.tool_call_accumulator import ToolCallAccumulator

        if input_provider is not None:
            self._input_provider = input_provider

        if not self._client:
            raise RuntimeError("Agent not initialized - use async context manager")

        # Set thread_id from conversation_id (needed for AG-UI protocol)
        if self._thread_id is None:
            self._thread_id = self.conversation_id

        run_id = str(uuid4())  # New run ID for each run
        chunk_transformer = ChunkTransformer()  # Create chunk transformer for this run
        # Track messages in pydantic-ai format: ModelRequest -> ModelResponse -> ModelRequest...
        # This mirrors pydantic-ai's new_messages() which includes the initial user request.
        model_messages: list[ModelResponse | ModelRequest] = []
        # Start with the user's request (same as pydantic-ai's new_messages())
        initial_request = ModelRequest(parts=[UserPromptPart(content=prompts)])
        model_messages.append(initial_request)
        response_parts: list[TextPart | ThinkingPart | ToolCallPart] = []
        assert self.conversation_id is not None  # Initialized by BaseAgent.run_stream()
        thread_id = self._thread_id or self.conversation_id
        run_started = RunStartedEvent(thread_id=thread_id, run_id=run_id, agent_name=self.name)
        await event_handlers(None, run_started)
        yield run_started
        # Get pending parts from conversation and convert them
        # Convert user message content to AGUI format using processed prompts
        final_content = to_agui_input_content(prompts)
        # Combine pending parts with new content
        user_message = UserMessage(id=str(uuid4()), content=final_content)
        # Convert registered tools to AG-UI format
        available_tools = await self.tools.get_tools(state="enabled")
        # Build initial messages list
        messages: list[Message] = [user_message]
        tool_accumulator = ToolCallAccumulator()
        pending_tool_results: list[ToolMessage] = []
        # Track files modified during this run
        file_tracker = FileTracker()
        # Loop to handle tool calls - agent may request multiple rounds
        try:
            while True:
                # Check for cancellation at start of each iteration
                if self._cancelled:
                    self.log.info("Stream cancelled by user")
                    break

                request_data = RunAgentInput(
                    thread_id=self._thread_id or self.conversation_id,
                    run_id=run_id,
                    state={},
                    messages=messages,
                    tools=[to_agui_tool(t) for t in available_tools],
                    context=[],
                    forwarded_props={},
                )

                data = request_data.model_dump(by_alias=True)
                tool_calls_pending: list[tuple[str, str, dict[str, Any]]] = []

                try:
                    async with self._client.stream("POST", self.endpoint, json=data) as response:
                        response.raise_for_status()
                        async for raw_event in parse_sse_stream(response):
                            # Check for cancellation during streaming
                            if self._cancelled:
                                self.log.info("Stream cancelled during event processing")
                                break
                            # Transform chunks to proper START/CONTENT/END sequences
                            for event in chunk_transformer.transform(raw_event):
                                # Handle events for accumulation and tool calls
                                match event:
                                    case TextMessageContentEvent(delta=delta):
                                        response_parts.append(TextPart(content=delta))
                                    case TextMessageChunkEvent(delta=delta) if delta:
                                        response_parts.append(TextPart(content=delta))
                                    case ThinkingTextMessageContentEvent(delta=delta):
                                        response_parts.append(ThinkingPart(content=delta))
                                    case AGUIToolCallStartEvent(
                                        tool_call_id=tc_id, tool_call_name=name
                                    ) if name:
                                        tool_accumulator.start(tc_id, name)
                                    case AGUIToolCallArgsEvent(tool_call_id=tc_id, delta=delta):
                                        tool_accumulator.add_args(tc_id, delta)
                                    case AGUIToolCallEndEvent(tool_call_id=tc_id):
                                        if result := tool_accumulator.complete(tc_id):
                                            tool_name, args = result
                                            tool_calls_pending.append((tc_id, tool_name, args))
                                            response_parts.append(
                                                ToolCallPart(
                                                    tool_name=tool_name,
                                                    args=args,
                                                    tool_call_id=tc_id,
                                                )
                                            )

                                # Convert to native event and distribute to handlers
                                if native_event := agui_to_native_event(event):
                                    # Track file modifications
                                    file_tracker.process_event(native_event)
                                    # Check for queued custom events first
                                    while not self._event_queue.empty():
                                        try:
                                            custom_event = self._event_queue.get_nowait()
                                            await event_handlers(None, custom_event)
                                            yield custom_event
                                        except asyncio.QueueEmpty:
                                            break
                                    # Distribute to handlers
                                    await event_handlers(None, native_event)
                                    yield native_event

                        # Flush any pending chunk events at end of stream
                        for event in chunk_transformer.flush():
                            if native_event := agui_to_native_event(event):
                                await event_handlers(None, native_event)
                                yield native_event

                except httpx.HTTPError:
                    self.log.exception("HTTP error during AG-UI run")
                    raise
                # If cancelled, break out of the while loop
                if self._cancelled:
                    break
                # If no tool calls pending, we're done
                if not tool_calls_pending:
                    break
                # Execute pending tool calls locally and collect results
                pending_tool_results = await execute_tool_calls(
                    tool_calls_pending,
                    {t.name: t for t in available_tools},
                    confirmation_mode=self.tool_confirmation_mode,
                    input_provider=self._input_provider,
                    context=self.get_context(),
                )
                # If no results (all tools were server-side), we're done
                if not pending_tool_results:
                    break

                # Flush current response parts to model_messages
                if response_parts:
                    model_messages.append(ModelResponse(parts=response_parts))
                    response_parts = []

                # Create ModelRequest with tool return parts
                tc_id_to_name = {tc_id: name for tc_id, name, _ in tool_calls_pending}
                tool_return_parts: list[ToolReturnPart] = [
                    ToolReturnPart(
                        tool_name=tc_id_to_name.get(r.tool_call_id, "unknown"),
                        content=r.content,
                        tool_call_id=r.tool_call_id,
                    )
                    for r in pending_tool_results
                ]
                model_messages.append(ModelRequest(parts=tool_return_parts))
                # Add tool results to messages for next iteration
                messages = [*pending_tool_results]
                self.log.debug("Continuing with tool results", count=len(pending_tool_results))
        except asyncio.CancelledError:
            self.log.info("Stream cancelled via task cancellation")
            self._cancelled = True

        # Handle cancellation - emit partial message
        text = "".join([i.content for i in response_parts if isinstance(i, TextPart)])
        if self._cancelled:
            # Flush any remaining response parts
            if response_parts:
                model_messages.append(ModelResponse(parts=response_parts))
            final_message = ChatMessage[str](
                content=text,
                role="assistant",
                name=self.name,
                message_id=message_id or str(uuid4()),
                conversation_id=self.conversation_id,
                parent_id=user_msg.message_id,
                messages=model_messages,
                finish_reason="stop",
                metadata=file_tracker.get_metadata(),
            )
            complete_event = StreamCompleteEvent(message=final_message)
            await event_handlers(None, complete_event)
            yield complete_event
            return

        # Flush any remaining response parts
        if response_parts:
            model_messages.append(ModelResponse(parts=response_parts))

        # Final drain of event queue after stream completes
        while not self._event_queue.empty():
            try:
                queued_event = self._event_queue.get_nowait()
                await event_handlers(None, queued_event)
                yield queued_event
            except asyncio.QueueEmpty:
                break

        # Calculate approximate token usage from what we can observe
        usage, cost_info = await calculate_usage_from_parts(
            input_parts=prompts,
            response_parts=response_parts,
            text_content=text,
            model_name=self.model_name,
        )

        final_message = ChatMessage[str](
            content=text,
            role="assistant",
            name=self.name,
            message_id=message_id or str(uuid4()),
            conversation_id=self.conversation_id,
            parent_id=user_msg.message_id,
            messages=model_messages,
            metadata=file_tracker.get_metadata(),
            usage=usage,
            cost_info=cost_info,
        )
        complete_event = StreamCompleteEvent(message=final_message)
        await event_handlers(None, complete_event)
        yield complete_event  # Post-processing handled by base class

    @property
    def model_name(self) -> str | None:
        """Get model name (AG-UI doesn't expose this)."""
        return None

    async def set_model(self, model: str) -> None:
        """Set model (no-op for AG-UI as model is controlled by remote server)."""
        # AG-UI agents don't support model selection - the model is
        # determined by the remote server configuration

    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models for AG-UI agent.

        AG-UI doesn't expose model information, so returns a placeholder model
        indicating the model is determined by the remote server.

        Returns:
            List with a single placeholder ModelInfo
        """
        from tokonomics.model_discovery.model_info import ModelInfo

        return [
            ModelInfo(
                id="server-determined",
                name="Determined by server",
                description="The model is determined by the remote AG-UI server",
            )
        ]

    async def get_modes(self) -> list[ModeCategory]:
        """Get available modes for AG-UI agent (not supported)."""
        return []

    async def _set_mode(self, mode_id: str, category_id: str) -> None:
        """AG-UI doesn't support mode switching."""
        msg = "AG-UI agent does not support mode switching - model is controlled by remote server"
        raise ValueError(msg)

    async def list_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[SessionInfo]:
        """List sessions for AG-UI agent (not supported)."""
        return []

    async def load_session(self, session_id: str) -> SessionData | None:
        """Load a session for AG-UI agent. (not supported)."""
        self.log.warning("AG-UI agents do not support session loading", session_id=session_id)
        return None


if __name__ == "__main__":

    async def main() -> None:
        """Example usage."""
        endpoint = "http://localhost:8000/agent/run"
        async with AGUIAgent(endpoint=endpoint, name="test-agent") as agent:
            result = await agent.run("What is 2+2?")
            print(f"Result: {result.content}")
            print("\nStreaming:")
            async for event in agent.run_stream("Tell me a short joke"):
                print(f"Event: {event}")

    anyio.run(main)
