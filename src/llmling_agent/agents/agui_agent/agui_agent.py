"""AG-UI remote agent implementation.

This module provides a MessageNode adapter that connects to remote AG-UI protocol servers,
enabling remote agent execution with streaming support.

Supports client-side tool execution: tools can be defined locally and sent to the
remote AG-UI agent. When the agent requests tool execution, the tools are executed
locally and results sent back.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from anyenv import MultiEventHandler
from anyenv.processes import hard_kill
import httpx
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from llmling_agent.agents.agui_agent.chunk_transformer import ChunkTransformer
from llmling_agent.agents.agui_agent.helpers import execute_tool_calls, parse_sse_stream
from llmling_agent.agents.base_agent import BaseAgent
from llmling_agent.agents.events import RunStartedEvent, StreamCompleteEvent, resolve_event_handlers
from llmling_agent.common_types import IndividualEventHandler
from llmling_agent.log import get_logger
from llmling_agent.messaging import ChatMessage, MessageHistory
from llmling_agent.messaging.processing import prepare_prompts
from llmling_agent.talk.stats import MessageStats
from llmling_agent.tools import ToolManager


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

    from ag_ui.core import Message, ToolMessage
    from evented.configs import EventConfig

    from llmling_agent.agents.base_agent import ToolConfirmationMode
    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.common_types import BuiltinEventHandlerType, PromptCompatible, ToolType
    from llmling_agent.delegation import AgentPool
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.tools import Tool
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.mcp_server import MCPServerConfig


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
    - Event conversion to native llmling-agent events
    - Message accumulation and final response generation
    - Client-side tool execution (tools defined locally, executed when requested)
    - Subscriber system for event hooks
    - Chunk transformation for compatibility with different server modes

    Client-Side Tools:
        Tools can be registered with this agent and sent to the remote AG-UI server.
        When the server requests a tool call, the tool is executed locally and the
        result is sent back. This enables human-in-the-loop workflows and local
        capability exposure to remote agents.

    Example:
        ```python
        # Connect to existing server
        async with AGUIAgent(
            endpoint="http://localhost:8000/agent/run",
            name="tool-agent",
            tools=[my_tool_function],
        ) as agent:
            # Remote agent can request execution of my_tool_function
            result = await agent.run("Use the tool to help me")

        # Start server automatically (useful for testing)
        async with AGUIAgent(
            endpoint="http://localhost:8000/agent/run",
            name="test-agent",
            startup_command="ag ui agent config.yml",
            startup_delay=2.0,
        ) as agent:
            result = await agent.run("Test prompt")
        ```
    """

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
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        debug: bool = False,
        tool_confirmation_mode: ToolConfirmationMode = "per_tool",
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
            event_configs: Event trigger configurations
            event_handlers: Sequence of event handlers to register
            debug: Enable debug logging for chunk transformer
            tool_confirmation_mode: Tool confirmation mode
        """
        from exxec import LocalExecutionEnvironment

        super().__init__(
            name=name,
            description=description,
            display_name=display_name,
            mcp_servers=mcp_servers,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs,
        )
        self.endpoint = endpoint
        self.timeout = timeout
        self.headers = headers or {}
        self._debug = debug
        # Startup command configuration
        self._startup_command = startup_command
        self._startup_delay = startup_delay
        self._startup_process: Process | None = None
        self._client: httpx.AsyncClient | None = None
        # Session tracking (inline, no separate state object for uniformity)
        self._thread_id: str | None = None
        self._run_id: str | None = None
        self._output_type = str
        self.conversation = MessageHistory()
        self._event_queue: asyncio.Queue[RichAgentStreamEvent[Any]] = asyncio.Queue()
        resolved_handlers = resolve_event_handlers(event_handlers)
        self.event_handler = MultiEventHandler[IndividualEventHandler](resolved_handlers)
        self.tools = ToolManager(tools)
        # Tool confirmation mode - defaults to "per_tool" like Agent class
        self.tool_confirmation_mode: ToolConfirmationMode = tool_confirmation_mode
        self._input_provider: InputProvider | None = None
        self.env = LocalExecutionEnvironment()
        # Chunk transformer for normalizing CHUNK events
        self._chunk_transformer = ChunkTransformer(debug=debug)

    @property
    def context(self) -> NodeContext:
        """Get node context."""
        from llmling_agent.messaging.context import NodeContext
        from llmling_agent.models.manifest import AgentsManifest
        from llmling_agent_config.nodes import NodeConfig

        cfg = NodeConfig(name=self.name, description=self.description)
        defn = self.agent_pool.manifest if self.agent_pool else AgentsManifest()
        return NodeContext(node=self, pool=self.agent_pool, config=cfg, definition=defn)

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
        self._run_id = None
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
        await asyncio.sleep(self._startup_delay)
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

    async def run(
        self,
        *prompts: PromptCompatible,
        message_id: str | None = None,
        input_provider: InputProvider | None = None,
        message_history: MessageHistory | None = None,
        **kwargs: Any,
    ) -> ChatMessage[str]:
        """Execute prompt against AG-UI agent.

        Sends the prompt to the AG-UI server and waits for completion.
        Events are collected via run_stream and event handlers are called.
        The final text content is returned as a ChatMessage.

        Args:
            prompts: Prompts to send (will be joined with spaces)
            message_id: Optional message id for the returned message
            input_provider: Optional input provider for tool confirmation requests
            message_history: Optional MessageHistory to use instead of agent's own
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            ChatMessage containing the agent's aggregated text response
        """
        final_message: ChatMessage[str] | None = None
        async for event in self.run_stream(
            *prompts,
            message_id=message_id,
            input_provider=input_provider,
            message_history=message_history,
        ):
            if isinstance(event, StreamCompleteEvent):
                final_message = event.message

        if final_message is None:
            raise RuntimeError("No final message received from stream")
        return final_message

    async def run_stream(  # noqa: PLR0915
        self,
        *prompts: PromptCompatible,
        message_id: str | None = None,
        input_provider: InputProvider | None = None,
        message_history: MessageHistory | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[str]]:
        """Execute prompt with streaming events.

        Sends the prompt to the remote AG-UI agent along with any registered tools.
        When the agent requests a tool call, the tool is executed locally and the
        result is sent back in a continuation request.

        Args:
            prompts: Prompts to send
            message_id: Optional message ID
            input_provider: Optional input provider for tool confirmation requests
            message_history: Optional MessageHistory to use instead of agent's own
            **kwargs: Additional arguments (ignored for compatibility)

        Yields:
            Native streaming events converted from AG-UI protocol
        """
        # Update input provider if provided
        if input_provider is not None:
            self._input_provider = input_provider
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

        from llmling_agent.agents.agui_agent.agui_converters import (
            ToolCallAccumulator,
            agui_to_native_event,
            to_agui_input_content,
            to_agui_tool,
        )

        if not self._client or not self._thread_id:
            msg = "Agent not initialized - use async context manager"
            raise RuntimeError(msg)

        conversation = message_history if message_history is not None else self.conversation
        user_msg, processed_prompts, _original_message = await prepare_prompts(*prompts)
        self._run_id = str(uuid4())  # New run ID for each run
        self._chunk_transformer.reset()  # Reset chunk transformer
        # Track messages in pydantic-ai format: ModelRequest -> ModelResponse -> ModelRequest...
        # This mirrors pydantic-ai's new_messages() which includes the initial user request.
        model_messages: list[ModelResponse | ModelRequest] = []
        # Start with the user's request (same as pydantic-ai's new_messages())
        initial_request = ModelRequest(parts=[UserPromptPart(content=processed_prompts)])
        model_messages.append(initial_request)
        current_response_parts: list[TextPart | ThinkingPart | ToolCallPart] = []
        text_chunks: list[str] = []  # For final content string

        run_started = RunStartedEvent(
            thread_id=self._thread_id or self.conversation_id,
            run_id=self._run_id or str(uuid4()),
            agent_name=self.name,
        )
        for handler in self.event_handler._wrapped_handlers:
            await handler(None, run_started)
        yield run_started

        # Get pending parts from conversation and convert them
        pending_parts = conversation.get_pending_parts()
        pending_content = to_agui_input_content(pending_parts)
        # Convert user message content to AGUI format using processed prompts
        user_content = to_agui_input_content(processed_prompts)
        # Combine pending parts with new content
        final_content = [*pending_content, *user_content]
        user_message = UserMessage(id=str(uuid4()), content=final_content)
        # Convert registered tools to AG-UI format
        available_tools = await self.tools.get_tools(state="enabled")
        agui_tools = [to_agui_tool(t) for t in available_tools]
        tools_by_name = {t.name: t for t in available_tools}
        # Build initial messages list
        messages: list[Message] = [user_message]
        tool_accumulator = ToolCallAccumulator()
        pending_tool_results: list[ToolMessage] = []
        self.log.debug("Sending prompt to AG-UI agent", tool_names=[t.name for t in agui_tools])
        # Loop to handle tool calls - agent may request multiple rounds
        while True:
            request_data = RunAgentInput(
                thread_id=self._thread_id or self.conversation_id,
                run_id=self._run_id,
                state={},
                messages=messages,
                tools=agui_tools,
                context=[],
                forwarded_props={},
            )

            data = request_data.model_dump(by_alias=True)
            tool_calls_pending: list[tuple[str, str, dict[str, Any]]] = []

            try:
                async with self._client.stream("POST", self.endpoint, json=data) as response:
                    response.raise_for_status()
                    async for raw_event in parse_sse_stream(response):
                        # Transform chunks to proper START/CONTENT/END sequences
                        transformed_events = self._chunk_transformer.transform(raw_event)

                        for event in transformed_events:
                            # Handle events for accumulation and tool calls
                            match event:
                                case TextMessageContentEvent(delta=delta):
                                    text_chunks.append(delta)
                                    current_response_parts.append(TextPart(content=delta))
                                case TextMessageChunkEvent(delta=delta) if delta:
                                    text_chunks.append(delta)
                                    current_response_parts.append(TextPart(content=delta))
                                case ThinkingTextMessageContentEvent(delta=delta):
                                    current_response_parts.append(ThinkingPart(content=delta))
                                case AGUIToolCallStartEvent(
                                    tool_call_id=tc_id, tool_call_name=name
                                ) if name:
                                    tool_accumulator.start(tc_id, name)
                                    self.log.debug(
                                        "Tool call started",
                                        tool_call_id=tc_id,
                                        tool=name,
                                    )

                                case AGUIToolCallArgsEvent(tool_call_id=tc_id, delta=delta):
                                    tool_accumulator.add_args(tc_id, delta)

                                case AGUIToolCallEndEvent(tool_call_id=tc_id):
                                    if result := tool_accumulator.complete(tc_id):
                                        tool_name, args = result
                                        tool_calls_pending.append((tc_id, tool_name, args))
                                        current_response_parts.append(
                                            ToolCallPart(
                                                tool_name=tool_name,
                                                args=args,
                                                tool_call_id=tc_id,
                                            )
                                        )
                                        self.log.debug(
                                            "Tool call completed",
                                            tool_call_id=tc_id,
                                            tool=tool_name,
                                            args=args,
                                        )

                            # Convert to native event and distribute to handlers
                            if native_event := agui_to_native_event(event):
                                # Check for queued custom events first
                                while not self._event_queue.empty():
                                    try:
                                        custom_event = self._event_queue.get_nowait()
                                        for handler in self.event_handler._wrapped_handlers:
                                            await handler(None, custom_event)
                                        yield custom_event
                                    except asyncio.QueueEmpty:
                                        break
                                # Distribute to handlers
                                for handler in self.event_handler._wrapped_handlers:
                                    await handler(None, native_event)
                                yield native_event

                    # Flush any pending chunk events at end of stream
                    for event in self._chunk_transformer.flush():
                        if native_event := agui_to_native_event(event):
                            for handler in self.event_handler._wrapped_handlers:
                                await handler(None, native_event)
                            yield native_event

            except httpx.HTTPError:
                self.log.exception("HTTP error during AG-UI run")
                raise

            # If no tool calls pending, we're done
            if not tool_calls_pending:
                break

            # Execute pending tool calls locally and collect results
            pending_tool_results = await execute_tool_calls(
                tool_calls_pending,
                tools_by_name,
                confirmation_mode=self.tool_confirmation_mode,
                input_provider=self._input_provider,
                context=self.context,
            )
            # If no results (all tools were server-side), we're done
            if not pending_tool_results:
                break

            # Flush current response parts to model_messages
            if current_response_parts:
                model_messages.append(ModelResponse(parts=current_response_parts))
                current_response_parts = []

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

        # Flush any remaining response parts
        if current_response_parts:
            model_messages.append(ModelResponse(parts=current_response_parts))

        text_content = "".join(text_chunks)
        final_message = ChatMessage[str](
            content=text_content,
            role="assistant",
            name=self.name,
            message_id=message_id or str(uuid4()),
            conversation_id=self.conversation_id,
            messages=model_messages,
        )
        complete_event = StreamCompleteEvent(message=final_message)
        for handler in self.event_handler._wrapped_handlers:
            await handler(None, complete_event)
        yield complete_event
        # Record to conversation history
        conversation.add_chat_messages([user_msg, final_message])

    async def run_iter(
        self,
        *prompt_groups: Sequence[PromptCompatible],
        message_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[str]]:
        """Execute multiple prompt groups sequentially.

        Args:
            prompt_groups: Groups of prompts to execute
            message_id: Optional message ID base
            **kwargs: Additional arguments (ignored for compatibility)

        Yields:
            ChatMessage for each completed prompt group
        """
        for i, prompts in enumerate(prompt_groups):
            mid = f"{message_id or 'msg'}_{i}" if message_id else None
            yield await self.run(*prompts, message_id=mid)

    @property
    def model_name(self) -> str | None:
        """Get model name (AG-UI doesn't expose this)."""
        return None

    async def get_stats(self) -> MessageStats:
        """Get message statistics for this node."""
        return MessageStats()


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

    asyncio.run(main())
