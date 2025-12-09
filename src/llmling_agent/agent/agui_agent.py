"""AG-UI remote agent implementation.

This module provides a MessageNode adapter that connects to remote AG-UI protocol servers,
enabling remote agent execution with streaming support.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from anyenv import MultiEventHandler
from anyenv.processes import hard_kill
import httpx
from pydantic import TypeAdapter

from llmling_agent.agent.agui_converters import (
    agui_to_native_event,
    convert_to_agui_content,
    extract_text_from_event,
    to_agui_input_content,
)
from llmling_agent.agent.builtin_handlers import resolve_event_handlers
from llmling_agent.agent.events import RunStartedEvent, StreamCompleteEvent
from llmling_agent.common_types import IndividualEventHandler
from llmling_agent.log import get_logger
from llmling_agent.messaging import ChatMessage, MessageHistory
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent.messaging.processing import prepare_prompts
from llmling_agent.talk.stats import MessageStats


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator, Callable, Sequence
    from types import TracebackType

    from ag_ui.core import Event
    from evented.configs import EventConfig

    from llmling_agent.agent.events import RichAgentStreamEvent
    from llmling_agent.common_types import BuiltinEventHandlerType, PromptCompatible
    from llmling_agent.delegation import AgentPool
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


def get_client(headers: dict[str, str], timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        headers={
            **headers,
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
    )


@dataclass
class AGUISessionState:
    """Track state for an active AG-UI session."""

    thread_id: str
    """Thread ID for this session."""
    run_id: str | None = None
    """Current run ID."""
    text_chunks: list[str] = field(default_factory=list)
    """Accumulated text chunks."""
    thought_chunks: list[str] = field(default_factory=list)
    """Accumulated thought chunks."""
    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Active tool calls by ID."""
    is_complete: bool = False
    """Whether the current run is complete."""
    error: str | None = None
    """Error message if run failed."""

    def clear(self) -> None:
        """Clear session state."""
        self.text_chunks.clear()
        self.thought_chunks.clear()
        self.tool_calls.clear()
        self.is_complete = False
        self.error = None
        self.run_id = str(uuid4())


class AGUIAgent[TDeps = None](MessageNode[TDeps, str]):
    """MessageNode that wraps a remote AG-UI protocol server.

    Connects to AG-UI compatible endpoints via HTTP/SSE and provides the same
    interface as native agents, enabling composition with other nodes via
    connections, teams, etc.

    The agent manages:
    - HTTP client lifecycle (create on enter, close on exit)
    - AG-UI protocol communication via SSE streams
    - Event conversion to native llmling-agent events
    - Message accumulation and final response generation

    Supports both blocking `run()` and streaming `run_stream()` execution modes.

    Example:
        ```python
        # Connect to existing server
        async with AGUIAgent(
            endpoint="http://localhost:8000/agent/run",
            name="remote-agent"
        ) as agent:
            result = await agent.run("Hello, world!")
            async for event in agent.run_stream("Tell me a story"):
                print(event)

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
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
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
            mcp_servers: MCP servers to connect
            agent_pool: Agent pool for multi-agent coordination
            enable_logging: Whether to enable database logging
            event_configs: Event trigger configurations
            event_handlers: Sequence of event handlers to register
        """
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
        # Startup command configuration
        self._startup_command = startup_command
        self._startup_delay = startup_delay
        self._startup_process: Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._state: AGUISessionState | None = None
        self._message_count = 0
        self.conversation = MessageHistory()
        self._total_tokens = 0
        self._event_queue: asyncio.Queue[RichAgentStreamEvent[Any]] = asyncio.Queue()
        resolved_handlers = resolve_event_handlers(event_handlers)
        self.event_handler = MultiEventHandler[IndividualEventHandler](resolved_handlers)

    @property
    def context(self) -> NodeContext:
        """Get node context."""
        from llmling_agent.messaging.context import NodeContext
        from llmling_agent.models.manifest import AgentsManifest
        from llmling_agent_config.nodes import NodeConfig

        cfg = NodeConfig(name=self.name, description=self.description)
        defn = self.agent_pool.manifest if self.agent_pool else AgentsManifest()
        return NodeContext(node_name=self.name, pool=self.agent_pool, config=cfg, definition=defn)

    async def __aenter__(self) -> Self:
        """Enter async context - initialize client and base resources."""
        await super().__aenter__()
        self._client = get_client(self.headers, self.timeout)
        self._state = AGUISessionState(thread_id=self.conversation_id)
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
        self._state = None
        if self._startup_process:  # Stop server if we started it
            await self._stop_server()
        self.log.debug("AG-UI client closed")
        await super().__aexit__(exc_type, exc_val, exc_tb)

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
            message_history: Optional MessageHistory to use instead of agent's own
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            ChatMessage containing the agent's aggregated text response
        """
        final_message: ChatMessage[str] | None = None
        async for event in self.run_stream(
            *prompts, message_id=message_id, message_history=message_history
        ):
            if isinstance(event, StreamCompleteEvent):
                final_message = event.message

        if final_message is None:
            raise RuntimeError("No final message received from stream")
        return final_message

    async def run_stream(
        self,
        *prompts: PromptCompatible,
        message_id: str | None = None,
        message_history: MessageHistory | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[str]]:
        """Execute prompt with streaming events.

        Args:
            prompts: Prompts to send
            message_id: Optional message ID
            message_history: Optional MessageHistory to use instead of agent's own
            **kwargs: Additional arguments (ignored for compatibility)

        Yields:
            Native streaming events converted from AG-UI protocol
        """
        from ag_ui.core import RunAgentInput, UserMessage

        if not self._client or not self._state:
            msg = "Agent not initialized - use async context manager"
            raise RuntimeError(msg)

        conversation = message_history if message_history is not None else self.conversation
        user_msg, _processed_prompts, _original_message = await prepare_prompts(*prompts)
        self._state.clear()  # Reset state
        run_started = RunStartedEvent(
            thread_id=self._state.thread_id,
            run_id=self._state.run_id or str(uuid4()),
            agent_name=self.name,
        )
        for handler in self.event_handler._wrapped_handlers:
            await handler(None, run_started)
        yield run_started

        # Get pending parts from conversation and convert them
        pending_parts = conversation.get_pending_parts()
        pending_content = to_agui_input_content(pending_parts)

        # Convert prompts to AGUI content (always returns list)
        content = await convert_to_agui_content(prompts)

        # Combine pending parts with new content
        final_content = [*pending_content, *content]

        user_message = UserMessage(id=str(uuid4()), content=final_content)
        request_data = RunAgentInput(
            thread_id=self._state.thread_id,
            run_id=self._state.run_id,
            state={},
            messages=[user_message],
            tools=[],
            context=[],
            forwarded_props={},
        )

        self.log.debug("Sending prompt to AG-UI agent")
        data = request_data.model_dump(by_alias=True)
        try:  # Send request and stream events
            async with self._client.stream("POST", self.endpoint, json=data) as response:
                response.raise_for_status()
                async for event in self._parse_sse_stream(response):
                    # Track text chunks
                    if text := extract_text_from_event(event):
                        self._state.text_chunks.append(text)

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

        except httpx.HTTPError as e:
            self._state.error = str(e)
            self.log.exception("HTTP error during AG-UI run")
            raise
        finally:
            self._state.is_complete = True
            # Emit final message
            final_message = ChatMessage[str](
                content="".join(self._state.text_chunks),
                role="assistant",
                name=self.name,
                message_id=message_id or str(uuid4()),
                conversation_id=self.conversation_id,
            )
            complete_event = StreamCompleteEvent(message=final_message)
            for handler in self.event_handler._wrapped_handlers:
                await handler(None, complete_event)
            yield complete_event
            # Record to conversation history
            conversation.add_chat_messages([user_msg, final_message])

    async def _parse_sse_stream(self, response: httpx.Response) -> AsyncIterator[Event]:
        """Parse Server-Sent Events stream.

        Args:
            response: HTTP response with SSE stream

        Yields:
            Parsed AG-UI events
        """
        from ag_ui.core import Event

        event_adapter: TypeAdapter[Event] = TypeAdapter(Event)
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            # Process complete SSE events
            while "\n\n" in buffer:
                event_text, buffer = buffer.split("\n\n", 1)
                # Parse SSE format: "data: {json}\n"
                for line in event_text.split("\n"):
                    if line.startswith("data: "):
                        json_str = line[6:]  # Remove "data: " prefix
                        try:
                            event = event_adapter.validate_json(json_str)
                            yield event
                        except (ValueError, TypeError) as e:
                            self.log.warning(
                                "Failed to parse AG-UI event",
                                json=json_str[:100],
                                error=str(e),
                            )

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

    def to_tool(self, description: str | None = None) -> Callable[[str], Any]:
        """Convert agent to a callable tool.

        Args:
            description: Tool description

        Returns:
            Async function that can be used as a tool
        """

        async def wrapped(prompt: str) -> str:
            """Execute AG-UI agent with given prompt."""
            result = await self.run(prompt)
            return result.content

        wrapped.__name__ = self.name
        wrapped.__doc__ = description or f"Call {self.name} AG-UI agent"
        return wrapped

    @property
    def model_name(self) -> str | None:
        """Get model name (AG-UI doesn't expose this)."""
        return None

    async def get_stats(self) -> MessageStats:
        """Get message statistics for this node."""
        return MessageStats()


async def main() -> None:
    """Example usage."""
    async with AGUIAgent(endpoint="http://localhost:8000/agent/run", name="test-agent") as agent:
        result = await agent.run("What is 2+2?")
        print(f"Result: {result.content}")
        print("\nStreaming:")
        async for event in agent.run_stream("Tell me a short joke"):
            print(f"Event: {event}")


if __name__ == "__main__":
    asyncio.run(main())
