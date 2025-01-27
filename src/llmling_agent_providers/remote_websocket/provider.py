"""Remote WebSocket provider implementation."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field
import websockets
import websockets.client

from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo
from llmling_agent.models.messages import TokenCost
from llmling_agent.utils.tasks import TaskManagerMixin
from llmling_agent_providers.base import (
    AgentProvider,
    ProviderResponse,
    StreamingResponseProtocol,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent.common_types import ModelType
    from llmling_agent.models.content import Content


logger = get_logger(__name__)


MessageType = Literal[
    "init",  # Initial context setup
    "prompt",  # User prompt
    "tool_call",  # Server requests tool execution
    "tool_result",  # Client sends tool result
    "response",  # Final response
    "error",  # Error message
    "stream_chunk",  # Streaming chunk
    "stream_end",  # End of stream
]


class WebSocketMessage(BaseModel):
    """Base message format for WebSocket communication."""

    type: MessageType
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    content: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    ref_id: str | None = None  # For linking responses to requests


class ToolContext(BaseModel):
    """Context for tool execution."""

    name: str
    args: dict[str, Any]
    description: str | None = None


class WebSocketProvider(AgentProvider, TaskManagerMixin):
    """Provider that connects to remote agent via WebSocket."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
    ):
        super().__init__()
        self.url = url
        self.timeout = timeout
        self._ws: websockets.ClientConnection | None = None
        self._pending_responses: dict[str, asyncio.Future[Any]] = {}
        self._active_streams: dict[str, asyncio.Queue[Any]] = {}

    async def _ensure_connection(self) -> websockets.ClientConnection:
        """Ensure we have an active connection."""
        if not self._ws:
            self._ws = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10,
            )
            self.create_task(self._handle_messages())
            logger.info("Connected to remote agent at %s", self.url)
        return self._ws

    async def _handle_messages(self) -> None:
        """Handle incoming messages from WebSocket."""
        if not self._ws:
            return

        try:
            async for raw_message in self._ws:
                try:
                    message = WebSocketMessage.model_validate_json(raw_message)

                    match message.type:
                        case "response":
                            id_ = message.ref_id or ""
                            if future := self._pending_responses.get(id_):
                                future.set_result(message.content)

                        case "stream_chunk":
                            if queue := self._active_streams.get(message.ref_id or ""):
                                await queue.put(message.content)

                        case "stream_end":
                            if queue := self._active_streams.get(message.ref_id or ""):
                                await queue.put(None)

                        case "tool_call":
                            result = await self._handle_tool_call(message)
                            await self._send_message(
                                "tool_result",
                                content=result,
                                ref_id=message.message_id,
                            )

                        case "error":
                            if future := self._pending_responses.get(
                                message.ref_id or ""
                            ):
                                future.set_exception(RuntimeError(message.content))

                except Exception:
                    logger.exception("Error handling message")

        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._ws = None

    async def _send_message(
        self,
        type_: MessageType,
        content: Any,
        *,
        metadata: dict[str, Any] | None = None,
        ref_id: str | None = None,
    ) -> str:
        """Send message to WebSocket server."""
        ws = await self._ensure_connection()

        message = WebSocketMessage(
            type=type_,
            content=content,
            metadata=metadata or {},
            ref_id=ref_id,
        )

        await ws.send(message.model_dump_json())
        return message.message_id

    async def _send_context(self) -> None:
        """Send initial context to server."""
        if not self.conversation or not self.tool_manager:
            return

        # Get available tools
        tools = [tool for tool in self.tool_manager.values() if tool.enabled]

        # Convert history to dicts (since ChatMessage is a dataclass)
        history = [
            {
                "content": msg.content,
                "role": msg.role,
                "name": msg.name,
                "model": msg.model,
                "metadata": msg.metadata,
            }
            for msg in self.conversation.get_history()
        ]

        # Send context
        await self._send_message(
            type_="init",
            content={
                "conversation": history,
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "schema": t.callable.get_schema(),
                    }
                    for t in tools
                ],
            },
        )

    async def _handle_tool_call(self, message: WebSocketMessage) -> Any:
        """Execute local tool and return result."""
        try:
            tool_ctx = ToolContext.model_validate(message.content)

            if not self.tool_manager:
                msg = "No tool manager available"
                raise RuntimeError(msg)  # noqa: TRY301

            tool = self.tool_manager.get(tool_ctx.name)
            if not tool or not tool.enabled:
                msg = f"Tool not found or disabled: {tool_ctx.name}"
                raise RuntimeError(msg)  # noqa: TRY301

            # Execute tool
            result = await tool.execute(**tool_ctx.args)

            # Emit tool usage signal
            self.tool_used.emit(
                ToolCallInfo(
                    agent_name=self.name,
                    tool_name=tool_ctx.name,
                    args=tool_ctx.args,
                    result=result,
                    message_id=message.message_id,
                    tool_call_id=str(uuid4()),
                )
            )
        except Exception as e:
            logger.exception("Tool execution failed")
            return {"error": str(e)}
        else:
            return result

    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        result_type: type[Any] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate response via WebSocket connection."""
        try:
            # Send context if first time
            if not self._ws:
                await self._send_context()

            # Send prompt
            msg_id = await self._send_message(
                type_="prompt",
                content={
                    "prompts": prompts,
                    "result_type": str(result_type) if result_type else None,
                },
                metadata={"store_history": store_history},
            )

            # Create future for response
            future: asyncio.Future[Any] = asyncio.Future()
            self._pending_responses[msg_id] = future

            try:
                result = await asyncio.wait_for(future, timeout=self.timeout)
            finally:
                self._pending_responses.pop(msg_id, None)

            # Convert to ProviderResponse
            return ProviderResponse(
                content=result["content"],
                tool_calls=result.get("tool_calls", []),
                cost_and_usage=TokenCost(**result["cost_info"])
                if "cost_info" in result
                else None,
                model_name=result.get("model"),
            )

        except TimeoutError as e:
            msg = f"Response timed out after {self.timeout}s"
            raise TimeoutError(msg) from e
        except Exception as e:
            msg = f"Error generating response: {e}"
            raise RuntimeError(msg) from e

    @asynccontextmanager
    async def stream_response(
        self,
        *prompts: str | Content,
        message_id: str,
        result_type: type[Any] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingResponseProtocol]:
        """Stream response from remote agent."""
        # Send context if first time
        if not self._ws:
            await self._send_context()

        msg_id = await self._send_message(
            type_="prompt",
            content={
                "prompts": prompts,
                "stream": True,
                "result_type": str(result_type) if result_type else None,
            },
            metadata={"store_history": store_history},
        )

        # Create queue for streaming
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._active_streams[msg_id] = queue

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=self.timeout)
                    if chunk is None:  # End of stream
                        break
                    yield chunk
                except TimeoutError as e:
                    msg = f"Stream timed out after {self.timeout}s"
                    raise TimeoutError(msg) from e

        finally:
            self._active_streams.pop(msg_id, None)

    async def cleanup(self) -> None:
        """Clean up WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4

    from llmling_agent import Agent

    async def main():
        # Create provider with WebSocket URL
        provider = WebSocketProvider("ws://localhost:8000/agent")

        # Create agent with provider directly
        agent = Agent[None](
            name="remote_agent",
            provider=provider,  # provider passed directly
            system_prompt="You are a helpful assistant.",
        )

        async with agent:
            # Basic usage
            result = await agent.run("What is the capital of France?")
            print(f"Response: {result.content}")

            # With tool usage
            result = await agent.run(
                "Search for the latest news about Python programming"
            )
            print(f"\nResponse: {result.content}")

            # Show any tool calls that were made
            for tool_call in result.tool_calls:
                print(f"\nTool used: {tool_call.tool_name}")
                print(f"Arguments: {tool_call.args}")
                print(f"Result: {tool_call.result}")

    if __name__ == "__main__":
        asyncio.run(main())
