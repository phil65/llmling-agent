"""ClaudeCodeAgent - Native Claude Agent SDK integration.

This module provides an agent implementation that wraps the Claude Agent SDK's
ClaudeSDKClient for native integration with llmling-agent.

The ClaudeCodeAgent acts as a client to the Claude Code CLI, enabling:
- Bidirectional streaming communication
- Tool permission handling via callbacks
- Integration with llmling-agent's event system

Example:
    ```python
    async with ClaudeCodeAgent(
        name="claude_coder",
        cwd="/path/to/project",
        allowed_tools=["Read", "Write", "Bash"],
    ) as agent:
        async for event in agent.run_stream("Write a hello world program"):
            print(event)
    ```
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Self
import uuid

from anyenv import MultiEventHandler
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage

from llmling_agent.agents.base_agent import BaseAgent
from llmling_agent.agents.claude_code_agent.converters import claude_message_to_events
from llmling_agent.agents.events import RunErrorEvent, RunStartedEvent, StreamCompleteEvent
from llmling_agent.common_types import IndividualEventHandler
from llmling_agent.log import get_logger
from llmling_agent.messaging import ChatMessage, MessageHistory
from llmling_agent.messaging.messages import TokenCost
from llmling_agent.messaging.processing import prepare_prompts


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        PermissionResult,
        ToolPermissionContext,
        ToolUseBlock,
    )
    from evented.configs import EventConfig
    from exxec import ExecutionEnvironment

    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.common_types import BuiltinEventHandlerType, PromptCompatible
    from llmling_agent.delegation import AgentPool
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.talk.stats import MessageStats
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.nodes import ToolConfirmationMode


logger = get_logger(__name__)


class ClaudeCodeAgent[TDeps = None, TResult = str](BaseAgent[TDeps, TResult]):
    """Agent wrapping Claude Agent SDK's ClaudeSDKClient.

    This provides native integration with Claude Code, enabling:
    - Bidirectional streaming for interactive conversations
    - Tool permission handling via can_use_tool callback
    - Full access to Claude Code's capabilities (file ops, terminals, etc.)

    The agent manages:
    - ClaudeSDKClient lifecycle (connect on enter, disconnect on exit)
    - Event conversion from Claude SDK to llmling-agent events
    - Tool confirmation via input provider
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        cwd: str | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        max_thinking_tokens: int | None = None,
        permission_mode: str | None = None,
        env: ExecutionEnvironment | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        tool_confirmation_mode: ToolConfirmationMode = "always",
        output_type: type[TResult] | None = None,
    ) -> None:
        """Initialize ClaudeCodeAgent.

        Args:
            name: Agent name
            description: Agent description
            display_name: Display name for UI
            cwd: Working directory for Claude Code
            allowed_tools: List of allowed tool names
            disallowed_tools: List of disallowed tool names
            system_prompt: Custom system prompt
            model: Model to use (e.g., "claude-sonnet-4-5")
            max_turns: Maximum conversation turns
            max_thinking_tokens: Max tokens for extended thinking
            permission_mode: Permission mode ("default", "acceptEdits", "plan", "bypassPermissions")
            env: Execution environment
            input_provider: Provider for user input/confirmations
            agent_pool: Agent pool for multi-agent coordination
            enable_logging: Whether to enable logging
            event_configs: Event configuration
            event_handlers: Event handlers for streaming events
            tool_confirmation_mode: Tool confirmation behavior
            output_type: Type for structured output (uses JSON schema)
        """
        from exxec import LocalExecutionEnvironment

        from llmling_agent.agents.events import resolve_event_handlers
        from llmling_agent.tools.manager import ToolManager

        super().__init__(
            name=name or "claude_code",
            description=description,
            display_name=display_name,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs or [],
        )

        # Store configuration
        self._cwd = cwd
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools
        self._system_prompt = system_prompt
        self._model = model
        self._max_turns = max_turns
        self._max_thinking_tokens = max_thinking_tokens
        self._permission_mode = permission_mode
        # Client state
        self._client: ClaudeSDKClient | None = None
        self._current_model: str | None = model
        # Infrastructure
        self.env = env or LocalExecutionEnvironment()
        self._input_provider = input_provider
        self._output_type: type[TResult] = output_type or str  # type: ignore[assignment]
        self.conversation = MessageHistory()
        self.deps_type = type(None)
        self._event_queue: asyncio.Queue[RichAgentStreamEvent[Any]] = asyncio.Queue()
        resolved_handlers = resolve_event_handlers(event_handlers)
        self.event_handler = MultiEventHandler[IndividualEventHandler](resolved_handlers)
        self.tools = ToolManager()
        self.tool_confirmation_mode: ToolConfirmationMode = tool_confirmation_mode

    @property
    def context(self) -> NodeContext:
        """Get node context."""
        from llmling_agent.messaging.context import NodeContext
        from llmling_agent.models.manifest import AgentsManifest
        from llmling_agent_config.nodes import NodeConfig

        cfg = NodeConfig(name=self.name, description=self.description)
        defn = self.agent_pool.manifest if self.agent_pool else AgentsManifest()
        return NodeContext(node=self, pool=self.agent_pool, config=cfg, definition=defn)

    @property
    def model_name(self) -> str | None:
        """Get the model name."""
        return self._current_model

    def _build_options(self) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from configuration."""
        from claude_agent_sdk import ClaudeAgentOptions

        options_kwargs: dict[str, Any] = {}

        if self._cwd:
            options_kwargs["cwd"] = self._cwd
        if self._allowed_tools:
            options_kwargs["allowed_tools"] = self._allowed_tools
        if self._disallowed_tools:
            options_kwargs["disallowed_tools"] = self._disallowed_tools
        if self._system_prompt:
            options_kwargs["system_prompt"] = self._system_prompt
        if self._model:
            options_kwargs["model"] = self._model
        if self._max_turns:
            options_kwargs["max_turns"] = self._max_turns
        if self._max_thinking_tokens:
            options_kwargs["max_thinking_tokens"] = self._max_thinking_tokens
        if self._permission_mode:
            options_kwargs["permission_mode"] = self._permission_mode

        # Add tool permission callback if not in bypass mode
        if self.tool_confirmation_mode != "never" and self._permission_mode != "bypassPermissions":
            options_kwargs["can_use_tool"] = self._can_use_tool

        # Add structured output schema if output_type is not str
        if self._output_type is not str:
            from pydantic import TypeAdapter

            adapter = TypeAdapter(self._output_type)
            schema = adapter.json_schema()
            options_kwargs["output_format"] = {"type": "json_schema", "schema": schema}

        return ClaudeAgentOptions(**options_kwargs)

    async def _can_use_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResult:
        """Handle tool permission requests.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input arguments
            context: Permission context with suggestions

        Returns:
            PermissionResult indicating allow or deny
        """
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        # Auto-grant if confirmation mode is "never"
        if self.tool_confirmation_mode == "never":
            return PermissionResultAllow()

        # Use input provider if available
        if self._input_provider:
            from llmling_agent.tools.base import Tool

            # Create a dummy Tool for the confirmation dialog
            tool = Tool(
                callable=lambda: None,
                name=tool_name,
                description=f"Claude Code tool: {tool_name}",
            )

            result = await self._input_provider.get_tool_confirmation(
                context=self.context,
                tool=tool,
                args=input_data,
            )

            match result:
                case "allow":
                    return PermissionResultAllow()
                case "skip":
                    return PermissionResultDeny(message="User skipped tool execution")
                case "abort_run" | "abort_chain":
                    return PermissionResultDeny(message="User aborted execution", interrupt=True)
                case _:
                    return PermissionResultDeny(message="Unknown confirmation result")

        # Default: deny if no input provider
        return PermissionResultDeny(message="No input provider configured")

    async def __aenter__(self) -> Self:
        """Connect to Claude Code."""
        await super().__aenter__()

        from claude_agent_sdk import ClaudeSDKClient

        options = self._build_options()
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self.log.info("Claude Code client connected")

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Disconnect from Claude Code."""
        if self._client:
            try:
                await self._client.disconnect()
                self.log.info("Claude Code client disconnected")
            except Exception:
                self.log.exception("Error disconnecting Claude Code client")
            self._client = None

        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def run(
        self,
        *prompts: PromptCompatible,
        message_id: str | None = None,
        input_provider: InputProvider | None = None,
        message_history: MessageHistory | None = None,
    ) -> ChatMessage[TResult]:
        """Execute prompt against Claude Code.

        Args:
            prompts: Prompts to send
            message_id: Optional message ID for the returned message
            input_provider: Optional input provider for permission requests
            message_history: Optional MessageHistory to use instead of agent's own

        Returns:
            ChatMessage containing the agent's response
        """
        final_message: ChatMessage[TResult] | None = None
        async for event in self.run_stream(
            *prompts,
            message_id=message_id,
            input_provider=input_provider,
            message_history=message_history,
        ):
            if isinstance(event, StreamCompleteEvent):
                final_message = event.message

        if final_message is None:
            msg = "No final message received from stream"
            raise RuntimeError(msg)

        return final_message

    async def run_stream(  # noqa: PLR0915
        self,
        *prompts: PromptCompatible,
        message_id: str | None = None,
        input_provider: InputProvider | None = None,
        message_history: MessageHistory | None = None,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Stream events from Claude Code execution.

        Args:
            prompts: Prompts to send
            message_id: Optional message ID for the final message
            input_provider: Optional input provider for permission requests
            message_history: Optional MessageHistory to use instead of agent's own

        Yields:
            RichAgentStreamEvent instances during execution
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock as ToolUseBlockType,
        )

        # Update input provider if provided
        if input_provider is not None:
            self._input_provider = input_provider

        if not self._client:
            msg = "Agent not initialized - use async context manager"
            raise RuntimeError(msg)

        conversation = message_history if message_history is not None else self.conversation
        # Prepare prompts
        user_msg, processed_prompts, _original_message = await prepare_prompts(*prompts)
        # Join prompts into single string for Claude SDK
        prompt_text = " ".join(str(p) for p in processed_prompts)
        run_id = str(uuid.uuid4())
        # Emit run started
        run_started = RunStartedEvent(
            thread_id=self.conversation_id,
            run_id=run_id,
            agent_name=self.name,
        )
        for handler in self.event_handler._wrapped_handlers:
            await handler(None, run_started)
        yield run_started

        model_messages: list[ModelResponse | ModelRequest] = [
            ModelRequest(parts=[UserPromptPart(content=prompt_text)])
        ]
        current_response_parts: list[TextPart | ThinkingPart | ToolCallPart] = []
        text_chunks: list[str] = []
        pending_tool_calls: dict[str, ToolUseBlock] = {}

        try:
            await self._client.query(prompt_text)
            async for message in self._client.receive_response():
                # Process assistant messages - extract parts incrementally
                if isinstance(message, AssistantMessage):
                    # Update model name from first assistant message
                    if message.model:
                        self._current_model = message.model
                    for block in message.content:
                        match block:
                            case TextBlock(text=text):
                                text_chunks.append(text)
                                current_response_parts.append(TextPart(content=text))
                            case ThinkingBlock(thinking=thinking):
                                current_response_parts.append(ThinkingPart(content=thinking))
                            case ToolUseBlockType(id=tc_id, name=name, input=input_data):
                                pending_tool_calls[tc_id] = block
                                current_response_parts.append(
                                    ToolCallPart(
                                        tool_name=name, args=input_data, tool_call_id=tc_id
                                    )
                                )
                            case ToolResultBlock(tool_use_id=tc_id, content=content):
                                # Tool result received - flush response parts and add request
                                if current_response_parts:
                                    model_messages.append(
                                        ModelResponse(parts=current_response_parts)
                                    )
                                    current_response_parts = []

                                # Get tool name from pending calls
                                tool_use = pending_tool_calls.pop(tc_id, None)
                                tool_name = tool_use.name if tool_use else "unknown"
                                # Add tool return as ModelRequest
                                part = ToolReturnPart(
                                    tool_name=tool_name, content=content, tool_call_id=tc_id
                                )
                                model_messages.append(ModelRequest(parts=[part]))

                # Convert to events and yield
                events = claude_message_to_events(
                    message,
                    agent_name=self.name,
                    pending_tool_calls={},  # Already handled above
                )
                for event in events:
                    for handler in self.event_handler._wrapped_handlers:
                        await handler(None, event)
                    yield event

                # Check for result (end of response) and capture usage info
                if isinstance(message, ResultMessage):
                    result_message = message
                    break
            else:
                result_message = None

        except Exception as e:
            error_event = RunErrorEvent(message=str(e), run_id=run_id, agent_name=self.name)
            for handler in self.event_handler._wrapped_handlers:
                await handler(None, error_event)
            yield error_event
            raise

        # Flush any remaining response parts
        if current_response_parts:
            model_messages.append(ModelResponse(parts=current_response_parts))

        text_content = "".join(text_chunks)

        # Determine final content - use structured output if available
        final_content: TResult
        if self._output_type is not str and result_message and result_message.structured_output:
            final_content = result_message.structured_output
        else:
            final_content = text_content  # type: ignore[assignment]

        # Build cost_info from ResultMessage if available
        cost_info = None
        if result_message and result_message.usage:
            usage = result_message.usage
            run_usage = RunUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
            )
            cost_info = TokenCost(
                token_usage=run_usage,
                total_cost=Decimal(str(result_message.total_cost_usd or 0)),
            )

        chat_message = ChatMessage[TResult](
            content=final_content,
            role="assistant",
            name=self.name,
            message_id=message_id or str(uuid.uuid4()),
            conversation_id=self.conversation_id,
            model_name=self.model_name,
            messages=model_messages,
            cost_info=cost_info,
            response_time=result_message.duration_ms / 1000 if result_message else None,
        )

        # Emit stream complete
        complete_event: StreamCompleteEvent[TResult] = StreamCompleteEvent(message=chat_message)
        for handler in self.event_handler._wrapped_handlers:
            await handler(None, complete_event)
        yield complete_event

        # Record to history
        self.message_sent.emit(chat_message)
        conversation.add_chat_messages([user_msg, chat_message])

    async def run_iter(
        self,
        *prompt_groups: Sequence[PromptCompatible],
    ) -> AsyncIterator[ChatMessage[TResult]]:
        """Run agent sequentially on multiple prompt groups.

        Args:
            prompt_groups: Groups of prompts to process sequentially

        Yields:
            Response messages in sequence
        """
        for prompts in prompt_groups:
            response = await self.run(*prompts)
            yield response

    async def set_model(self, model: str) -> None:
        """Set the model for future requests.

        Note: This updates the model for the next query. The client
        maintains the connection, so this takes effect on the next query().

        Args:
            model: Model name to use
        """
        self._model = model
        self._current_model = model

        if self._client:
            await self._client.set_model(model)
            self.log.info("Model changed", model=model)

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set tool confirmation mode.

        Args:
            mode: Confirmation mode - "always", "never", or "per_tool"
        """
        self.tool_confirmation_mode = mode
        # Update permission mode on client if connected
        if self._client and mode == "never":
            await self._client.set_permission_mode("bypassPermissions")
        elif self._client and mode == "always":
            await self._client.set_permission_mode("default")

    async def get_stats(self) -> MessageStats:
        """Get message statistics."""
        from llmling_agent.talk.stats import MessageStats

        return MessageStats(messages=list(self.conversation.chat_messages))


if __name__ == "__main__":
    import os

    os.environ["ANTHROPIC_API_KEY"] = ""

    async def main() -> None:
        """Demo: Basic call to Claude Code."""
        async with ClaudeCodeAgent(
            name="demo",
            allowed_tools=["Read"],
            event_handlers=["detailed"],
        ) as agent:
            print("Response (streaming): ", end="", flush=True)
            async for event in agent.run_stream("What files are in the current directory?"):
                print(event, end="", flush=True)
            print()

    asyncio.run(main())
