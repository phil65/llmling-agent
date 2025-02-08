"""Handles tool execution and tracking for LLM responses."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from psygnal import Signal

from llmling_agent.agent.context import AgentContext
from llmling_agent.log import get_logger
from llmling_agent.models.tools import ToolCallInfo
from llmling_agent.tasks.exceptions import (
    ChainAbortedError,
    RunAbortedError,
    ToolSkippedError,
)
from llmling_agent.utils.inspection import execute, has_argument_type


if TYPE_CHECKING:
    from litellm import ChatCompletionMessageToolCall

    from llmling_agent.tools.base import ToolInfo

logger = get_logger(__name__)


@dataclass
class ToolCallResult:
    """Result from a tool call execution."""

    info: ToolCallInfo
    """Information about the tool call."""

    message: dict[str, Any]
    """Message to send back to the model."""


class ToolCallHandler:
    """Handles tool execution and tracking independently of provider."""

    tool_used = Signal(ToolCallInfo)
    """Emitted when a tool is used."""

    def __init__(self, agent_name: str, context: AgentContext | None = None):
        """Initialize tool call handler.

        Args:
            agent_name: Name of the agent this handler belongs to
            context: Optional agent context for confirmation handling
        """
        self.agent_name = agent_name
        self.context = context

    async def handle_tool_call(
        self,
        tool_call: ChatCompletionMessageToolCall,
        tool: ToolInfo,
        message_id: str,
    ) -> ToolCallResult:
        """Handle a single tool call execution.

        Args:
            tool_call: Tool call from LLM
            tool: Tool info containing callable
            message_id: ID of the message that triggered this

        Returns:
            Tool call result with info and message for model

        Raises:
            Various exceptions from tool execution
        """
        import json

        function_args = json.loads(tool_call.function.arguments)
        original_tool = tool.callable.callable
        start_time = perf_counter()

        try:
            # 1. Handle confirmation if we have context
            if self.context:
                result = await self.context.handle_confirmation(tool, function_args)
                match result:
                    case "skip":
                        msg = f"Tool {tool.name} execution skipped"
                        raise ToolSkippedError(msg)  # noqa: TRY301
                    case "abort_run":
                        msg = "Run aborted by user"
                        raise RunAbortedError(msg)  # noqa: TRY301
                    case "abort_chain":
                        msg = "Agent chain aborted by user"
                        raise ChainAbortedError(msg)  # noqa: TRY301
                    case "allow":
                        pass  # Continue with execution

            # 2. Add context if needed
            if has_argument_type(original_tool, AgentContext):
                enhanced_function_args = {"ctx": self.context, **function_args}
            else:
                enhanced_function_args = function_args

            # 3. Handle sync/async execution
            result = await execute(tool.execute, **enhanced_function_args)

            # Create tool call info
            info = ToolCallInfo(
                tool_name=tool.name,
                agent_name=self.agent_name,
                args=function_args,
                result=result,
                tool_call_id=tool_call.id,
                timing=perf_counter() - start_time,
                message_id=message_id,
                context_data=self.context.data if self.context else None,
            )
            self.tool_used.emit(info)

            message = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(result),
            }

        except (ToolSkippedError, RunAbortedError, ChainAbortedError) as e:
            # Handle confirmation-related errors
            info = ToolCallInfo(
                tool_name=tool.name,
                agent_name=self.agent_name,
                args=function_args,
                result=str(e),
                tool_call_id=tool_call.id,
                error=str(e),
                message_id=message_id,
                context_data=self.context.data if self.context else None,
            )
            message = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(e),
            }

        return ToolCallResult(info=info, message=message)

    async def handle_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        tools: list[ToolInfo],
        message_id: str,
    ) -> tuple[list[dict[str, Any]], list[ToolCallInfo]]:
        """Handle multiple tool calls in sequence.

        Args:
            tool_calls: List of tool calls from LLM
            tools: Available tools
            message_id: ID of message that triggered these calls

        Returns:
            Tuple of (messages for model, tool call info records)
        """
        calls: list[ToolCallInfo] = []
        new_messages = []
        pre = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        new_messages.append(pre)

        for i, tool_call in enumerate(tool_calls):
            if self.context and self.context.report_progress:
                await self.context.report_progress(i, None)

            function_name = tool_call.function.name
            if not function_name:
                continue

            tool = next(t for t in tools if t.name == function_name)
            result = await self.handle_tool_call(tool_call, tool, message_id)
            calls.append(result.info)
            new_messages.append(result.message)

        return new_messages, calls
