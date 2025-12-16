"""Mock input provider implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from llmling_agent.ui.base import InputProvider


if TYPE_CHECKING:
    from mcp import types

    from llmling_agent.agents.context import ConfirmationResult
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.tools.base import Tool

InputMethod = Literal["get_input", "get_tool_confirmation", "get_elicitation"]


@dataclass
class InputCall:
    """Record of an input provider call."""

    method: InputMethod
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Any


class MockInputProvider(InputProvider):
    """Provider that records calls and returns pre-configured responses."""

    def __init__(
        self,
        *,
        input_response: str = "mock response",
        tool_confirmation: ConfirmationResult = "allow",
        elicitation_response: dict[str, Any] | None = None,
    ) -> None:
        self.input_response = input_response
        self.tool_confirmation = tool_confirmation
        self.elicitation_response = elicitation_response or {"response": "mock response"}
        self.calls: list[InputCall] = []

    async def get_input(
        self,
        context: NodeContext,
        prompt: str,
        output_type: type | None = None,
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> Any:
        kwargs = {"output_type": output_type, "message_history": message_history}
        call = InputCall("get_input", (context, prompt), kwargs, result=self.input_response)
        self.calls.append(call)
        return self.input_response

    async def get_tool_confirmation(
        self,
        context: NodeContext,
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> ConfirmationResult:
        kwargs = {"message_history": message_history}
        result = self.tool_confirmation
        call = InputCall("get_tool_confirmation", (context, tool, args), kwargs, result=result)
        self.calls.append(call)
        return result  # pyright: ignore

    async def get_elicitation(
        self,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        from mcp import types

        result = types.ElicitResult(action="accept", content=self.elicitation_response)
        call = InputCall("get_elicitation", (params,), {}, result=result)
        self.calls.append(call)
        return result
