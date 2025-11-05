"""Callback input provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.utils.inspection import execute
from llmling_agent_input.base import InputProvider


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp import types

    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.tools.base import Tool


class CallbackInputProvider(InputProvider):
    """Input provider that delegates to provided callbacks."""

    def __init__(
        self,
        get_input: Callable[..., str | Awaitable[str]] | None = None,
        get_tool_confirmation: Callable[
            ..., ConfirmationResult | Awaitable[ConfirmationResult]
        ]
        | None = None,
        get_code_input: Callable[..., str | Awaitable[str]] | None = None,
        get_elicitation: Callable[
            ...,
            types.ElicitResult
            | types.ErrorData
            | Awaitable[types.ElicitResult | types.ErrorData],
        ]
        | None = None,
    ):
        self._get_input = get_input
        self._get_confirmation = get_tool_confirmation
        self._get_code = get_code_input
        self._get_elicitation = get_elicitation

    async def get_input(
        self,
        context: AgentContext,
        prompt: str,
        output_type: type | None = None,
        message_history: list[ChatMessage] | None = None,
    ) -> Any:
        if not self._get_input:
            return input(prompt)  # fallback
        return await execute(
            self._get_input,
            context=context,
            prompt=prompt,
            output_type=output_type,
            message_history=message_history,
        )

    async def get_tool_confirmation(
        self,
        context: AgentContext,
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage] | None = None,
    ) -> ConfirmationResult:
        if not self._get_confirmation:
            return "allow"  # fallback: always allow
        return await execute(
            self._get_confirmation,
            context=context,
            tool=tool,
            args=args,
            message_history=message_history,
        )

    async def get_code_input(
        self,
        context: AgentContext,
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ) -> str:
        if not self._get_code:
            return input("Enter code: ")  # basic fallback
        return await execute(
            self._get_code,
            context=context,
            template=template,
            language=language,
            description=description,
        )

    async def get_elicitation(
        self,
        context: AgentContext,
        params: types.ElicitRequestParams,
        message_history: list[ChatMessage] | None = None,
    ) -> types.ElicitResult | types.ErrorData:
        from mcp import types

        if not self._get_elicitation:
            # fallback: return not supported
            return types.ErrorData(
                code=types.INVALID_REQUEST,
                message="Elicitation not supported",
            )
        return await execute(
            self._get_elicitation,
            context=context,
            params=params,
            message_history=message_history,
        )
