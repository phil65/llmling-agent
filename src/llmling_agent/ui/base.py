"""Base input provider class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from mcp import types
    from pydantic import BaseModel

    from llmling_agent.agents.context import ConfirmationResult
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.tools.base import Tool


class InputProvider(ABC):
    """Base class for handling all UI interactions."""

    async def get_input(
        self,
        context: NodeContext,
        prompt: str,
        output_type: type | None = None,
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> Any:
        """Get normal input (used by HumanProvider).

        Args:
            context: Current agent context
            prompt: The prompt to show to the user
            output_type: Optional type for structured responses
            message_history: Optional conversation history
        """
        if output_type:
            return await self.get_structured_input(context, prompt, output_type, message_history)
        return await self.get_text_input(context, prompt, message_history)

    async def get_text_input(
        self,
        context: NodeContext[Any],
        prompt: str,
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> str:
        """Get normal text input."""
        raise NotImplementedError

    async def get_structured_input(
        self,
        context: NodeContext[Any],
        prompt: str,
        output_type: type[BaseModel],
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> BaseModel:
        """Get structured input."""
        raise NotImplementedError

    @abstractmethod
    def get_tool_confirmation(
        self,
        context: NodeContext[Any],
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> Coroutine[Any, Any, ConfirmationResult]:
        """Get tool execution confirmation.

        Args:
            context: Current node context
            tool: Information about the tool to be executed
            args: Tool arguments
            message_history: Optional conversation history
        """

    @abstractmethod
    def get_elicitation(
        self,
        params: types.ElicitRequestParams,
    ) -> Coroutine[Any, Any, types.ElicitResult | types.ErrorData]:
        """Get user response to elicitation request.

        Args:
            context: Current agent context
            params: MCP elicit request parameters
        """
