"""Base input provider class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from mcp import types
    from pydantic import BaseModel

    from agentpool.agents.context import AgentContext, ConfirmationResult
    from agentpool.messaging import ChatMessage
    from agentpool.messaging.context import NodeContext


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
        context: AgentContext[Any],
        tool_name: str,
        tool_description: str,
        args: dict[str, Any],
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> Coroutine[Any, Any, ConfirmationResult]:
        """Get tool execution confirmation.

        Args:
            context: Current node context
            tool_name: Name of the tool to be executed
            tool_description: Human-readable description of the tool
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
