"""Base class for all agent types."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from llmling_agent.log import get_logger
from llmling_agent.messaging import MessageNode


if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncIterator

    from anyenv import MultiEventHandler
    from exxec import ExecutionEnvironment

    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.common_types import IndividualEventHandler
    from llmling_agent.messaging import MessageHistory
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.tools import ToolManager
    from llmling_agent.ui.base import InputProvider


logger = get_logger(__name__)

ToolConfirmationMode = Literal["always", "never", "per_tool"]


class BaseAgent[TDeps = None, TResult = str](MessageNode[TDeps, TResult]):
    """Base class for Agent, ACPAgent, and AGUIAgent.

    Provides shared infrastructure:
    - tools: ToolManager for tool registration and execution
    - conversation: MessageHistory for conversation state
    - event_handler: MultiEventHandler for event distribution
    - _event_queue: Queue for streaming events
    - tool_confirmation_mode: Tool confirmation behavior
    - _input_provider: Provider for user input/confirmations
    - env: ExecutionEnvironment for running code/commands
    - context property: Returns NodeContext for the agent
    """

    tools: ToolManager
    """Tool manager for this agent."""

    conversation: MessageHistory
    """Conversation history manager."""

    event_handler: MultiEventHandler[IndividualEventHandler]
    """Event handler for distributing events."""

    _event_queue: asyncio.Queue[RichAgentStreamEvent[Any]]
    """Queue for streaming events."""

    tool_confirmation_mode: ToolConfirmationMode
    """How tool execution confirmation is handled."""

    _input_provider: InputProvider | None
    """Provider for user input and confirmations."""

    env: ExecutionEnvironment
    """Execution environment for running code/commands."""

    _output_type: type = str
    """Output type for this agent (default: str)."""

    @property
    @abstractmethod
    def context(self) -> NodeContext[Any]:
        """Get agent context."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str | None:
        """Get the model name used by this agent."""
        ...

    @abstractmethod
    def run_stream(
        self,
        *prompt: Any,
        **kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[TResult]]:
        """Run agent with streaming output.

        Args:
            *prompt: Input prompts
            **kwargs: Additional arguments

        Yields:
            Stream events during execution
        """
        ...

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set tool confirmation mode.

        Args:
            mode: Confirmation mode - "always", "never", or "per_tool"
        """
        self.tool_confirmation_mode = mode
