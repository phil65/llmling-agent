"""Base class for all agent types."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from llmling_agent.log import get_logger
from llmling_agent.messaging import MessageNode


if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncIterator, Sequence

    from anyenv import MultiEventHandler
    from evented.configs import EventConfig
    from exxec import ExecutionEnvironment

    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.common_types import IndividualEventHandler
    from llmling_agent.delegation import AgentPool
    from llmling_agent.messaging import MessageHistory
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.tools import ToolManager
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.mcp_server import MCPServerConfig


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

    def __init__(
        self,
        name: str,
        *,
        description: str | None = None,
        display_name: str | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
    ) -> None:
        """Initialize base agent.

        Args:
            name: Agent name
            description: Agent description
            display_name: Human-readable display name
            mcp_servers: MCP servers to connect to
            agent_pool: Agent pool this agent belongs to
            enable_logging: Whether to enable database logging
            event_configs: Event trigger configurations
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
