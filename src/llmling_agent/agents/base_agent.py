"""Base class for all agent types."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import TYPE_CHECKING, Any, Literal

from anyenv import MultiEventHandler
from exxec import LocalExecutionEnvironment

from llmling_agent.agents.events import resolve_event_handlers
from llmling_agent.log import get_logger
from llmling_agent.messaging import MessageHistory, MessageNode
from llmling_agent.tools.manager import ToolManager


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from evented.configs import EventConfig
    from exxec import ExecutionEnvironment

    from llmling_agent.agents.context import AgentContext
    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.common_types import BuiltinEventHandlerType, IndividualEventHandler
    from llmling_agent.delegation import AgentPool
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)

ToolConfirmationMode = Literal["always", "never", "per_tool"]


class BaseAgent[TDeps = None, TResult = str](MessageNode[TDeps, TResult]):
    """Base class for Agent, ACPAgent, AGUIAgent, and ClaudeCodeAgent.

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

    def __init__(
        self,
        *,
        name: str = "agent",
        description: str | None = None,
        display_name: str | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        # New shared parameters
        env: ExecutionEnvironment | None = None,
        input_provider: InputProvider | None = None,
        output_type: type[TResult] = str,  # type: ignore[assignment]
        tool_confirmation_mode: ToolConfirmationMode = "per_tool",
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
    ) -> None:
        """Initialize base agent with shared infrastructure.

        Args:
            name: Agent name
            description: Agent description
            display_name: Human-readable display name
            mcp_servers: MCP server configurations
            agent_pool: Agent pool for coordination
            enable_logging: Whether to enable database logging
            event_configs: Event trigger configurations
            env: Execution environment for running code/commands
            input_provider: Provider for user input and confirmations
            output_type: Output type for this agent
            tool_confirmation_mode: How tool execution confirmation is handled
            event_handlers: Event handlers for this agent
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

        # Shared infrastructure - previously duplicated in all 4 agents
        self._event_queue: asyncio.Queue[RichAgentStreamEvent[Any]] = asyncio.Queue()
        self.conversation = MessageHistory()
        self.env = env or LocalExecutionEnvironment()
        self._input_provider = input_provider
        self._output_type: type[TResult] = output_type
        self.tool_confirmation_mode: ToolConfirmationMode = tool_confirmation_mode
        self.tools = ToolManager()
        resolved_handlers = resolve_event_handlers(event_handlers)
        self.event_handler: MultiEventHandler[IndividualEventHandler] = MultiEventHandler(
            resolved_handlers
        )

    @property
    @abstractmethod
    def context(self) -> AgentContext[Any]:
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
