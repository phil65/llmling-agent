"""Base class for message processing nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, overload
from uuid import uuid4

from psygnal import Signal

from llmling_agent.log import get_logger
from llmling_agent.messaging import ChatMessage
from llmling_agent.talk import AggregatedTalkStats
from llmling_agent.utils.tasks import TaskManager


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import timedelta
    from types import TracebackType

    from evented.configs import EventConfig
    from evented.event_data import EventData

    from llmling_agent.common_types import (
        AnyTransformFn,
        AsyncFilterFn,
        ProcessorCallback,
        QueueStrategy,
    )
    from llmling_agent.delegation import AgentPool
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.storage import StorageManager
    from llmling_agent.talk import Talk, TeamTalk
    from llmling_agent.talk.stats import AggregatedMessageStats, MessageStats
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.forward_targets import ConnectionType
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


class MessageNode[TDeps, TResult](ABC):
    """Base class for all message processing nodes."""

    message_received = Signal(ChatMessage)
    """Signal emitted when node receives a message."""

    message_sent = Signal(ChatMessage)
    """Signal emitted when node creates a message."""

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
    ) -> None:
        """Initialize message node."""
        super().__init__()

        from llmling_agent.mcp_server.manager import MCPManager
        from llmling_agent.messaging import EventManager
        from llmling_agent.messaging.connection_manager import ConnectionManager

        async def _event_handler(event: EventData) -> None:
            if prompt := event.to_prompt():
                await self.run(prompt)

        self.task_manager = TaskManager()
        self._name = name or self.__class__.__name__
        self._display_name = display_name
        self.log = logger.bind(agent_name=self._name)
        self.agent_pool = agent_pool
        self.description = description
        self.connections = ConnectionManager(self)
        cfgs = list(event_configs) if event_configs else None
        self._events = EventManager(configs=cfgs, event_callbacks=[_event_handler])
        name_ = f"node_{self._name}"
        self.mcp = MCPManager(name_, servers=mcp_servers, owner=self.name)
        self.enable_db_logging = enable_logging
        self.conversation_id = str(uuid4())
        # Connect to the combined signal to capture all messages
        # TODO: need to check this
        # node.message_received.connect(self.log_message)

    async def __aenter__(self) -> Self:
        """Initialize base message node."""
        if self.enable_db_logging and self.storage:
            await self.storage.log_conversation(
                conversation_id=self.conversation_id,
                node_name=self.name,
            )
        try:
            await self._events.__aenter__()
            await self.mcp.__aenter__()
        except Exception as e:
            await self.__aexit__(type(e), e, e.__traceback__)
            msg = f"Failed to initialize {self.name}"
            raise RuntimeError(msg) from e
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up base resources."""
        await self._events.__aexit__(exc_type, exc_val, exc_tb)
        await self.mcp.__aexit__(exc_type, exc_val, exc_tb)
        await self.task_manager.cleanup_tasks()

    @property
    def connection_stats(self) -> AggregatedTalkStats:
        """Get stats for all active connections of this node."""
        stats = [talk.stats for talk in self.connections.get_connections()]
        return AggregatedTalkStats(stats=stats)

    @property
    def context(self) -> NodeContext:
        """Get node context."""
        raise NotImplementedError

    @property
    def storage(self) -> StorageManager | None:
        """Get storage manager from pool."""
        return self.agent_pool.storage if self.agent_pool else None

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name or "llmling-agent"

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def display_name(self) -> str:
        """Get human-readable display name, falls back to name."""
        return self._display_name or self._name

    @display_name.setter
    def display_name(self, value: str | None) -> None:
        self._display_name = value

    def to_tool(
        self, *, name: str | None = None, description: str | None = None, **kwargs: Any
    ) -> Tool[TResult]:
        """Convert node to a callable tool.

        Args:
            name: Optional tool name override
            description: Optional tool description override
            **kwargs: Additional arguments for subclass customization

        Returns:
            Tool instance that can be registered
        """
        from llmling_agent.tools.base import Tool

        async def wrapped(prompt: str) -> TResult:
            result = await self.run(prompt)
            return result.content

        tool_name = name or f"ask_{self.name}"
        docstring = description or f"Get expert answer from {self.name}"
        if self.description:
            docstring = f"{docstring}\n\n{self.description}"
        wrapped.__doc__ = docstring
        wrapped.__name__ = tool_name
        return Tool.from_callable(wrapped, name_override=tool_name, description_override=docstring)

    @overload
    def __rshift__(
        self, other: MessageNode[Any, Any] | ProcessorCallback[Any]
    ) -> Talk[TResult]: ...

    @overload
    def __rshift__(
        self, other: Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]]
    ) -> TeamTalk[TResult]: ...

    def __rshift__(
        self,
        other: MessageNode[Any, Any]
        | ProcessorCallback[Any]
        | Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
    ) -> Talk[Any] | TeamTalk[Any]:
        """Connect agent to another agent or group.

        Example:
            agent >> other_agent  # Connect to single agent
            agent >> (agent2 & agent3)  # Connect to group
            agent >> "other_agent"  # Connect by name (needs pool)
        """
        return self.connect_to(other)

    @overload
    def connect_to(
        self,
        target: MessageNode[Any, Any] | ProcessorCallback[Any],
        *,
        queued: Literal[True],
        queue_strategy: Literal["concat"],
    ) -> Talk[str]: ...

    @overload
    def connect_to(
        self,
        target: MessageNode[Any, Any] | ProcessorCallback[Any],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn[Any] | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> Talk[TResult]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
        *,
        queued: Literal[True],
        queue_strategy: Literal["concat"],
    ) -> TeamTalk[str]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[MessageNode[Any, TResult] | ProcessorCallback[TResult]],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn[Any] | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> TeamTalk[TResult]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn[Any] | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> TeamTalk: ...

    def connect_to(
        self,
        target: MessageNode[Any, Any]
        | ProcessorCallback[Any]
        | Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn[Any] | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> Talk[Any] | TeamTalk:
        """Create connection(s) to target(s)."""
        # Handle callable case
        from llmling_agent.agents import Agent
        from llmling_agent.delegation.base_team import BaseTeam

        if callable(target):
            target = Agent.from_callback(target)
            if pool := self.context.pool:
                pool.register(target.name, target)
        # we are explicit here just to make disctinction clear, we only want sequences
        # of message units
        if isinstance(target, Sequence) and not isinstance(target, BaseTeam):
            targets: list[MessageNode[Any, Any]] = []
            for t in target:
                match t:
                    case _ if callable(t):
                        other = Agent.from_callback(t)
                        if pool := self.context.pool:
                            pool.register(other.name, other)
                        targets.append(other)
                    case MessageNode():
                        targets.append(t)
                    case _:
                        msg = f"Invalid node type: {type(t)}"
                        raise TypeError(msg)
        else:
            targets = target  # type: ignore
        return self.connections.create_connection(
            self,
            targets,
            connection_type=connection_type,
            priority=priority,
            name=name,
            delay=delay,
            queued=queued,
            queue_strategy=queue_strategy,
            transform=transform,
            filter_condition=filter_condition,
            stop_condition=stop_condition,
            exit_condition=exit_condition,
        )

    async def disconnect_all(self) -> None:
        """Disconnect from all nodes."""
        for target in list(self.connections.get_targets()):
            self.stop_passing_results_to(target)

    def stop_passing_results_to(self, other: MessageNode[Any, Any]) -> None:
        """Stop forwarding results to another node."""
        self.connections.disconnect(other)

    @abstractmethod
    async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[TResult]:
        """Execute node with prompts. Implementation-specific run logic."""

    async def get_message_history(self, limit: int | None = None) -> list[ChatMessage[Any]]:
        """Get message history from storage."""
        from llmling_agent_config.session import SessionQuery

        if not self.enable_db_logging or not self.storage:
            return []
        query = SessionQuery(name=self.conversation_id, limit=limit)
        return await self.storage.filter_messages(query)

    async def log_message(self, message: ChatMessage[Any]) -> None:
        """Handle message from chat signal."""
        if self.enable_db_logging and self.storage:
            await self.storage.log_message(message)

    @abstractmethod
    async def get_stats(self) -> MessageStats | AggregatedMessageStats:
        """Get message statistics for this node."""

    @abstractmethod
    def run_iter(self, *prompts: Any, **kwargs: Any) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages during execution."""
