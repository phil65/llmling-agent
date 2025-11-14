"""Base class for message processing nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, overload
from uuid import uuid4

from anyenv import method_spawner
from psygnal import Signal

from llmling_agent.log import get_logger
from llmling_agent.messaging import ChatMessage
from llmling_agent.prompts.convert import convert_prompts
from llmling_agent.talk import AggregatedTalkStats
from llmling_agent.utils.tasks import TaskManager


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import timedelta
    from types import TracebackType

    from llmling_agent.common_types import (
        AnyTransformFn,
        AsyncFilterFn,
        ProcessorCallback,
        PromptCompatible,
        QueueStrategy,
        RichProgressCallback,
    )
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.models.content import BaseContent
    from llmling_agent.talk import Talk, TeamTalk
    from llmling_agent.talk.stats import AggregatedMessageStats, MessageStats
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
        context: NodeContext | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        enable_logging: bool = True,
        progress_handler: RichProgressCallback | None = None,
    ):
        """Initialize message node."""
        super().__init__()
        from llmling_agent.mcp_server.manager import MCPManager
        from llmling_agent.messaging import EventManager
        from llmling_agent.messaging.connection_manager import ConnectionManager

        self.task_manager = TaskManager()
        self._name = name or self.__class__.__name__
        self.log = logger.bind(agent_name=self._name)

        self.description = description
        self.connections = ConnectionManager(self)
        self._events = EventManager(self, enable_events=True)
        self.mcp = MCPManager(
            f"node_{self._name}",
            servers=mcp_servers,
            context=context,
            owner=self.name,
            progress_handler=progress_handler,
        )
        self.enable_db_logging = enable_logging
        self.conversation_id = str(uuid4())
        # Connect to the combined signal to capture all messages
        # TODO: need to check this
        # node.message_received.connect(self.log_message)

    async def __aenter__(self) -> Self:
        """Initialize base message node."""
        if self.enable_db_logging:
            await self.context.storage.log_conversation(
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
    ):
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
    def name(self) -> str:
        """Get agent name."""
        return self._name or "llmling-agent"

    @name.setter
    def name(self, value: str):
        self._name = value

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
        transform: AnyTransformFn | None = None,
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
        transform: AnyTransformFn | None = None,
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
        transform: AnyTransformFn | None = None,
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
        transform: AnyTransformFn | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> Talk[Any] | TeamTalk:
        """Create connection(s) to target(s)."""
        # Handle callable case
        from llmling_agent.agent import Agent
        from llmling_agent.delegation.base_team import BaseTeam

        if callable(target):
            target = Agent.from_callback(target)
            if pool := self.context.pool:
                pool.register(target.name, target)
        # we are explicit here just to make disctinction clear, we only want sequences
        # of message units
        if isinstance(target, Sequence) and not isinstance(target, BaseTeam):
            targets: list[MessageNode] = []
            for t in target:
                match t:
                    case _ if callable(t):
                        other: MessageNode = Agent.from_callback(t)
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

    async def disconnect_all(self):
        """Disconnect from all nodes."""
        for target in list(self.connections.get_targets()):
            self.stop_passing_results_to(target)

    def stop_passing_results_to(self, other: MessageNode):
        """Stop forwarding results to another node."""
        self.connections.disconnect(other)

    async def pre_run(
        self,
        *prompt: PromptCompatible | ChatMessage,
    ) -> tuple[ChatMessage[Any], list[BaseContent | str]]:
        """Hook to prepare a MessgeNode run call.

        Args:
            *prompt: The prompt(s) to prepare.

        Returns:
            A tuple of:
                - Either incoming message, or a constructed incoming message based
                  on the prompt(s).
                - A list of prompts to be sent to the model.
        """
        if len(prompt) == 1 and isinstance(prompt[0], ChatMessage):
            user_msg = prompt[0]
            prompts = await convert_prompts([user_msg.content])
            # Update received message's chain to show it came through its source
            user_msg = user_msg.forwarded(prompt[0]).to_request()
            # clear cost info to avoid double-counting
            final_prompt = "\n\n".join(str(p) for p in prompts)
        else:
            prompts = await convert_prompts(prompt)
            final_prompt = "\n\n".join(str(p) for p in prompts)
            # use format_prompts?
            messages = [i if isinstance(i, str) else i.to_pydantic_ai() for i in prompts]
            user_msg = ChatMessage.user_prompt(message=messages)
        self.message_received.emit(user_msg)
        self.context.current_prompt = final_prompt
        return user_msg, prompts

    # async def post_run(
    #     self,
    #     message: ChatMessage[TResult],
    #     previous_message: ChatMessage[Any] | None,
    #     wait_for_connections: bool | None = None,
    # ) -> ChatMessage[Any]:
    #     # For chain processing, update the response's chain
    #     if previous_message:
    #         message = message.forwarded(previous_message)
    #         conversation_id = previous_message.conversation_id
    #     else:
    #         conversation_id = str(uuid4())
    #     # Set conversation_id on response message
    #     message = replace(message, conversation_id=conversation_id)
    #     self.message_sent.emit(message)
    #     await self.log_message(response_msg)
    #     await self.connections.route_message(message, wait=wait_for_connections)
    #     return message

    # @overload
    # async def run(
    #     self,
    #     *prompt: PromptCompatible | ChatMessage,
    #     wait_for_connections: bool | None = None,
    #     store_history: bool = True,
    #     output_type: None,
    #     **kwargs: Any,
    # ) -> ChatMessage[TResult]: ...

    # @overload
    # async def run[OutputTypeT](
    #     self,
    #     *prompt: PromptCompatible | ChatMessage,
    #     wait_for_connections: bool | None = None,
    #     store_history: bool = True,
    #     output_type: type[OutputTypeT],
    #     **kwargs: Any,
    # ) -> ChatMessage[OutputTypeT]: ...

    @method_spawner
    async def run[OutputTypeT](
        self,
        *prompt: PromptCompatible | ChatMessage,
        wait_for_connections: bool | None = None,
        store_history: bool = True,
        output_type: type[OutputTypeT] | None = None,
        **kwargs: Any,
    ) -> ChatMessage[Any]:
        """Execute node with prompts and handle message routing.

        Args:
            prompt: Input prompts
            wait_for_connections: Whether to wait for forwarded messages
            store_history: Whether to store in conversation history
            output_type: Type of output to expect
            **kwargs: Additional arguments for _run
        """
        from llmling_agent import Agent

        user_msg, prompts = await self.pre_run(*prompt)
        message = await self._run(
            *prompts,
            store_history=store_history,
            conversation_id=user_msg.conversation_id,
            output_type=output_type,
            **kwargs,
        )

        # For chain processing, update the response's chain
        if len(prompt) == 1 and isinstance(prompt[0], ChatMessage):
            message = message.forwarded(prompt[0])

        if store_history and isinstance(self, Agent):
            self.conversation.add_chat_messages([user_msg, message])
        self.message_sent.emit(message)
        await self.log_message(message)
        await self.connections.route_message(message, wait=wait_for_connections)
        return message

    @abstractmethod
    async def _run(
        self,
        *prompts: Any,
        **kwargs: Any,
    ) -> ChatMessage[TResult]:
        """Implementation-specific run logic."""

    async def get_message_history(self, limit: int | None = None) -> list[ChatMessage]:
        """Get message history from storage."""
        if not self.enable_db_logging:
            return []  # No history if not logging

        from llmling_agent_config.session import SessionQuery

        query = SessionQuery(name=self.conversation_id, limit=limit)
        return await self.context.storage.filter_messages(query)

    async def log_message(self, message: ChatMessage):
        """Handle message from chat signal."""
        if self.enable_db_logging:
            await self.context.storage.log_message(message)  # pyright: ignore

    @abstractmethod
    async def get_stats(self) -> MessageStats | AggregatedMessageStats:
        """Get message statistics for this node."""

    @abstractmethod
    def run_iter(
        self,
        *prompts: Any,
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages during execution."""
