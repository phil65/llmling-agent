"""Base class for message processing nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Coroutine, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from psygnal import Signal

from llmling_agent.utils.inspection import has_return_type
from llmling_agent.utils.tasks import TaskManagerMixin


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import timedelta
    import os

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.common_types import AnyTransformFn, AsyncFilterFn
    from llmling_agent.models.forward_targets import ConnectionType
    from llmling_agent.models.messages import ChatMessage
    from llmling_agent.models.providers import ProcessorCallback
    from llmling_agent.talk import QueueStrategy, Talk, TeamTalk


NodeType = TypeVar("NodeType", bound="MessageNode")
TResult = TypeVar("TResult")


class MessageNode[TDeps, TResult](TaskManagerMixin, ABC):
    """Base class for all message processing nodes."""

    outbox = Signal(object)  # ChatMessage
    """Signal emitted when node produces a message."""

    def __init__(self, name: str | None = None):
        """Initialize message node."""
        super().__init__()
        from llmling_agent.agent.connection import ConnectionManager

        self._name = name or self.__class__.__name__
        self.connections = ConnectionManager(self)

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
        connection_type: ConnectionType = ...,
        priority: int = ...,
        delay: timedelta | None = ...,
        queued: bool = ...,
        queue_strategy: QueueStrategy = ...,
        transform: AnyTransformFn | None = ...,
        filter_condition: AsyncFilterFn | None = ...,
        stop_condition: AsyncFilterFn | None = ...,
        exit_condition: AsyncFilterFn | None = ...,
    ) -> TeamTalk[TResult]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
        *,
        connection_type: ConnectionType = "run",
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
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.base_team import BaseTeam

        if callable(target):
            if has_return_type(target, str):
                target = Agent.from_callback(target)
            else:
                target = StructuredAgent.from_callback(target)
        # we are explicit here just to make disctinction clear, we only want sequences
        # of message units
        if isinstance(target, Sequence) and not isinstance(target, BaseTeam):
            targets: list[Agent | StructuredAgent] = []
            for t in target:
                match t:
                    case _ if callable(t):
                        if has_return_type(t, str):
                            targets.append(Agent.from_callback(t))
                        else:
                            targets.append(StructuredAgent.from_callback(t))
                    case Agent() | StructuredAgent():
                        targets.append(t)
                    case _:
                        msg = f"Invalid agent type: {type(t)}"
                        raise TypeError(msg)
        else:
            targets = target  # type: ignore
        return self.connections.create_connection(
            self,
            targets,
            connection_type=connection_type,
            priority=priority,
            delay=delay,
            queued=queued,
            queue_strategy=queue_strategy,
            transform=transform,
            filter_condition=filter_condition,
            stop_condition=stop_condition,
            exit_condition=exit_condition,
        )

    async def run(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        wait_for_connections: bool | None = None,
        **kwargs: Any,
    ) -> ChatMessage[TResult]:
        """Execute node with prompts and handle message routing.

        Args:
            prompts: Input prompts
            wait_for_connections: Whether to wait for forwarded messages
            **kwargs: Additional arguments for _run
        """
        message = await self._run(*prompts, **kwargs)
        await self.connections.route_message(
            message,
            wait=wait_for_connections,
        )
        return message

    @abstractmethod
    def _run(
        self,
        *prompts: Any,
        **kwargs: Any,
    ) -> Coroutine[None, None, ChatMessage[TResult]]:
        """Implementation-specific run logic."""

    @abstractmethod
    def run_iter(
        self,
        *prompts: Any,
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages during execution."""
