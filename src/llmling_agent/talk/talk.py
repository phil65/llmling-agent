"""Manages message flow between agents/groups."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
import inspect
from typing import TYPE_CHECKING, Any, Literal, Self

from psygnal import Signal
from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage
from llmling_agent.talk.stats import TalkStats, TeamTalkStats


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from datetime import timedelta

    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.common_types import AnyFilterFn
    from llmling_agent.models.forward_targets import ConnectionType

TContent = TypeVar("TContent")
QueueStrategy = Literal["concat", "latest", "buffer"]
logger = get_logger(__name__)


class Talk[TTransmittedData]:
    """Manages message flow between agents/groups."""

    message_received = Signal(ChatMessage[TTransmittedData])  # Original message
    message_forwarded = Signal(ChatMessage[Any])  # After any transformation

    def __init__(
        self,
        source: AnyAgent[Any, TTransmittedData],
        targets: list[AnyAgent[Any, Any]],
        group: TeamTalk | None = None,
        *,
        connection_type: ConnectionType = "run",
        wait_for_connections: bool = False,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: Callable[[Any], Any | Awaitable[Any]] | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
    ):
        """Initialize talk connection.

        Args:
            source: Agent sending messages
            targets: Agents receiving messages
            group: Optional group this talk belongs to
            connection_type: How to handle messages:
                - "run": Execute message as a new run in target
                - "context": Add message as context to target
                - "forward": Forward message to target's outbox
            wait_for_connections: Whether to wait for all targets to complete
            priority: Task priority (lower = higher priority)
            delay: Optional delay before processing
            queued: Whether messages should be queued for manual processing
            queue_strategy: How to process queued messages:
                - "concat": Combine all messages with newlines
                - "latest": Use only the most recent message
                - "buffer": Process all messages individually
            transform: Optional function to transform messages
            filter_condition: Optional condition for filtering messages
            stop_condition: Optional condition for disconnecting
            exit_condition: Optional condition for stopping the event loop
        """
        self.source = source
        self.targets = targets
        self.group = group
        self.priority = priority
        self.delay = delay
        self.active = True
        self.connection_type = connection_type
        self.wait_for_connections = wait_for_connections
        self.queued = queued
        self.queue_strategy = queue_strategy
        self._pending_messages: dict[str, list[ChatMessage[TTransmittedData]]] = {
            target.name: [] for target in targets
        }
        names = {t.name for t in targets}
        self._stats = TalkStats(source_name=source.name, target_names=names)
        self._transform = transform
        self._filter_condition = filter_condition
        self._stop_condition = stop_condition
        self._exit_condition = exit_condition

    def __repr__(self):
        targets = [t.name for t in self.targets]
        return f"<Talk({self.connection_type}) {self.source.name} -> {targets}>"

    async def _evaluate_condition(
        self,
        condition: Callable[..., bool | Awaitable[bool]] | None,
        message: ChatMessage[Any],
        target: AnyAgent[Any, Any],
        *,
        default_return: bool = False,
    ) -> bool:
        """Evaluate a condition with flexible parameter handling."""
        if not condition:
            return default_return

        # Get number of parameters
        sig = inspect.signature(condition)
        param_count = len(sig.parameters)

        # Call with appropriate number of arguments
        match param_count:
            case 1:
                result = condition(message)
            case 2:
                result = condition(message, target)
            case 3:
                result = condition(message, target, self.stats)
            case _:
                msg = f"Condition must take 1-3 parameters, got {param_count}"
                raise ValueError(msg)

        if inspect.isawaitable(result):
            return await result
        return result

    async def _should_route_to(
        self,
        message: ChatMessage[Any],
        target: AnyAgent[Any, Any],
    ) -> bool:
        """Determine if message should be routed to target."""
        return await self._evaluate_condition(
            self._filter_condition,
            message,
            target,
            default_return=True,
        )

    async def _handle_message(
        self,
        message: ChatMessage[TTransmittedData],
        prompt: str | None = None,
    ) -> list[ChatMessage[Any]]:
        """Handle message forwarding based on connection configuration."""
        # 1. Initial message handling
        self.source.outbox.emit(message, None)

        # 2. Early exit checks
        if not (self.active and (not self.group or self.group.active)):
            return []

        # 3. Check exit condition for any target
        for target in self.targets:
            # Exit if condition returns True
            if await self._evaluate_condition(self._exit_condition, message, target):
                raise SystemExit

        # 4. Check stop condition for any target
        for target in self.targets:
            # Stop if condition returns True
            if await self._evaluate_condition(self._stop_condition, message, target):
                self.disconnect()
                return []

        # 5. Transform if configured
        processed_message = message
        if self._transform:
            transformed = self._transform(message)
            if inspect.isawaitable(transformed):
                processed_message = await transformed
            else:
                processed_message = transformed

        # 8. Process for each target
        responses: list[ChatMessage[Any]] = []
        is_forwarded = False
        for target in self.targets:
            if await self._should_route_to(processed_message, target):
                is_forwarded = True
                if self.queued:
                    # Queue per agent
                    self._pending_messages[target.name].append(processed_message)
                    continue
                if response := await self._process_for_target(
                    processed_message, target, prompt
                ):
                    responses.append(response)
        if is_forwarded:
            self._stats = replace(
                self._stats,
                messages=[*self._stats.messages, processed_message],
            )
            # 9. Emit forwarded signal after processing
            self.message_forwarded.emit(processed_message)
        return responses

    async def _process_for_target(
        self,
        message: ChatMessage[Any],
        target: AnyAgent[Any, Any],
        prompt: str | None = None,
    ) -> ChatMessage[Any] | None:
        """Process message for a single target."""
        match self.connection_type:
            case "run":
                prompts: list[AnyPromptType] = [message.content]
                if prompt:
                    prompts.append(prompt)
                response = await target.run(*prompts)
                response.forwarded_from.append(target.name)
                target.outbox.emit(response, None)
                return response

            case "context":
                meta = {
                    "type": "forwarded_message",
                    "role": message.role,
                    "model": message.model,
                    "cost_info": message.cost_info,
                    "timestamp": message.timestamp.isoformat(),
                    "prompt": prompt,
                }

                async def add_context():
                    target.conversation.add_context_message(
                        str(message.content),
                        source=self.source.name,
                        metadata=meta,
                    )

                if self.delay is not None or self.priority != 0:
                    target.run_background(
                        add_context(),
                        priority=self.priority,
                        delay=self.delay,
                    )
                else:
                    await add_context()
                return None

            case "forward":
                if self.delay is not None or self.priority != 0:

                    async def delayed_emit():
                        target.outbox.emit(message, prompt)

                    target.run_background(
                        delayed_emit(),
                        priority=self.priority,
                        delay=self.delay,
                    )
                else:
                    target.outbox.emit(message, prompt)
                return None

    async def trigger(self) -> list[ChatMessage[TTransmittedData]]:
        """Process queued messages."""
        if not self._pending_messages:
            return []

        match self.queue_strategy:
            case "buffer":
                results: list[ChatMessage[TTransmittedData]] = []
                # Process each agent's queue
                for target in self.targets:
                    queue = self._pending_messages[target.name]
                    for message in queue:
                        if response := await self._process_for_target(
                            message, target, None
                        ):
                            results.append(response)  # noqa: PERF401
                    queue.clear()
                return results

            case "latest":
                results = []
                # Get latest message for each agent
                for target in self.targets:
                    queue = self._pending_messages[target.name]
                    if queue:
                        latest = queue[-1]
                        if response := await self._process_for_target(
                            latest, target, None
                        ):
                            results.append(response)
                        queue.clear()
                return results

            case "concat":
                results = []
                # Concat messages per agent
                for target in self.targets:
                    queue = self._pending_messages[target.name]
                    if not queue:
                        continue

                    base = queue[-1]
                    contents = [str(m.content) for m in queue]
                    meta = {
                        **base.metadata,
                        "merged_count": len(queue),
                        "queue_strategy": self.queue_strategy,
                    }
                    content = "\n\n".join(contents)
                    merged = replace(base, content=content, metadata=meta)  # type: ignore

                    if response := await self._process_for_target(merged, target, None):
                        results.append(response)
                    queue.clear()

                return results
            case _:
                msg = f"Invalid queue strategy: {self.queue_strategy}"
                raise ValueError(msg)

    def when(self, condition: AnyFilterFn) -> Self:
        """Add condition for message forwarding."""
        self._filter_condition = condition
        return self

    @asynccontextmanager
    async def paused(self):
        """Temporarily set inactive."""
        previous = self.active
        self.active = False
        try:
            yield self
        finally:
            self.active = previous

    def disconnect(self):
        """Permanently disconnect the connection."""
        self.active = False

    @property
    def stats(self) -> TalkStats:
        """Get current connection statistics."""
        return self._stats


class TeamTalk(list["Talk | TeamTalk"]):
    """Group of connections with aggregate operations."""

    def __init__(self, talks: Sequence[Talk | TeamTalk]):
        super().__init__(talks)
        self._filter_condition: AnyFilterFn | None = None
        self.active = True

    def __repr__(self):
        return f"TeamTalk({list(self)})"

    @property
    def targets(self) -> list[AnyAgent[Any, Any]]:
        """Get all targets from all connections."""
        return [t for talk in self for t in talk.targets]

    async def _handle_message(self, message: ChatMessage[Any], prompt: str | None = None):
        for talk in self:
            await talk._handle_message(message, prompt)

    @classmethod
    def from_agents(
        cls,
        agents: Sequence[AnyAgent[Any, Any]],
        targets: list[AnyAgent[Any, Any]] | None = None,
    ) -> TeamTalk:
        """Create TeamTalk from a collection of agents."""
        return cls([Talk(agent, targets or []) for agent in agents])

    @asynccontextmanager
    async def paused(self):
        """Temporarily set inactive."""
        previous = self.active
        self.active = False
        try:
            yield self
        finally:
            self.active = previous

    def has_active_talks(self) -> bool:
        """Check if any contained talks are active."""
        return any(talk.active for talk in self)

    def get_active_talks(self) -> list[Talk | TeamTalk]:
        """Get list of currently active talks."""
        return [talk for talk in self if talk.active]

    @property
    def stats(self) -> TeamTalkStats:
        """Get aggregated statistics for all connections."""
        return TeamTalkStats(stats=[talk.stats for talk in self])

    def when(self, condition: AnyFilterFn) -> Self:
        """Add condition to all connections in group."""
        for talk in self:
            talk.when(condition)
        return self

    def disconnect(self):
        """Disconnect all connections in group."""
        for talk in self:
            talk.disconnect()
