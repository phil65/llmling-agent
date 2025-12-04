"""Two-node team with a validator node."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.delegation.teamrun import TeamRun


if TYPE_CHECKING:
    from llmling_agent import Agent, MessageNode


class StructuredTeam[TResult](TeamRun[Any, TResult]):
    """A team that produces typed/structured output through a processor node."""

    def __init__(
        self,
        worker: MessageNode[Any, Any],
        processor: Agent[Any, TResult],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize structured team.

        Args:
            worker: The node doing the main work
            processor: Node that processes/validates the output
            name: Optional name for this team
            description: Optional description
        """
        name = name or f"{worker.name}>{processor.name}"
        super().__init__(agents=[worker, processor], name=name, description=description)

    @property
    def worker(self) -> MessageNode[Any, Any]:
        """Get the worker node."""
        return self.nodes[0]

    @worker.setter
    def worker(self, value: MessageNode[Any, Any]) -> None:
        """Set the worker node."""
        self.nodes[0] = value

    @property
    def processor(self) -> Agent[Any, TResult]:
        """Get the processor node."""
        return self.nodes[1]  # type: ignore

    @processor.setter
    def processor(self, value: Agent[Any, TResult]) -> None:
        """Set the processor node."""
        self.nodes[1] = value
