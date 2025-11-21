"""Worker configuration models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field
from schemez import Schema


class BaseWorkerConfig(Schema):
    """Base configuration for workers.

    Workers are nodes that can be registered as tools with a parent node.
    This allows building hierarchies of specialized nodes.
    """

    name: str = Field(
        examples=["web_agent", "code_analyzer", "data_processor"],
        title="Worker node name",
    )
    """Name of the node to use as a worker."""

    model_config = ConfigDict(frozen=True)


class TeamWorkerConfig(BaseWorkerConfig):
    """Configuration for team workers.

    Team workers allow using entire teams as tools for other nodes.
    """

    type: Literal["team"] = Field("team", init=False)
    """Team worker configuration."""


class AgentWorkerConfig(BaseWorkerConfig):
    """Configuration for agent workers.

    Agent workers provide advanced features like history management and
    context sharing between agents.
    """

    type: Literal["agent"] = Field("agent", init=False)
    """Agent worker configuration."""

    reset_history_on_run: bool = Field(default=True, title="Reset history on run")
    """Whether to clear worker's conversation history before each run.
    True (default): Fresh conversation each time
    False: Maintain conversation context between runs
    """

    pass_message_history: bool = Field(default=False, title="Pass message history")
    """Whether to pass parent agent's message history to worker.
    True: Worker sees parent's conversation context
    False (default): Worker only sees current request
    """


WorkerConfig = Annotated[
    TeamWorkerConfig | AgentWorkerConfig,
    Field(discriminator="type"),
]
