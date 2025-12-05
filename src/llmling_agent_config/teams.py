"""Team configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from llmling_agent_config.nodes import NodeConfig


ExecutionMode = Literal["parallel", "sequential"]


class TeamConfig(NodeConfig):
    """Configuration for a team or chain of message nodes.

    Teams can be either parallel execution groups or sequential chains.
    They can contain both agents and other teams as members.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/team_configuration/
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-icon": "fluent:people-team-24-regular",
            "x-doc-title": "Team Configuration",
        }
    )

    mode: ExecutionMode = Field(examples=["parallel", "sequential"], title="Execution mode")
    """Execution mode for team members."""

    members: list[str] = Field(
        examples=[["agent1", "agent2"], ["web_searcher", "data_analyzer", "reporter"]],
        title="Team members",
    )
    """Names of agents or other teams that are part of this team."""

    shared_prompt: str | None = Field(
        default=None,
        examples=["Work together to solve this problem", "Follow the team guidelines"],
        title="Shared prompt",
    )
    """Optional shared prompt for this team."""

    # Future extensions:
    # tools: list[str] | None = None
    # """Tools available to all team members."""

    # knowledge: Knowledge | None = None
    # """Knowledge sources shared by all team members."""
