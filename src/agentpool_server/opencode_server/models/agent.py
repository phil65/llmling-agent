"""Agent and command models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class AgentPermission(OpenCodeBaseModel):
    """Agent permission settings."""

    edit: Literal["ask", "allow", "deny"] = "ask"
    bash: dict[str, Literal["ask", "allow", "deny"]] = Field(default_factory=dict)
    skill: dict[str, Literal["ask", "allow", "deny"]] = Field(default_factory=dict)
    webfetch: Literal["ask", "allow", "deny"] | None = None
    doom_loop: Literal["ask", "allow", "deny"] | None = None
    external_directory: Literal["ask", "allow", "deny"] | None = None


class AgentModel(OpenCodeBaseModel):
    """Agent model configuration."""

    model_id: str
    provider_id: str


class Agent(OpenCodeBaseModel):
    """Agent information matching SDK type."""

    name: str
    description: str | None = None
    mode: Literal["subagent", "primary", "all"] = "primary"
    native: bool | None = None
    hidden: bool | None = None
    default: bool | None = None
    top_p: float | None = None
    temperature: float | None = None
    color: str | None = None
    permission: AgentPermission = Field(default_factory=AgentPermission)
    model: AgentModel | None = None
    prompt: str | None = None
    tools: dict[str, bool] = Field(default_factory=dict)
    options: dict[str, str] = Field(default_factory=dict)


class Command(OpenCodeBaseModel):
    """Slash command."""

    name: str
    description: str = ""
