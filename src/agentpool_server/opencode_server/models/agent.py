"""Agent and command models."""

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class Agent(OpenCodeBaseModel):
    """Agent information."""

    id: str
    name: str
    description: str = ""


class Command(OpenCodeBaseModel):
    """Slash command."""

    name: str
    description: str = ""
