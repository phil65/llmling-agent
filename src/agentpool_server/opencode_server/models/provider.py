"""Provider, model, and mode related models."""

from typing import Any

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class ModelCost(OpenCodeBaseModel):
    """Cost information for a model."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None


class ModelLimit(OpenCodeBaseModel):
    """Limit information for a model."""

    context: float
    output: float


class Model(OpenCodeBaseModel):
    """Model information."""

    id: str
    name: str
    attachment: bool = False
    cost: ModelCost
    limit: ModelLimit
    options: dict[str, Any] = Field(default_factory=dict)
    reasoning: bool = False
    release_date: str = ""
    temperature: bool = True
    tool_call: bool = True
    variants: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Model variants for reasoning/thinking levels.

    Maps variant names (e.g., 'low', 'medium', 'high', 'max') to
    provider-specific configuration options. The TUI uses this to
    let users cycle through thinking effort levels.
    """


class Provider(OpenCodeBaseModel):
    """Provider information."""

    id: str
    name: str
    env: list[str] = Field(default_factory=list)
    models: dict[str, Model] = Field(default_factory=dict)
    api: str | None = None
    npm: str | None = None


class ProvidersResponse(OpenCodeBaseModel):
    """Response for /config/providers endpoint."""

    providers: list[Provider]
    default: dict[str, str] = Field(default_factory=dict)


class ProviderListResponse(OpenCodeBaseModel):
    """Response for /provider endpoint."""

    all: list[Provider]
    default: dict[str, str] = Field(default_factory=dict)
    connected: list[str] = Field(default_factory=list)


class ModeModel(OpenCodeBaseModel):
    """Model selection for a mode."""

    model_id: str
    provider_id: str


class Mode(OpenCodeBaseModel):
    """Agent mode configuration."""

    name: str
    tools: dict[str, bool] = Field(default_factory=dict)
    model: ModeModel | None = None
    prompt: str | None = None
    temperature: float | None = None
