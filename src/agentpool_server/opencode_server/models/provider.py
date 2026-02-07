"""Provider, model, and mode related models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import ModelRef  # noqa: TC001


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo


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

    @classmethod
    def from_tokonomics(cls, model: TokoModelInfo) -> Self:
        """Convert a tokonomics ModelInfo to an OpenCode Model."""
        # Convert pricing (tokonomics uses per-token, OpenCode uses per-million-token)
        from tokonomics.model_discovery.model_info import ModelPricing

        pricing = model.pricing or ModelPricing()
        cost = ModelCost(
            input=(pricing.prompt * 1_000_000) if pricing.prompt else 0.0,
            output=(pricing.completion * 1_000_000) if pricing.completion else 0.0,
            cache_read=(pricing.input_cache_read * 1_000_000) if pricing.input_cache_read else None,
            cache_write=(pricing.input_cache_write * 1_000_000)
            if pricing.input_cache_write
            else None,
        )
        # Convert limits
        context = float(model.context_window) if model.context_window else 128000.0
        output = float(model.max_output_tokens) if model.max_output_tokens else 4096.0
        limit = ModelLimit(context=context, output=output)
        # Determine capabilities from modalities and metadata
        has_vision = "image" in model.input_modalities
        has_reasoning = "reasoning" in model.output_modalities or "thinking" in model.name.lower()
        # Format release date if available
        release_date = model.created_at.strftime("%Y-%m-%d") if model.created_at else ""
        # Use id_override if available (e.g., "opus" for Claude Code SDK)
        model_id = model.id_override or model.id
        return cls(
            id=model_id,
            name=model.name,
            attachment=has_vision,
            cost=cost,
            limit=limit,
            reasoning=has_reasoning,
            release_date=release_date,
            temperature=True,
        )


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


class Mode(OpenCodeBaseModel):
    """Agent mode configuration."""

    name: str
    tools: dict[str, bool] = Field(default_factory=dict)
    model: ModelRef | None = None
    prompt: str | None = None
    temperature: float | None = None
