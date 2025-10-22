"""Prompt models for agent configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic.networks import HttpUrl


class BasePromptHubConfig(BaseModel):
    """Configuration for prompt providers."""

    type: str = Field(init=False)
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class PromptLayerConfig(BasePromptHubConfig):
    """Configuration for PromptLayer prompt provider."""

    type: Literal["promptlayer"] = Field("promptlayer", init=False)
    """Configuration for PromptLayer prompt provider."""

    api_key: SecretStr
    """API key for the PromptLayer API."""


class OpenLITConfig(BasePromptHubConfig):
    """Configuration for OpenLIT prompt provider."""

    type: Literal["openlit"] = Field("openlit", init=False)
    """Configuration for OpenLIT prompt provider."""

    url: HttpUrl | None = None  # Optional, defaults to OPENLIT_URL env var
    """URL of the OpenLIT API."""

    api_key: SecretStr | None = None  # Optional, defaults to OPENLIT_API_KEY env var
    """API key for the OpenLIT API."""


class LangfuseConfig(BasePromptHubConfig):
    """Configuration for Langfuse prompt provider."""

    type: Literal["langfuse"] = Field("langfuse", init=False)
    """Configuration for Langfuse prompt provider."""

    secret_key: SecretStr
    """Secret key for the Langfuse API."""

    public_key: SecretStr
    """Public key for the Langfuse API."""

    host: HttpUrl = HttpUrl("https://cloud.langfuse.com")
    """Langfuse host address."""

    cache_ttl_seconds: int = Field(default=60, ge=0)
    """Cache TTL for responses in seconds."""

    max_retries: int = Field(default=2, ge=0)
    """Maximum number of retries for failed requests."""

    fetch_timeout_seconds: int = Field(default=20, ge=0)
    """Timeout for fetching responses in seconds."""


class TraceloopConfig(BasePromptHubConfig):
    """Configuration for Traceloop prompt provider."""

    type: Literal["traceloop"] = Field("traceloop", init=False)
    """Configuration for Traceloop prompt provider."""

    api_key: SecretStr | None = None  # Optional, defaults to TRACELOOP_API_KEY env var
    """API key for the Traceloop API."""


class BraintrustConfig(BasePromptHubConfig):
    """Configuration for Braintrust prompt provider."""

    type: Literal["braintrust"] = Field("braintrust", init=False)
    """Configuration for Braintrust prompt provider."""

    api_key: SecretStr | None = None  # Optional, defaults to BRAINTRUST_API_KEY env var
    """API key for the Braintrust API."""

    project: str | None = None
    """Braintrust Project name."""


class FabricConfig(BasePromptHubConfig):
    """Configuration for Fabric GitHub prompt provider."""

    type: Literal["fabric"] = Field("fabric", init=False)
    """Configuration for Fabric GitHub prompt provider."""


PromptHubConfig = Annotated[
    PromptLayerConfig
    | OpenLITConfig
    | LangfuseConfig
    | TraceloopConfig
    | FabricConfig
    | BraintrustConfig,
    Field(discriminator="type"),
]
