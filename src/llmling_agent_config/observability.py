"""Configuration models for observability providers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, PrivateAttr, SecretStr
from schemez import Schema


class BaseObservabilityConfig(Schema):
    """Base configuration for observability endpoints."""

    type: str
    enabled: bool = True

    # Standard OTEL settings
    service_name: str | None = None
    environment: str | None = None
    protocol: Literal["http/protobuf", "grpc", "http/json"] = "http/protobuf"


class LogfireObservabilityConfig(BaseObservabilityConfig):
    """Configuration for Logfire endpoint."""

    type: Literal["logfire"] = "logfire"
    token: SecretStr | None = None
    region: Literal["us", "eu"] = "us"

    # Private - computed from region
    _endpoint: str = PrivateAttr()
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context, /) -> None:
        """Compute private attributes from user config."""
        endpoint = (
            "https://logfire-eu.pydantic.dev"
            if self.region == "eu"
            else "https://logfire-us.pydantic.dev"
        )
        object.__setattr__(self, "_endpoint", endpoint)

        if self.token:
            headers = {"Authorization": f"Bearer {self.token.get_secret_value()}"}
            object.__setattr__(self, "_headers", headers)


class LangsmithObservabilityConfig(BaseObservabilityConfig):
    """Configuration for Langsmith endpoint."""

    type: Literal["langsmith"] = "langsmith"
    api_key: SecretStr | None = None
    project_name: str | None = None

    _endpoint: str = PrivateAttr(default="https://api.smith.langchain.com")
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context, /) -> None:
        """Compute private attributes from user config."""
        if self.api_key:
            headers = {"x-api-key": self.api_key.get_secret_value()}
            object.__setattr__(self, "_headers", headers)


class AgentOpsObservabilityConfig(BaseObservabilityConfig):
    """Configuration for AgentOps endpoint."""

    type: Literal["agentops"] = "agentops"
    api_key: SecretStr | None = None

    _endpoint: str = PrivateAttr(default="https://api.agentops.ai")
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context, /) -> None:
        """Compute private attributes from user config."""
        if self.api_key:
            headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
            object.__setattr__(self, "_headers", headers)


class ArizePhoenixObservabilityConfig(BaseObservabilityConfig):
    """Configuration for Arize Phoenix endpoint."""

    type: Literal["arize"] = "arize"
    api_key: SecretStr | None = None
    space_key: str | None = None
    model_id: str | None = None

    _endpoint: str = PrivateAttr(default="https://api.arize.com")
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context, /) -> None:
        """Compute private attributes from user config."""
        if self.api_key:
            headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
            object.__setattr__(self, "_headers", headers)


class CustomObservabilityConfig(BaseObservabilityConfig):
    """Configuration for custom OTEL endpoint."""

    type: Literal["custom"] = "custom"
    endpoint: str
    headers: dict[str, str] = Field(default_factory=dict)


# Union of all provider configs
ObservabilityProviderConfig = Annotated[
    LogfireObservabilityConfig
    | LangsmithObservabilityConfig
    | AgentOpsObservabilityConfig
    | ArizePhoenixObservabilityConfig
    | CustomObservabilityConfig,
    Field(discriminator="type"),
]


class ObservabilityConfig(Schema):
    """Global observability configuration - supports single backend only."""

    enabled: bool = True
    """Whether observability is enabled."""

    provider: ObservabilityProviderConfig | None = None
    """Single observability provider configuration."""
