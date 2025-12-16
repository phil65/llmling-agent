"""Configuration models for observability providers."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr, SecretStr
from schemez import Schema


class BaseObservabilityConfig(Schema):
    """Base configuration for observability endpoints."""

    type: str = Field(title="Observability provider type")
    """Observability provider type."""

    enabled: bool = Field(default=True, title="Provider enabled")
    """Provider enabled state."""

    service_name: str | None = Field(
        default=None,
        examples=["llmling-agent", "my-ai-service", "production-bot"],
        title="Service name",
    )
    environment: str | None = Field(
        default=None,
        examples=["production", "staging", "development"],
        title="Environment",
    )
    protocol: Literal["http/protobuf", "grpc", "http/json"] = Field(
        default="http/protobuf",
        examples=["http/protobuf", "grpc", "http/json"],
        title="Protocol",
    )


class LogfireObservabilityConfig(BaseObservabilityConfig):
    """Configuration for Logfire endpoint."""

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "Logfire"})

    type: Literal["logfire"] = "logfire"
    """Logfire endpoint type."""

    token: SecretStr | None = Field(default=None, title="Logfire token")
    """Logfire token."""

    region: Literal["us", "eu"] = Field(default="us", examples=["us", "eu"], title="Region")
    """Logfire region."""

    # Private - computed from region
    _endpoint: str = PrivateAttr()
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
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

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "Langsmith"})

    type: Literal["langsmith"] = "langsmith"
    """Langsmith observability configuration."""

    api_key: SecretStr | None = Field(default=None, title="Langsmith API key")
    """Langsmith API key."""

    project_name: str | None = Field(
        default=None,
        examples=["my-project", "ai-agents", "production"],
        title="Project name",
    )

    _endpoint: str = PrivateAttr(default="https://api.smith.langchain.com")
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Compute private attributes from user config."""
        if self.api_key:
            headers = {"x-api-key": self.api_key.get_secret_value()}
            object.__setattr__(self, "_headers", headers)


class AgentOpsObservabilityConfig(BaseObservabilityConfig):
    """Configuration for AgentOps endpoint."""

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "AgentOps"})

    type: Literal["agentops"] = "agentops"
    """AgentOps observability configuration."""

    api_key: SecretStr | None = Field(default=None, title="AgentOps API key")
    """AgentOps API key."""

    _endpoint: str = PrivateAttr(default="https://api.agentops.ai")
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Compute private attributes from user config."""
        if self.api_key:
            headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
            object.__setattr__(self, "_headers", headers)


class ArizePhoenixObservabilityConfig(BaseObservabilityConfig):
    """Configuration for Arize Phoenix endpoint."""

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "Arize Phoenix"})

    type: Literal["arize"] = "arize"
    """Arize observability configuration."""

    api_key: SecretStr | None = Field(default=None, title="Arize API key")
    """Arize API key."""

    space_key: str | None = Field(
        default=None,
        examples=["default", "team-space", "production-space"],
        title="Space key",
    )
    """Arize space key."""

    model_id: str | None = Field(
        default=None,
        examples=["gpt-4", "claude-3", "my-model"],
        title="Model ID",
    )
    """Arize model ID."""

    _endpoint: str = PrivateAttr(default="https://api.arize.com")
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Compute private attributes from user config."""
        if self.api_key:
            headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
            object.__setattr__(self, "_headers", headers)


class CustomObservabilityConfig(BaseObservabilityConfig):
    """Configuration for custom OTEL endpoint."""

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "Custom OTEL"})

    type: Literal["custom"] = "custom"
    """Custom OTEL endpoint configuration."""

    endpoint: str = Field(
        examples=["https://otel.example.com", "http://localhost:4318"],
        title="OTEL endpoint",
    )
    """Custom OTEL endpoint URL."""

    headers: dict[str, str] = Field(default_factory=dict, title="Custom headers")
    """Custom headers for the OTEL endpoint."""


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

    enabled: bool = Field(default=True, title="Observability enabled")
    """Whether observability is enabled."""

    provider: ObservabilityProviderConfig | None = Field(
        default=None,
        title="Observability provider",
    )
    """Single observability provider configuration."""
