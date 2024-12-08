"""Models for agent configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from llmling.config.store import ConfigStore
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from pydantic_ai import models  # noqa: TC002
import yamling


if TYPE_CHECKING:
    import os


class ResponseField(BaseModel):
    """Field definition for agent responses."""

    type: str
    """Data type of the response field"""
    description: str | None = None
    """Optional description of what this field represents"""
    constraints: dict[str, Any] | None = None
    """Optional validation constraints for the field"""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
    )


class ResponseDefinition(BaseModel):
    """Definition of an agent response type."""

    description: str | None = None
    """Optional description of the response type"""
    fields: dict[str, ResponseField]
    """Mapping of field names to their definitions"""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
    )


class SystemPrompt(BaseModel):
    """System prompt configuration."""

    type: Literal["text", "function", "template"]
    """Type of system prompt: static text, function call, or template"""
    value: str
    """The prompt text, function path, or template string"""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
    )


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    name: str | None = None
    """Name of the agent"""

    description: str | None = None
    """Optional description of the agent's purpose"""

    model: models.Model | models.KnownModelName | None = None
    """The LLM model to use"""

    environment: str | None = None
    """Path or name of the environment configuration to use"""

    result_type: str
    """Name of the response definition to use"""

    system_prompts: list[str] = Field(default_factory=list)
    """System prompts for the agent"""

    user_prompts: list[str] = Field(default_factory=list)
    """Default user prompts for the agent"""

    model_settings: dict[str, Any] = Field(default_factory=dict)
    """Additional settings to pass to the model"""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )

    @field_validator("environment")
    @classmethod
    def resolve_environment(cls, env: str, info: ValidationInfo) -> str:
        """Try to resolve environment as name first, then as path."""
        try:
            config_store = ConfigStore()
            return config_store.get_config(env)
        except KeyError:
            # If not found, treat as direct path
            return env


class AgentDefinition(BaseModel):
    """Complete agent definition including responses."""

    responses: dict[str, ResponseDefinition]
    """Mapping of response names to their definitions"""
    agents: dict[str, AgentConfig]
    """Mapping of agent IDs to their configurations"""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Self:
        """Load agent configuration from YAML file."""
        try:
            data = yamling.load_yaml_file(path)
            return cls.model_validate(data)
        except Exception as exc:
            msg = f"Failed to load agent config from {path}"
            raise ValueError(msg) from exc
