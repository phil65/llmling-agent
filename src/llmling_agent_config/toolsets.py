"""Models for toolsets."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Literal

from llmling import ConfigModel
from llmling.tools.toolsets import ToolSet
from llmling.utils.importing import import_class
from pydantic import EmailStr, Field, HttpUrl, SecretStr, field_validator  # noqa: TC002
from upath import UPath  # noqa: TC002


if TYPE_CHECKING:
    from llmling_agent.resource_providers.base import ResourceProvider


class BaseToolsetConfig(ConfigModel):
    """Base configuration for toolsets."""

    namespace: str | None = Field(default=None)
    """Optional namespace prefix for tool names"""


class OpenAPIToolsetConfig(BaseToolsetConfig):
    """Configuration for OpenAPI toolsets."""

    type: Literal["openapi"] = Field("openapi", init=False)
    """OpenAPI toolset."""

    spec: UPath = Field(...)
    """URL or path to the OpenAPI specification document."""

    base_url: HttpUrl | None = None
    """Optional base URL for API requests, overrides the one in spec."""

    def get_provider(self) -> ResourceProvider:
        """Create OpenAPI tools provider from this config."""
        from llmling_agent_toolsets.openapi import OpenAPITools

        return OpenAPITools(
            spec=self.spec,
            base_url=str(self.base_url) if self.base_url else "",
        )


class EntryPointToolsetConfig(BaseToolsetConfig):
    """Configuration for entry point toolsets."""

    type: Literal["entry_points"] = Field("entry_points", init=False)
    """Entry point toolset."""

    module: str = Field(...)
    """Python module path to load tools from via entry points."""

    def get_provider(self) -> ResourceProvider:
        """Create provider from this config."""
        from llmling_agent_toolsets.entry_points import EntryPointTools

        return EntryPointTools(module=self.module)


class ComposioToolSetConfig(BaseToolsetConfig):
    """Configuration for Composio toolsets."""

    type: Literal["composio"] = Field("composio", init=False)
    """Composio Toolsets."""

    api_key: SecretStr | None = None
    """Composio API Key."""

    user_id: EmailStr = "user@example.com"
    """User ID for composio tools."""

    toolsets: list[str] = Field(default_factory=list)
    """List of toolsets to load."""

    def get_provider(self) -> ResourceProvider:
        """Create provider from this config."""
        from llmling_agent_toolsets.composio_toolset import ComposioTools

        key = (
            self.api_key.get_secret_value()
            if self.api_key
            else os.getenv("COMPOSIO_API_KEY")
        )
        return ComposioTools(user_id=self.user_id, toolsets=self.toolsets, api_key=key)


class UpsonicToolSetConfig(BaseToolsetConfig):
    """Configuration for Upsonic toolsets."""

    type: Literal["upsonic"] = Field("upsonic", init=False)
    """Upsonic Toolsets."""

    base_url: HttpUrl | None = None
    """Upsonic API URL."""

    api_key: SecretStr | None = None
    """Upsonic API Key."""

    entity_id: str = "default"
    """Toolset entity id."""

    def get_provider(self) -> ResourceProvider:
        """Create provider from this config."""
        from llmling_agent_toolsets.upsonic_toolset import UpsonicTools

        return UpsonicTools(
            base_url=str(self.base_url) if self.base_url else None, api_key=self.api_key
        )


class AgentManagementToolsetConfig(BaseToolsetConfig):
    """Configuration for agent management toolset."""

    type: Literal["agent_management"] = Field("agent_management", init=False)
    """Agent management toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create agent management tools provider."""
        from llmling_agent_toolsets.builtin import AgentManagementTools

        return AgentManagementTools(name="agent_management")


class FileAccessToolsetConfig(BaseToolsetConfig):
    """Configuration for file access toolset."""

    type: Literal["file_access"] = Field("file_access", init=False)
    """File access toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create file access tools provider."""
        from llmling_agent_toolsets.builtin import FileAccessTools

        return FileAccessTools(name="file_access")


class CodeExecutionToolsetConfig(BaseToolsetConfig):
    """Configuration for code execution toolset."""

    type: Literal["code_execution"] = Field("code_execution", init=False)
    """Code execution toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create code execution tools provider."""
        from llmling_agent_toolsets.builtin import CodeExecutionTools

        return CodeExecutionTools(name="code_execution")


class ProcessManagementToolsetConfig(BaseToolsetConfig):
    """Configuration for process management toolset."""

    type: Literal["process_management"] = Field("process_management", init=False)
    """Process management toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create process management tools provider."""
        from llmling_agent_toolsets.builtin import ProcessManagementTools

        return ProcessManagementTools(name="process_management")


class ResourceAccessToolsetConfig(BaseToolsetConfig):
    """Configuration for resource access toolset."""

    type: Literal["resource_access"] = Field("resource_access", init=False)
    """Resource access toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create resource access tools provider."""
        from llmling_agent_toolsets.builtin import ResourceAccessTools

        return ResourceAccessTools(name="resource_access")


class ToolManagementToolsetConfig(BaseToolsetConfig):
    """Configuration for tool management toolset."""

    type: Literal["tool_management"] = Field("tool_management", init=False)
    """Tool management toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create tool management tools provider."""
        from llmling_agent_toolsets.builtin import ToolManagementTools

        return ToolManagementTools(name="tool_management")


class UserInteractionToolsetConfig(BaseToolsetConfig):
    """Configuration for user interaction toolset."""

    type: Literal["user_interaction"] = Field("user_interaction", init=False)
    """User interaction toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create user interaction tools provider."""
        from llmling_agent_toolsets.builtin import UserInteractionTools

        return UserInteractionTools(name="user_interaction")


class HistoryToolsetConfig(BaseToolsetConfig):
    """Configuration for history toolset."""

    type: Literal["history"] = Field("history", init=False)
    """History toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create history tools provider."""
        from llmling_agent_toolsets.builtin import HistoryTools

        return HistoryTools(name="history")


class IntegrationToolsetConfig(BaseToolsetConfig):
    """Configuration for integration toolset."""

    type: Literal["integrations"] = Field("integrations", init=False)
    """Integration toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create integration tools provider."""
        from llmling_agent_toolsets.builtin import IntegrationTools

        return IntegrationTools(name="integrations")


class CodeToolsetConfig(BaseToolsetConfig):
    """Configuration for code toolset."""

    type: Literal["code"] = Field("code", init=False)
    """Code toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create code tools provider."""
        from llmling_agent_toolsets.builtin.code import CodeTools

        return CodeTools(name="code")


class CustomToolsetConfig(BaseToolsetConfig):
    """Configuration for custom toolsets."""

    type: Literal["custom"] = Field("custom", init=False)
    """Custom toolset."""

    import_path: str = Field(...)
    """Dotted import path to the custom toolset implementation class."""

    @field_validator("import_path", mode="after")
    @classmethod
    def validate_import_path(cls, v: str) -> str:
        # v is already confirmed to be a str here
        try:
            cls = import_class(v)
            if not issubclass(cls, ToolSet):
                msg = f"{v} must be a ToolSet class"
                raise ValueError(msg)  # noqa: TRY004, TRY301
        except Exception as exc:
            msg = f"Invalid toolset class: {v}"
            raise ValueError(msg) from exc
        return v

    def get_provider(self) -> ResourceProvider:
        """Create custom provider from import path."""
        from llmling.utils.importing import import_class

        from llmling_agent.resource_providers.base import ResourceProvider

        provider_cls = import_class(self.import_path)
        if not issubclass(provider_cls, ResourceProvider):
            msg = f"{self.import_path} must be a ResourceProvider subclass"
            raise ValueError(msg)  # noqa: TRY004
        return provider_cls(name=provider_cls.__name__)


# Use discriminated union for toolset types
ToolsetConfig = Annotated[
    OpenAPIToolsetConfig
    | EntryPointToolsetConfig
    | ComposioToolSetConfig
    | UpsonicToolSetConfig
    | AgentManagementToolsetConfig
    | FileAccessToolsetConfig
    | CodeExecutionToolsetConfig
    | ProcessManagementToolsetConfig
    | ResourceAccessToolsetConfig
    | ToolManagementToolsetConfig
    | UserInteractionToolsetConfig
    | HistoryToolsetConfig
    | IntegrationToolsetConfig
    | CodeToolsetConfig
    | CustomToolsetConfig,
    Field(discriminator="type"),
]

if __name__ == "__main__":
    import upsonic

    tools = upsonic.Tiger().crewai
