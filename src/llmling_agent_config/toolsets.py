"""Models for toolsets."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Literal

from anyenv.code_execution.configs import ExecutionEnvironmentConfig
from llmling_models.configs.model_configs import AnyModelConfig
from pydantic import EmailStr, Field, HttpUrl, SecretStr
from schemez import Schema
from searchly.config import NewsSearchProviderConfig, WebSearchProviderConfig
from tokonomics import ModelName
from upath import UPath

from llmling_agent.utils.importing import import_class
from llmling_agent_config.converters import ConversionConfig


if TYPE_CHECKING:
    from llmling_agent.resource_providers import ResourceProvider


class BaseToolsetConfig(Schema):
    """Base configuration for toolsets."""

    namespace: str | None = Field(default=None, examples=["web", "files"], title="Tool namespace")
    """Optional namespace prefix for tool names"""


class OpenAPIToolsetConfig(BaseToolsetConfig):
    """Configuration for OpenAPI toolsets."""

    type: Literal["openapi"] = Field("openapi", init=False)
    """OpenAPI toolset."""

    spec: UPath = Field(
        examples=["https://api.example.com/openapi.json", "/path/to/spec.yaml"],
        title="OpenAPI specification",
    )
    """URL or path to the OpenAPI specification document."""

    base_url: HttpUrl | None = Field(
        default=None,
        examples=["https://api.example.com", "http://localhost:8080"],
        title="Base URL override",
    )
    """Optional base URL for API requests, overrides the one in spec."""

    def get_provider(self) -> ResourceProvider:
        """Create OpenAPI tools provider from this config."""
        from llmling_agent_toolsets.openapi import OpenAPITools

        base_url = str(self.base_url) if self.base_url else ""
        return OpenAPITools(spec=self.spec, base_url=base_url)


class EntryPointToolsetConfig(BaseToolsetConfig):
    """Configuration for entry point toolsets."""

    type: Literal["entry_points"] = Field("entry_points", init=False)
    """Entry point toolset."""

    module: str = Field(
        examples=["myapp.tools", "external_package.plugins"],
        title="Module path",
    )
    """Python module path to load tools from via entry points."""

    def get_provider(self) -> ResourceProvider:
        """Create provider from this config."""
        from llmling_agent_toolsets.entry_points import EntryPointTools

        return EntryPointTools(module=self.module)


class ComposioToolSetConfig(BaseToolsetConfig):
    """Configuration for Composio toolsets."""

    type: Literal["composio"] = Field("composio", init=False)
    """Composio Toolsets."""

    api_key: SecretStr | None = Field(default=None, title="Composio API key")
    """Composio API Key."""

    user_id: EmailStr = Field(
        default="user@example.com",
        examples=["user@example.com", "admin@company.com"],
        title="User ID",
    )
    """User ID for composio tools."""

    toolsets: list[str] = Field(
        default_factory=list,
        examples=[["github", "slack"], ["gmail", "calendar"]],
        title="Toolset list",
    )
    """List of toolsets to load."""

    def get_provider(self) -> ResourceProvider:
        """Create provider from this config."""
        from llmling_agent_toolsets.composio_toolset import ComposioTools

        key = self.api_key.get_secret_value() if self.api_key else os.getenv("COMPOSIO_API_KEY")
        return ComposioTools(user_id=self.user_id, toolsets=self.toolsets, api_key=key)


class UpsonicToolSetConfig(BaseToolsetConfig):
    """Configuration for Upsonic toolsets."""

    type: Literal["upsonic"] = Field("upsonic", init=False)
    """Upsonic Toolsets."""

    base_url: HttpUrl | None = Field(
        default=None,
        examples=["https://api.upsonic.co", "http://localhost:9000"],
        title="Upsonic API URL",
    )
    """Upsonic API URL."""

    api_key: SecretStr | None = Field(default=None, title="Upsonic API key")
    """Upsonic API Key."""

    entity_id: str = Field(
        default="default",
        examples=["default", "team1", "project_alpha"],
        title="Entity ID",
    )
    """Toolset entity id."""

    def get_provider(self) -> ResourceProvider:
        """Create provider from this config."""
        from llmling_agent_toolsets.upsonic_toolset import UpsonicTools

        base_url = str(self.base_url) if self.base_url else None
        return UpsonicTools(base_url=base_url, api_key=self.api_key)


class AgentManagementToolsetConfig(BaseToolsetConfig):
    """Configuration for agent pool building tools."""

    type: Literal["agent_management"] = Field("agent_management", init=False)
    """Agent pool building toolset (create_worker_agent, add_agent, add_team, connect_nodes)."""

    def get_provider(self) -> ResourceProvider:
        """Create agent management tools provider."""
        from llmling_agent_toolsets.builtin import AgentManagementTools

        return AgentManagementTools(name="agent_management")


class SubagentToolsetConfig(BaseToolsetConfig):
    """Configuration for subagent interaction tools."""

    type: Literal["subagent"] = Field("subagent", init=False)
    """Subagent interaction toolset (delegate_to, ask_agent, list_available_agents/teams)."""

    def get_provider(self) -> ResourceProvider:
        """Create subagent tools provider."""
        from llmling_agent_toolsets.builtin.subagent_tools import SubagentTools

        return SubagentTools(name="subagent_tools")


class ExecutionEnvironmentToolsetConfig(BaseToolsetConfig):
    """Configuration for execution environment toolset (code + process management)."""

    type: Literal["execution"] = Field("execution", init=False)
    """Execution environment toolset."""

    environment: ExecutionEnvironmentConfig | None = Field(
        default=None,
        title="Execution environment",
    )
    """Optional execution environment configuration (defaults to local)."""

    def get_provider(self) -> ResourceProvider:
        """Create execution environment tools provider."""
        from llmling_agent_toolsets.builtin import ExecutionEnvironmentTools

        env = self.environment.get_provider() if self.environment else None
        return ExecutionEnvironmentTools(env=env, name="execution")


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


class FSSpecToolsetConfig(BaseToolsetConfig):
    """Configuration for file access toolset (supports local and remote filesystems)."""

    type: Literal["file_access"] = Field("file_access", init=False)
    """File access toolset."""

    url: str | None = Field(
        default=None,
        examples=["file:///", "s3://my-bucket"],
        title="Filesystem URL",
    )
    """Filesystem URL or protocol. If None set, use agent default FS."""

    model: str | ModelName | AnyModelConfig | None = Field(
        default=None,
        examples=["openai:gpt-5-nano"],
        title="Model for edit sub-agent",
    )

    storage_options: dict[str, str] = Field(
        default_factory=dict,
        examples=[
            {"region": "us-east-1", "profile": "default"},
            {"token": "ghp_123456789", "timeout": "30"},
            {"key": "value", "ssl_verify": "true"},
        ],
        title="Storage options",
    )
    """Additional options to pass to the filesystem constructor."""

    conversion: ConversionConfig | None = Field(default=None, title="Conversion config")
    """Optional conversion configuration for markdown conversion."""

    def get_provider(self) -> ResourceProvider:
        """Create FSSpec filesystem tools provider."""
        import fsspec  # type: ignore[import-untyped]

        from llmling_agent.prompts.conversion_manager import ConversionManager
        from llmling_agent_toolsets.fsspec_toolset import FSSpecTools

        model = (
            self.model
            if isinstance(self.model, str) or self.model is None
            else self.model.get_model()
        )
        # Extract protocol name for the provider name
        if self.url:
            fs, _url_path = fsspec.url_to_fs(self.url, **self.storage_options)
        else:
            fs = None
        converter = ConversionManager(self.conversion) if self.conversion else None
        return FSSpecTools(fs, converter=converter, edit_model=model)


class VFSToolsetConfig(BaseToolsetConfig):
    """Configuration for VFS registry filesystem toolset."""

    type: Literal["vfs"] = Field("vfs", init=False)
    """VFS registry filesystem toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create VFS registry filesystem tools provider."""
        from llmling_agent_toolsets.vfs_toolset import VFSTools

        return VFSTools(name="vfs")


class SearchToolsetConfig(BaseToolsetConfig):
    """Configuration for web/news search toolset."""

    type: Literal["search"] = Field("search", init=False)
    """Search toolset."""

    web_search: WebSearchProviderConfig | None = Field(default=None, title="Web search")
    """Web search provider configuration."""

    news_search: NewsSearchProviderConfig | None = Field(default=None, title="News search")
    """News search provider configuration."""

    def get_provider(self) -> ResourceProvider:
        """Create search tools provider."""
        from llmling_agent_toolsets.search_toolset import SearchTools

        web = self.web_search.get_provider() if self.web_search else None
        news = self.news_search.get_provider() if self.news_search else None
        return SearchTools(web_search=web, news_search=news)


class NotificationsToolsetConfig(BaseToolsetConfig):
    """Configuration for Apprise-based notifications toolset."""

    type: Literal["notifications"] = Field("notifications", init=False)
    """Notifications toolset."""

    channels: dict[str, str | list[str]] = Field(
        default_factory=dict,
        examples=[
            {
                "team_slack": "slack://TokenA/TokenB/TokenC/",
                "personal": "tgram://bottoken/ChatID",
                "ops_alerts": ["slack://ops/", "mailto://ops@company.com"],
            }
        ],
        title="Notification channels",
    )
    """Named notification channels. Values can be a single Apprise URL or list of URLs."""

    def get_provider(self) -> ResourceProvider:
        """Create notifications tools provider."""
        from llmling_agent_toolsets.notifications import NotificationsTools

        return NotificationsTools(channels=self.channels)


class SemanticMemoryToolsetConfig(BaseToolsetConfig):
    """Configuration for semantic memory / knowledge processing toolset."""

    type: Literal["semantic_memory"] = Field("semantic_memory", init=False)
    """Semantic memory toolset using TypeAgent's KnowPro."""

    model: str | ModelName | AnyModelConfig | None = Field(
        default=None,
        examples=["openai:gpt-4o", "anthropic:claude-sonnet-4-20250514"],
        title="Model for LLM sampling",
    )
    """Model to use for query translation and answer generation."""

    dbname: str | None = Field(
        default=None,
        examples=["knowledge.db", "/path/to/memory.db"],
        title="Database path",
    )
    """SQLite database path for persistent storage, or None for in-memory."""

    def get_provider(self) -> ResourceProvider:
        """Create semantic memory tools provider."""
        from llmling_agent_toolsets.semantic_memory_toolset import SemanticMemoryTools

        model = (
            self.model
            if isinstance(self.model, str) or self.model is None
            else self.model.get_model()
        )
        return SemanticMemoryTools(model=model, dbname=self.dbname)


class CustomToolsetConfig(BaseToolsetConfig):
    """Configuration for custom toolsets."""

    type: Literal["custom"] = Field("custom", init=False)
    """Custom toolset."""

    import_path: str = Field(
        examples=["myapp.toolsets.CustomTools", "external.providers:MyProvider"],
        title="Import path",
    )
    """Dotted import path to the custom toolset implementation class."""

    def get_provider(self) -> ResourceProvider:
        """Create custom provider from import path."""
        from llmling_agent.resource_providers import ResourceProvider

        provider_cls = import_class(self.import_path)
        if not issubclass(provider_cls, ResourceProvider):
            raise ValueError(f"{self.import_path} must be a ResourceProvider subclass")  # noqa: TRY004
        return provider_cls(name=provider_cls.__name__)


class CodeModeToolsetConfig(BaseToolsetConfig):
    """Configuration for code mode tools."""

    type: Literal["code_mode"] = Field("code_mode", init=False)
    """Code mode toolset."""

    toolsets: list[ToolsetConfig] = Field(title="Wrapped toolsets")
    """List of toolsets to expose as a codemode toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create Codemode toolset."""
        from llmling_agent.resource_providers.codemode import CodeModeResourceProvider

        providers = [p.get_provider() for p in self.toolsets]
        return CodeModeResourceProvider(providers=providers)


class RemoteCodeModeToolsetConfig(BaseToolsetConfig):
    """Configuration for code mode tools."""

    type: Literal["remote_code_mode"] = Field("remote_code_mode", init=False)
    """Code mode toolset."""

    environment: ExecutionEnvironmentConfig = Field(title="Execution environment")
    """Execution environment configuration."""

    toolsets: list[ToolsetConfig] = Field(title="Wrapped toolsets")
    """List of toolsets to expose as a codemode toolset."""

    def get_provider(self) -> ResourceProvider:
        """Create Codemode toolset."""
        from llmling_agent.resource_providers.codemode import RemoteCodeModeResourceProvider

        providers = [p.get_provider() for p in self.toolsets]
        return RemoteCodeModeResourceProvider(
            providers=providers,
            execution_config=self.environment,
        )


ToolsetConfig = Annotated[
    OpenAPIToolsetConfig
    | EntryPointToolsetConfig
    | ComposioToolSetConfig
    | UpsonicToolSetConfig
    | AgentManagementToolsetConfig
    | ExecutionEnvironmentToolsetConfig
    | ToolManagementToolsetConfig
    | UserInteractionToolsetConfig
    | HistoryToolsetConfig
    | IntegrationToolsetConfig
    | CodeToolsetConfig
    | FSSpecToolsetConfig
    | VFSToolsetConfig
    | SubagentToolsetConfig
    | CodeModeToolsetConfig
    | RemoteCodeModeToolsetConfig
    | SearchToolsetConfig
    | NotificationsToolsetConfig
    | SemanticMemoryToolsetConfig
    | CustomToolsetConfig,
    Field(discriminator="type"),
]

if __name__ == "__main__":
    import upsonic  # type: ignore[import-untyped]

    tools = upsonic.Tiger().crewai
