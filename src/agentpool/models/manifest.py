"""Models for agent configuration."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import ConfigDict, Field, model_validator
from schemez import Schema
from upathtools.configs import FilesystemConfigType
from upathtools.configs.base import URIFileSystemConfig

from agentpool import log
from agentpool.models.acp_agents import ACPAgentConfigTypes
from agentpool.models.agents import NativeAgentConfig
from agentpool.models.agui_agents import AGUIAgentConfig
from agentpool.models.claude_code_agents import ClaudeCodeAgentConfig
from agentpool.models.file_agents import FileAgentConfig
from agentpool_config.commands import CommandConfig, StaticCommandConfig
from agentpool_config.converters import ConversionConfig
from agentpool_config.mcp_server import BaseMCPServerConfig, MCPServerConfig
from agentpool_config.observability import ObservabilityConfig
from agentpool_config.output_types import StructuredResponseConfig
from agentpool_config.pool_server import MCPPoolServerConfig
from agentpool_config.storage import StorageConfig
from agentpool_config.system_prompts import PromptLibraryConfig
from agentpool_config.task import Job
from agentpool_config.teams import TeamConfig
from agentpool_config.workers import (
    ACPAgentWorkerConfig,
    AgentWorkerConfig,
    AGUIAgentWorkerConfig,
    BaseWorkerConfig,
    TeamWorkerConfig,
)


if TYPE_CHECKING:
    from upathtools import JoinablePathLike

    from agentpool.models.acp_agents import BaseACPAgentConfig
    from agentpool.prompts.manager import PromptManager
    from agentpool.vfs_registry import VFSRegistry

logger = log.get_logger(__name__)


# Model union with discriminator for typed configs
_FileSystemConfigUnion = Annotated[
    FilesystemConfigType | URIFileSystemConfig,
    Field(discriminator="type"),
]

# Final type allowing models or URI shorthand string
ResourceConfig = _FileSystemConfigUnion | str

# Unified agent config type with top-level discriminator
AnyAgentConfig = Annotated[
    NativeAgentConfig | AGUIAgentConfig | ClaudeCodeAgentConfig | ACPAgentConfigTypes,
    Field(discriminator="type"),
]


class AgentsManifest(Schema):
    """Complete agent configuration manifest defining all available agents.

    This is the root configuration that:
    - Defines available response types (both inline and imported)
    - Configures all agent instances and their settings
    - Sets up custom role definitions and capabilities
    - Manages environment configurations

    A single manifest can define multiple agents that can work independently
    or collaborate through the orchestrator.
    """

    INHERIT: str | list[str] | None = None
    """Inheritance references."""

    resources: dict[str, ResourceConfig] = Field(
        default_factory=dict,
        examples=[
            {"docs": "file://./docs", "data": "s3://bucket/data"},
            {
                "api": {
                    "type": "uri",
                    "uri": "https://api.example.com",
                    "cached": True,
                }
            },
        ],
    )
    """Resource configurations defining available filesystems.

    Supports both full config and URI shorthand:
        resources:
          docs: "file://./docs"  # shorthand
          data:  # full config
            type: "uri"
            uri: "s3://bucket/data"
            cached: true
    """

    agents: dict[str, AnyAgentConfig] = Field(
        default_factory=dict,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/agentpool/YAML%20Configuration/agent_configuration/"
        },
    )
    """Mapping of agent IDs to their configurations.

    All agent types are unified under this single dict, discriminated by the 'type' field:
    - type: "native" (default) - pydantic-ai based agents
    - type: "agui" - AG-UI protocol agents
    - type: "claude_code" - Claude Agent SDK agents
    - type: "acp" - ACP protocol agents (further discriminated by 'provider')

    Example:
        ```yaml
        agents:
          assistant:
            type: native
            model: openai:gpt-4
            system_prompts:
              - "You are a helpful assistant."

          coder:
            type: claude_code
            cwd: /path/to/project
            model: claude-sonnet-4-5

          orchestrator:
            type: acp
            provider: claude
            model: sonnet

          remote:
            type: agui
            endpoint: http://localhost:8000/agent/run
        ```

    Docs: https://phil65.github.io/agentpool/YAML%20Configuration/agent_configuration/
    """

    file_agents: dict[str, str | FileAgentConfig] = Field(
        default_factory=dict,
        examples=[
            {
                "code_reviewer": ".claude/agents/reviewer.md",
                "debugger": "https://example.com/agents/debugger.md",
                "custom": {"type": "opencode", "path": "./agents/custom.md"},
            }
        ],
    )
    """Mapping of agent IDs to file-based agent definitions.

    Supports both simple path strings (auto-detect format) and explicit config
    with type discriminator.
    Files must have YAML frontmatter in Claude Code, OpenCode, or AgentPool format.
    The markdown body becomes the system prompt.

    Formats:
      - claude: name, description, tools (comma-separated), model, permissionMode
      - opencode: description, mode, model, temperature, maxSteps, tools (dict)
      - native: Full NativeAgentConfig fields in frontmatter

    Example:
        ```yaml
        file_agents:
          reviewer: .claude/agents/reviewer.md  # auto-detect
          debugger:
            type: opencode  # explicit type
            path: ./agents/debugger.md
        ```
    """

    teams: dict[str, TeamConfig] = Field(
        default_factory=dict,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/agentpool/YAML%20Configuration/team_configuration/"
        },
    )
    """Mapping of team IDs to their configurations.

    Docs: https://phil65.github.io/agentpool/YAML%20Configuration/team_configuration/
    """

    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/agentpool/YAML%20Configuration/storage_configuration/"
        },
    )
    """Storage provider configuration.

    Docs: https://phil65.github.io/agentpool/YAML%20Configuration/storage_configuration/
    """

    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    """Observability provider configuration."""

    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    """Document conversion configuration."""

    responses: dict[str, StructuredResponseConfig] = Field(
        default_factory=dict,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/agentpool/YAML%20Configuration/response_configuration/"
        },
    )
    """Mapping of response names to their definitions.

    Docs: https://phil65.github.io/agentpool/YAML%20Configuration/response_configuration/
    """

    jobs: dict[str, Job[Any]] = Field(default_factory=dict)
    """Pre-defined jobs, ready to be used by nodes."""

    mcp_servers: list[str | MCPServerConfig] = Field(
        default_factory=list,
        examples=[
            ["uvx some-server"],
            [{"type": "streamable-http", "url": "http://mcp.example.com"}],
        ],
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/agentpool/YAML%20Configuration/mcp_configuration/"
        },
    )
    """List of MCP server configurations:

    These MCP servers are used to provide tools and other resources to the nodes.

    Docs: https://phil65.github.io/agentpool/YAML%20Configuration/mcp_configuration/
    """
    pool_server: MCPPoolServerConfig = Field(default_factory=MCPPoolServerConfig)
    """Pool server configuration.

    This MCP server configuration is used for the pool MCP server,
    which exposes pool functionality to other applications / clients."""

    prompts: PromptLibraryConfig = Field(
        default_factory=PromptLibraryConfig,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/agentpool/YAML%20Configuration/prompt_configuration/"
        },
    )
    """Prompt library configuration.

    This configuration defines the prompt library, which is used to provide prompts to the nodes.

    Docs: https://phil65.github.io/agentpool/YAML%20Configuration/prompt_configuration/
    """

    commands: dict[str, CommandConfig | str] = Field(
        default_factory=dict,
        examples=[
            {"check_disk": "df -h", "analyze": "Analyze the current situation"},
            {
                "status": {
                    "type": "static",
                    "content": "Show system status",
                }
            },
        ],
    )
    """Global command shortcuts for prompt injection.

    Supports both shorthand string syntax and full command configurations:
        commands:
          df: "check disk space"  # shorthand -> StaticCommandConfig
          analyze:  # full config
            type: file
            path: "./prompts/analysis.md"
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-icon": "octicon:file-code-16",
            "x-doc-title": "Manifest Overview",
            "documentation_url": "https://phil65.github.io/agentpool/YAML%20Configuration/manifest_configuration/",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def set_default_agent_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Set default type='native' for agents without a type field."""
        agents = data.get("agents", {})
        for config in agents.values():
            if isinstance(config, dict) and "type" not in config:
                config["type"] = "native"
        return data

    @model_validator(mode="before")
    @classmethod
    def normalize_workers(cls, data: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
        """Convert string workers to appropriate WorkerConfig for all agents."""
        teams = data.get("teams", {})
        agents = data.get("agents", {})

        def get_agent_type(name: str) -> str | None:
            """Get the type of an agent by name from the unified agents dict."""
            if name not in agents:
                return None
            agent_cfg = agents[name]
            if isinstance(agent_cfg, dict):
                return str(agent_cfg.get("type", "native"))
            return str(getattr(agent_cfg, "type", "native"))

        # Process workers for all agents that have them (only dict configs need processing)
        for agent_name, agent_config in agents.items():
            if not isinstance(agent_config, dict):
                continue  # Already a model instance, skip
            workers = agent_config.get("workers", [])
            if workers:
                normalized: list[BaseWorkerConfig] = []

                for worker in workers:
                    match worker:
                        case str() as name if name in teams:
                            normalized.append(TeamWorkerConfig(name=name))
                        case str() as name:
                            # Determine worker config based on agent type
                            agent_type = get_agent_type(name)
                            match agent_type:
                                case "acp":
                                    normalized.append(ACPAgentWorkerConfig(name=name))
                                case "agui":
                                    normalized.append(AGUIAgentWorkerConfig(name=name))
                                case _:  # native, claude_code, or unknown
                                    normalized.append(AgentWorkerConfig(name=name))

                        case dict() as config:
                            # If type is explicitly specified, use it
                            if worker_type := config.get("type"):
                                match worker_type:
                                    case "team":
                                        normalized.append(TeamWorkerConfig(**config))
                                    case "agent":
                                        normalized.append(AgentWorkerConfig(**config))
                                    case "acp_agent":
                                        normalized.append(ACPAgentWorkerConfig(**config))
                                    case "agui_agent":
                                        normalized.append(AGUIAgentWorkerConfig(**config))
                                    case _:
                                        msg = f"Invalid worker type: {worker_type}"
                                        raise ValueError(msg)
                            else:
                                # Determine type based on worker name
                                worker_name = config.get("name")
                                if not worker_name:
                                    msg = "Worker config missing name"
                                    raise ValueError(msg)

                                if worker_name in teams:
                                    normalized.append(TeamWorkerConfig(**config))
                                else:
                                    agent_type = get_agent_type(worker_name)
                                    match agent_type:
                                        case "acp":
                                            normalized.append(ACPAgentWorkerConfig(**config))
                                        case "agui":
                                            normalized.append(AGUIAgentWorkerConfig(**config))
                                        case _:
                                            normalized.append(AgentWorkerConfig(**config))

                        case BaseWorkerConfig():  # Already normalized
                            normalized.append(worker)

                        case _:
                            msg = f"Invalid worker configuration: {worker}"
                            raise ValueError(msg)

                if isinstance(agent_config, dict):
                    agent_config["workers"] = normalized
                else:  # Need to create a new dict with updated workers
                    agent_dict = agent_config.model_dump()
                    agent_dict["workers"] = normalized
                    agents[agent_name] = agent_dict

        return data

    @cached_property
    def vfs_registry(self) -> VFSRegistry:
        """Get registry with all configured VFS resources."""
        from agentpool.vfs_registry import VFSRegistry

        registry = VFSRegistry()
        for name, config in self.resources.items():
            registry.register_from_config(name, config)
        return registry

    def clone_agent_config(
        self,
        name: str,
        new_name: str | None = None,
        *,
        template_context: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> str:
        """Create a copy of an agent configuration.

        Args:
            name: Name of agent to clone
            new_name: Optional new name (auto-generated if None)
            template_context: Variables for template rendering
            **overrides: Configuration overrides for the clone

        Returns:
            Name of the new agent

        Raises:
            KeyError: If original agent not found
            ValueError: If new name already exists or if overrides invalid
        """
        if name not in self.agents:
            msg = f"Agent {name} not found"
            raise KeyError(msg)

        actual_name = new_name or f"{name}_copy_{len(self.agents)}"
        if actual_name in self.agents:
            msg = f"Agent {actual_name} already exists"
            raise ValueError(msg)

        config = self.agents[name].model_copy(deep=True)
        for key, value in overrides.items():
            if not hasattr(config, key):
                msg = f"Invalid override: {key}"
                raise ValueError(msg)
            setattr(config, key, value)

        # Handle template rendering if context provided
        if template_context and "name" in template_context and "name" not in overrides:
            config.model_copy(update={"name": template_context["name"]})

        # Note: system_prompts will be rendered during agent creation, not here
        # config.system_prompts remains as PromptConfig objects
        self.agents[actual_name] = config
        return actual_name

    @model_validator(mode="before")
    @classmethod
    def resolve_inheritance(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Resolve agent inheritance chains."""
        nodes = data.get("agents", {})
        resolved: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()

        def resolve_node(name: str) -> dict[str, Any] | Any:
            if name in resolved:
                return resolved[name]

            node = nodes[name]
            # Skip model instances - they're already validated
            if not isinstance(node, dict):
                return node

            if name in seen:
                msg = f"Circular inheritance detected: {name}"
                raise ValueError(msg)

            seen.add(name)
            config = node.copy()
            inherit = config.get("inherits")
            if inherit:
                if inherit not in nodes:
                    msg = f"Parent agent {inherit} not found"
                    raise ValueError(msg)

                parent = resolve_node(inherit)  # Get resolved parent config
                if isinstance(parent, dict):
                    merged = parent.copy()
                    merged.update(config)
                    config = merged

            seen.remove(name)
            resolved[name] = config
            return config

        for name in nodes:
            resolved[name] = resolve_node(name)

        # Update nodes with resolved configs
        data["agents"] = resolved
        return data

    @cached_property
    def _loaded_file_agents(self) -> dict[str, NativeAgentConfig]:
        """Load and cache file-based agent configurations.

        Parses markdown files in Claude Code, OpenCode, or AgentPool format
        and converts them to NativeAgentConfig. Results are cached.
        """
        from agentpool.models.file_parsing import parse_file_agent_reference

        loaded: dict[str, NativeAgentConfig] = {}
        for name, reference in self.file_agents.items():
            try:
                config = parse_file_agent_reference(reference)
                # Ensure name is set from the key
                if config.name is None:
                    config = config.model_copy(update={"name": name})
                loaded[name] = config
            except Exception as e:
                path = reference if isinstance(reference, str) else reference.path
                logger.exception("Failed to load file agent %r from %s", name, path)
                msg = f"Failed to load file agent {name!r} from {path}: {e}"
                raise ValueError(msg) from e
        return loaded

    @property
    def node_names(self) -> list[str]:
        """Get list of all agent and team names."""
        return list(self.agents.keys()) + list(self.file_agents.keys()) + list(self.teams.keys())

    @property
    def nodes(self) -> dict[str, Any]:
        """Get all agent and team configurations."""
        return {**self.agents, **self._loaded_file_agents, **self.teams}

    @property
    def acp_agents(self) -> dict[str, BaseACPAgentConfig]:
        """Get ACP agents filtered from unified agents dict."""
        from agentpool.models.acp_agents import BaseACPAgentConfig

        return {k: v for k, v in self.agents.items() if isinstance(v, BaseACPAgentConfig)}

    @property
    def agui_agents(self) -> dict[str, AGUIAgentConfig]:
        """Get AG-UI agents filtered from unified agents dict."""
        return {k: v for k, v in self.agents.items() if isinstance(v, AGUIAgentConfig)}

    @property
    def claude_code_agents(self) -> dict[str, ClaudeCodeAgentConfig]:
        """Get Claude Code agents filtered from unified agents dict."""
        return {k: v for k, v in self.agents.items() if isinstance(v, ClaudeCodeAgentConfig)}

    @property
    def native_agents(self) -> dict[str, NativeAgentConfig]:
        """Get native agents filtered from unified agents dict."""
        return {k: v for k, v in self.agents.items() if isinstance(v, NativeAgentConfig)}

    def get_mcp_servers(self) -> list[MCPServerConfig]:
        """Get processed MCP server configurations.

        Converts string entries to appropriate MCP server configs based on heuristics:
        - URLs ending with "/sse" -> SSE server
        - URLs starting with http(s):// -> HTTP server
        - Everything else -> stdio command

        Returns:
            List of MCPServerConfig instances

        Raises:
            ValueError: If string entry is empty
        """
        return [
            BaseMCPServerConfig.from_string(cfg) if isinstance(cfg, str) else cfg
            for cfg in self.mcp_servers
        ]

    def get_command_configs(self) -> dict[str, CommandConfig]:
        """Get processed command configurations.

        Converts string entries to StaticCommandConfig instances.

        Returns:
            Dict mapping command names to CommandConfig instances
        """
        result: dict[str, CommandConfig] = {}
        for name, config in self.commands.items():
            if isinstance(config, str):
                result[name] = StaticCommandConfig(name=name, content=config)
            else:
                if config.name is None:  # Set name if not provided
                    config.name = name
                result[name] = config
        return result

    @cached_property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager for this manifest."""
        from agentpool.prompts.manager import PromptManager

        return PromptManager(self.prompts)

    # @model_validator(mode="after")
    # def validate_response_types(self) -> AgentsManifest:
    #     """Ensure all agent output_types exist in responses or are inline."""
    #     for agent_id, agent in self.agents.items():
    #         if (
    #             isinstance(agent.output_type, str)
    #             and agent.output_type not in self.responses
    #         ):
    #             msg = f"'{agent.output_type=}' for '{agent_id=}' not found in responses"
    #             raise ValueError(msg)
    #     return self

    @classmethod
    def from_file(cls, path: JoinablePathLike) -> Self:
        """Load agent configuration from YAML file.

        Args:
            path: Path to the configuration file

        Returns:
            Loaded agent definition

        Raises:
            ValueError: If loading fails
        """
        import yamling

        try:
            data = yamling.load_yaml_file(path, resolve_inherit=True)
            agent_def = cls.model_validate(data)
            path_str = str(path)

            def update_with_path(nodes: dict[str, Any]) -> dict[str, Any]:
                return {
                    name: config.model_copy(update={"config_file_path": path_str})
                    for name, config in nodes.items()
                }

            return agent_def.model_copy(
                update={
                    "agents": update_with_path(agent_def.agents),
                    "teams": update_with_path(agent_def.teams),
                }
            )
        except Exception as exc:
            msg = f"Failed to load agent config from {path}"
            raise ValueError(msg) from exc

    def get_output_type(self, agent_name: str) -> type[Any] | None:
        """Get the resolved result type for an agent.

        Returns None if no result type is configured or agent doesn't support output_type.
        """
        agent_config = self.agents[agent_name]
        # Only NativeAgentConfig and ClaudeCodeAgentConfig have output_type
        if not isinstance(agent_config, NativeAgentConfig | ClaudeCodeAgentConfig):
            return None
        if not agent_config.output_type:
            return None
        logger.debug("Building response model", type=agent_config.output_type)
        if isinstance(agent_config.output_type, str):
            response_def = self.responses[agent_config.output_type]
            return response_def.response_schema.get_schema()
        return agent_config.output_type.response_schema.get_schema()


if __name__ == "__main__":
    from llmling_models.configs import InputModelConfig

    model = InputModelConfig()
    agent_cfg = NativeAgentConfig(name="test_agent", model=model)
    manifest = AgentsManifest(agents=dict(test_agent=agent_cfg))
    print(AgentsManifest.generate_test_data(mode="maximal").model_dump_yaml())
