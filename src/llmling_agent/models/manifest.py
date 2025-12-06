"""Models for agent configuration."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import ConfigDict, Field, model_validator
from schemez import Schema
from upathtools.configs import FilesystemConfigType
from upathtools.configs.base import URIFileSystemConfig

from llmling_agent import log
from llmling_agent.models.acp_agents import ACPAgentConfigTypes
from llmling_agent.models.agents import AgentConfig, parse_agent_file
from llmling_agent.models.agui_agents import AGUIAgentConfig
from llmling_agent_config.commands import CommandConfig, StaticCommandConfig
from llmling_agent_config.converters import ConversionConfig
from llmling_agent_config.mcp_server import BaseMCPServerConfig, MCPServerConfig
from llmling_agent_config.observability import ObservabilityConfig
from llmling_agent_config.output_types import StructuredResponseConfig
from llmling_agent_config.pool_server import MCPPoolServerConfig
from llmling_agent_config.storage import StorageConfig
from llmling_agent_config.system_prompts import PromptLibraryConfig
from llmling_agent_config.task import Job
from llmling_agent_config.teams import TeamConfig
from llmling_agent_config.workers import (
    ACPAgentWorkerConfig,
    AgentWorkerConfig,
    AGUIAgentWorkerConfig,
    BaseWorkerConfig,
    TeamWorkerConfig,
)


if TYPE_CHECKING:
    from upath.types import JoinablePathLike

    from llmling_agent import Agent, AgentPool
    from llmling_agent.agent.acp_agent import ACPAgent
    from llmling_agent.agent.agui_agent import AGUIAgent
    from llmling_agent.common_types import IndividualEventHandler
    from llmling_agent.prompts.manager import PromptManager
    from llmling_agent.ui.base import InputProvider
    from llmling_agent.vfs_registry import VFSRegistry


logger = log.get_logger(__name__)


# Model union with discriminator for typed configs
_FileSystemConfigUnion = Annotated[
    FilesystemConfigType | URIFileSystemConfig,
    Field(discriminator="fs_type"),
]

# Final type allowing models or URI shorthand string
ResourceConfig = _FileSystemConfigUnion | str


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
                    "fs_type": "uri",
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
            fs_type: "uri"
            uri: "s3://bucket/data"
            cached: true
    """

    agents: dict[str, AgentConfig] = Field(
        default_factory=dict,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/agent_configuration/"
        },
    )
    """Mapping of agent IDs to their configurations.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/agent_configuration/
    """

    file_agents: dict[str, str] = Field(
        default_factory=dict,
        examples=[
            {
                "code_reviewer": ".claude/agents/reviewer.md",
                "debugger": "https://example.com/agents/debugger.md",
            }
        ],
    )
    """Mapping of agent IDs to Claude Code style subagent file paths.

    Supports local and remote paths via UPath. Files must have YAML frontmatter
    with Claude Code subagent format (name, description, tools, model, etc.).
    The markdown body becomes the system prompt.

    Example file format:
        ```markdown
        ---
        name: code-reviewer
        description: Expert code reviewer
        model: sonnet
        ---

        You are a senior code reviewer...
        ```
    """

    teams: dict[str, TeamConfig] = Field(
        default_factory=dict,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/team_configuration/"
        },
    )
    """Mapping of team IDs to their configurations.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/team_configuration/
    """

    acp_agents: dict[str, ACPAgentConfigTypes] = Field(
        default_factory=dict,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/acp_configuration/"
        },
    )
    """Mapping of ACP agent IDs to their configurations.

    ACP agents are external agents that communicate via the Agent Client Protocol.
    Supports custom ACP servers and pre-configured presets (claude, etc.).

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/acp_configuration/
    """

    agui_agents: dict[str, AGUIAgentConfig] = Field(
        default_factory=dict,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/agui_configuration/"
        },
    )
    """Mapping of AG-UI agent IDs to their configurations.

    AG-UI agents connect to remote HTTP endpoints implementing the AG-UI protocol.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/agui_configuration/
    """

    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/storage_configuration/"
        },
    )
    """Storage provider configuration.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/storage_configuration/
    """

    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    """Observability provider configuration."""

    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    """Document conversion configuration."""

    responses: dict[str, StructuredResponseConfig] = Field(
        default_factory=dict,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/response_configuration/"
        },
    )
    """Mapping of response names to their definitions.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/response_configuration/
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
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/mcp_configuration/"
        },
    )
    """List of MCP server configurations:

    These MCP servers are used to provide tools and other resources to the nodes.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/mcp_configuration/
    """
    pool_server: MCPPoolServerConfig = Field(default_factory=MCPPoolServerConfig)
    """Pool server configuration.

    This MCP server configuration is used for the pool MCP server,
    which exposes pool functionality to other applications / clients."""

    prompts: PromptLibraryConfig = Field(
        default_factory=PromptLibraryConfig,
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/prompt_configuration/"
        },
    )
    """Prompt library configuration.

    This configuration defines the prompt library, which is used to provide prompts to the nodes.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/prompt_configuration/
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
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/manifest_configuration/",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_workers(cls, data: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
        """Convert string workers to appropriate WorkerConfig for all agents."""
        teams = data.get("teams", {})
        agents = data.get("agents", {})
        acp_agents = data.get("acp_agents", {})
        agui_agents = data.get("agui_agents", {})

        # Process workers for all agents that have them
        for agent_name, agent_config in agents.items():
            if isinstance(agent_config, dict):
                workers = agent_config.get("workers", [])
            else:
                workers = agent_config.workers

            if workers:
                normalized: list[BaseWorkerConfig] = []

                for worker in workers:
                    match worker:
                        case str() as name if name in teams:
                            # Determine type based on presence in teams/agents/acp_agents
                            normalized.append(TeamWorkerConfig(name=name))
                        case str() as name if name in acp_agents:
                            normalized.append(ACPAgentWorkerConfig(name=name))
                        case str() as name if name in agui_agents:
                            normalized.append(AGUIAgentWorkerConfig(name=name))
                        case str() as name if name in agents:
                            normalized.append(AgentWorkerConfig(name=name))
                        case str():  # Default to agent if type can't be determined
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
                                elif worker_name in acp_agents:
                                    normalized.append(ACPAgentWorkerConfig(**config))
                                elif worker_name in agui_agents:
                                    normalized.append(AGUIAgentWorkerConfig(**config))
                                else:
                                    normalized.append(AgentWorkerConfig(**config))

                        case BaseWorkerConfig():  # Already normalized
                            normalized.append(worker)

                        case _:
                            msg = f"Invalid worker configuration: {worker}"
                            raise ValueError(msg)

                if isinstance(agent_config, dict):
                    agent_config["workers"] = normalized
                else:
                    # Need to create a new dict with updated workers
                    agent_dict = agent_config.model_dump()
                    agent_dict["workers"] = normalized
                    agents[agent_name] = agent_dict

        return data

    @cached_property
    def vfs_registry(self) -> VFSRegistry:
        """Get registry with all configured VFS resources."""
        from llmling_agent.vfs_registry import VFSRegistry

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

        def resolve_node(name: str) -> dict[str, Any]:
            if name in resolved:
                return resolved[name]

            if name in seen:
                msg = f"Circular inheritance detected: {name}"
                raise ValueError(msg)

            seen.add(name)
            config = (
                nodes[name].model_copy()
                if hasattr(nodes[name], "model_copy")
                else nodes[name].copy()
            )
            inherit = config.get("inherits") if isinstance(config, dict) else config.inherits
            if inherit:
                if inherit not in nodes:
                    msg = f"Parent agent {inherit} not found"
                    raise ValueError(msg)

                # Get resolved parent config
                parent = resolve_node(inherit)
                # Merge parent with child (child overrides parent)
                merged = parent.copy()
                merged.update(config)
                config = merged

            seen.remove(name)
            resolved[name] = config
            return config  # type: ignore[no-any-return]

        # Resolve all nodes
        for name in nodes:
            resolved[name] = resolve_node(name)

        # Update nodes with resolved configs
        data["agents"] = resolved
        return data

    @cached_property
    def _loaded_file_agents(self) -> dict[str, AgentConfig]:
        """Load and cache file-based agent configurations.

        Parses Claude Code style markdown files and converts them to AgentConfig.
        Results are cached to avoid repeated file reads.
        """
        loaded: dict[str, AgentConfig] = {}
        for name, file_path in self.file_agents.items():
            try:
                config = parse_agent_file(file_path)
                # Ensure name is set from the key
                if config.name is None:
                    config = config.model_copy(update={"name": name})
                loaded[name] = config
            except Exception as e:
                logger.exception("Failed to load file agent %r from %s", name, file_path)
                msg = f"Failed to load file agent {name!r} from {file_path}: {e}"
                raise ValueError(msg) from e
        return loaded

    @property
    def node_names(self) -> list[str]:
        """Get list of all agent, ACP agent, AG-UI agent, and team names."""
        return (
            list(self.agents.keys())
            + list(self.file_agents.keys())
            + list(self.acp_agents.keys())
            + list(self.agui_agents.keys())
            + list(self.teams.keys())
        )

    @property
    def nodes(self) -> dict[str, Any]:
        """Get all agent, ACP agent, AG-UI agent, and team configurations."""
        return {
            **self.agents,
            **self._loaded_file_agents,
            **self.acp_agents,
            **self.agui_agents,
            **self.teams,
        }

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
                # Set name if not provided
                if config.name is None:
                    config.name = name
                result[name] = config
        return result

    @cached_property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager for this manifest."""
        from llmling_agent.prompts.manager import PromptManager

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

    def get_agent[TAgentDeps](
        self,
        name: str,
        deps_type: type[TAgentDeps] | None = None,
        input_provider: InputProvider | None = None,
        pool: AgentPool[Any] | None = None,
        event_handlers: list[IndividualEventHandler] | None = None,
    ) -> Agent[TAgentDeps, Any]:
        from llmling_agent import Agent
        from llmling_agent.utils.result_utils import to_type
        from llmling_agent_config.system_prompts import (
            FilePromptConfig,
            FunctionPromptConfig,
            LibraryPromptConfig,
            StaticPromptConfig,
        )

        # Get config from inline agents or file agents
        if name in self.agents:
            config = self.agents[name]
        elif name in self.file_agents:
            config = self._loaded_file_agents[name]
        else:
            msg = f"Agent {name!r} not found in agents or file_agents"
            raise KeyError(msg)
        sys_prompts: list[str] = []
        for prompt in config.system_prompts:
            match prompt:
                case (str() as sys_prompt) | StaticPromptConfig(content=sys_prompt):
                    sys_prompts.append(sys_prompt)
                case FilePromptConfig(path=path, variables=variables):
                    template_path = Path(path)  # Load template from file
                    if not template_path.is_absolute() and config.config_file_path:
                        template_path = Path(config.config_file_path).parent / path

                    template_content = template_path.read_text("utf-8")
                    if variables:  # Apply variables if any
                        from jinja2 import Template

                        template = Template(template_content)
                        content = template.render(**variables)
                    else:
                        content = template_content
                    sys_prompts.append(content)
                case LibraryPromptConfig(reference=reference):
                    try:  # Load from library
                        content = self.prompt_manager.get.sync(reference)
                        sys_prompts.append(content)
                    except Exception as e:
                        msg = f"Failed to load library prompt {reference!r} for agent {name}"
                        logger.exception(msg)
                        raise ValueError(msg) from e
                case FunctionPromptConfig(function=function, arguments=arguments):
                    content = function(**arguments)  # Call function to get prompt content
                    sys_prompts.append(content)
        # Prepare toolsets list with config's tool provider
        toolsets_list = config.get_toolsets()
        if config_tool_provider := config.get_tool_provider():
            toolsets_list.append(config_tool_provider)
        # Step 1: Get agent-specific output type (same as before)
        agent_output_type = self.get_output_type(name) or str
        # Step 2: Resolve it fully with to_type (same as before)
        resolved_output_type = to_type(agent_output_type, self.responses)
        return Agent(
            # context=context,
            model=config.model
            if isinstance(config.model, str) or config.model is None
            else config.model.get_model(),
            system_prompt=sys_prompts,
            name=name,
            display_name=config.display_name,
            deps_type=deps_type,
            description=config.description,
            retries=config.retries,
            session=config.get_session_config(),
            output_retries=config.output_retries,
            end_strategy=config.end_strategy,
            debug=config.debug,
            agent_config=config,
            input_provider=input_provider,
            output_type=resolved_output_type,
            event_handlers=event_handlers,
            agent_pool=pool,
            tool_mode=config.tool_mode,
            knowledge=config.knowledge,
            toolsets=toolsets_list,
        )

    def get_acp_agent[TDeps](
        self,
        name: str,
        deps_type: type[TDeps] | None = None,
    ) -> ACPAgent[TDeps]:
        """Create an ACPAgent from configuration.

        Args:
            name: Name of the ACP agent in the manifest
            deps_type: Optional dependency type (not used by ACP agents currently)

        Returns:
            Configured ACPAgent instance
        """
        from llmling_agent.agent.acp_agent import ACPAgent

        config = self.acp_agents[name]
        # Ensure name is set on config
        if config.name is None:
            config = config.model_copy(update={"name": name})
        return ACPAgent(config=config)

    def get_agui_agent[TDeps](
        self,
        name: str,
        deps_type: type[TDeps] | None = None,
    ) -> AGUIAgent[TDeps]:
        """Create an AGUIAgent from configuration.

        Args:
            name: Name of the AG-UI agent in the manifest
            deps_type: Optional dependency type (not used by AG-UI agents currently)

        Returns:
            Configured AGUIAgent instance
        """
        from llmling_agent.agent.agui_agent import AGUIAgent

        config = self.agui_agents[name]
        # Ensure name is set on config
        if config.name is None:
            config = config.model_copy(update={"name": name})
        return AGUIAgent(
            endpoint=config.endpoint,
            name=config.name or "agui-agent",
            description=config.description,
            display_name=config.display_name,
            timeout=config.timeout,
            headers=config.headers,
            startup_command=config.startup_command,
            startup_delay=config.startup_delay,
            mcp_servers=config.mcp_servers,
        )

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
            # Update all agents with the config file path and ensure names
            agents = {
                name: config.model_copy(update={"config_file_path": str(path)})
                for name, config in agent_def.agents.items()
            }
            return agent_def.model_copy(update={"agents": agents})
        except Exception as exc:
            msg = f"Failed to load agent config from {path}"
            raise ValueError(msg) from exc

    @cached_property
    def pool(self) -> AgentPool:
        """Create an agent pool from this manifest.

        Returns:
            Configured agent pool
        """
        from llmling_agent import AgentPool

        return AgentPool(manifest=self)

    def get_output_type(self, agent_name: str) -> type[Any] | None:
        """Get the resolved result type for an agent.

        Returns None if no result type is configured.
        """
        agent_config = self.agents[agent_name]
        if not agent_config.output_type:
            return None
        logger.debug("Building response model", type=agent_config.output_type)
        if isinstance(agent_config.output_type, str):
            response_def = self.responses[agent_config.output_type]
            return response_def.response_schema.get_schema()
        return agent_config.output_type.response_schema.get_schema()


if __name__ == "__main__":
    model = {"type": "input"}
    agent_cfg = dict(name="test_agent", model=model)
    manifest = AgentsManifest(agents=dict(test_agent=agent_cfg))  # pyright: ignore[reportArgumentType]
    print(AgentsManifest.generate_test_data(mode="maximal").model_dump_yaml())
