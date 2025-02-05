"""Models for agent configuration."""

from __future__ import annotations

from functools import cached_property
import logging
from typing import TYPE_CHECKING, Any, Literal, Self

from llmling import ConfigModel
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import TypeVar

from llmling_agent.models.agents import AgentConfig
from llmling_agent.models.converters import ConversionConfig
from llmling_agent.models.mcp_server import MCPServerBase, MCPServerConfig, StdioMCPServer
from llmling_agent.models.observability import ObservabilityConfig
from llmling_agent.models.prompts import PromptConfig
from llmling_agent.models.providers import BaseProviderConfig
from llmling_agent.models.result_types import ResponseDefinition  # noqa: TC001
from llmling_agent.models.storage import StorageConfig
from llmling_agent.models.task import Job  # noqa: TC001
from llmling_agent.models.teams import TeamConfig  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent import AgentPool
    from llmling_agent.agent import AnyAgent
    from llmling_agent.common_types import StrPath
    from llmling_agent.prompts.manager import PromptManager


TDeps = TypeVar("TDeps", default=None)
ToolConfirmationMode = Literal["always", "never", "per_tool"]

logger = logging.getLogger(__name__)


# TODO: python 3.13: set defaults here
class AgentsManifest(ConfigModel):
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

    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    """Mapping of agent IDs to their configurations"""

    teams: dict[str, TeamConfig] = Field(default_factory=dict)
    """Mapping of team IDs to their configurations"""

    storage: StorageConfig = Field(default_factory=StorageConfig)
    """Storage provider configuration."""

    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    """Observability provider configuration."""

    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    """Document conversion configuration."""

    responses: dict[str, ResponseDefinition] = Field(default_factory=dict)
    """Mapping of response names to their definitions"""

    jobs: dict[str, Job] = Field(default_factory=dict)
    """Pre-defined jobs, ready to be used by agents."""

    mcp_servers: list[str | MCPServerConfig] = Field(default_factory=list)
    """List of MCP server configurations:
    - str entries are converted to StdioMCPServer
    - MCPServerConfig for full server configuration
    """
    prompts: PromptConfig = Field(default_factory=PromptConfig)

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

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

        # Deep copy the configuration
        config = self.agents[name].model_copy(deep=True)

        # Apply overrides
        for key, value in overrides.items():
            if not hasattr(config, key):
                msg = f"Invalid override: {key}"
                raise ValueError(msg)
            setattr(config, key, value)

        # Handle template rendering if context provided
        if template_context:
            # Apply name from context if not explicitly overridden
            if "name" in template_context and "name" not in overrides:
                config.name = template_context["name"]

            # Render system prompts
            config.system_prompts = config.render_system_prompts(template_context)

        self.agents[actual_name] = config
        return actual_name

    @model_validator(mode="before")
    @classmethod
    def resolve_inheritance(cls, data: dict) -> dict:
        """Resolve agent inheritance chains."""
        agents = data.get("agents", {})
        resolved: dict[str, dict] = {}
        seen: set[str] = set()

        def resolve_agent(name: str) -> dict:
            if name in resolved:
                return resolved[name]

            if name in seen:
                msg = f"Circular inheritance detected: {name}"
                raise ValueError(msg)

            seen.add(name)
            config = (
                agents[name].model_copy()
                if hasattr(agents[name], "model_copy")
                else agents[name].copy()
            )
            inherit = (
                config.get("inherits") if isinstance(config, dict) else config.inherits
            )
            if inherit:
                if inherit not in agents:
                    msg = f"Parent agent {inherit} not found"
                    raise ValueError(msg)

                # Get resolved parent config
                parent = resolve_agent(inherit)
                # Merge parent with child (child overrides parent)
                merged = parent.copy()
                merged.update(config)
                config = merged

            seen.remove(name)
            resolved[name] = config
            return config

        # Resolve all agents
        for name in agents:
            resolved[name] = resolve_agent(name)

        # Update agents with resolved configs
        data["agents"] = resolved
        return data

    @property
    def node_names(self) -> list[str]:
        """Get list of all agent and team names."""
        return list(self.agents.keys()) + list(self.teams.keys())

    @property
    def nodes(self) -> dict[str, Any]:
        """Get all agent and team configurations."""
        return {**self.agents, **self.teams}

    def get_mcp_servers(self) -> list[MCPServerConfig]:
        """Get processed MCP server configurations.

        Converts string entries to StdioMCPServer configs by splitting
        into command and arguments.

        Returns:
            List of MCPServerConfig instances

        Raises:
            ValueError: If string entry is empty
        """
        configs: list[MCPServerConfig] = []

        for server in self.mcp_servers:
            match server:
                case str():
                    parts = server.split()
                    if not parts:
                        msg = "Empty MCP server command"
                        raise ValueError(msg)

                    configs.append(StdioMCPServer(command=parts[0], args=parts[1:]))
                case MCPServerBase():
                    configs.append(server)

        return configs

    @cached_property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager for this manifest."""
        from llmling_agent.prompts.manager import PromptManager

        return PromptManager(self.prompts)

    # @model_validator(mode="after")
    # def validate_response_types(self) -> AgentsManifest:
    #     """Ensure all agent result_types exist in responses or are inline."""
    #     for agent_id, agent in self.agents.items():
    #         if (
    #             isinstance(agent.result_type, str)
    #             and agent.result_type not in self.responses
    #         ):
    #             msg = f"'{agent.result_type=}' for '{agent_id=}' not found in responses"
    #             raise ValueError(msg)
    #     return self

    def get_agent[TAgentDeps](
        self, name: str, deps: TAgentDeps | None = None
    ) -> AnyAgent[TAgentDeps, Any]:
        from llmling import RuntimeConfig

        from llmling_agent import Agent, AgentContext

        config = self.agents[name]
        # Create runtime without async context
        cfg = config.get_config()
        runtime = RuntimeConfig.from_config(cfg)

        # Create context with config path and capabilities
        context = AgentContext[TAgentDeps](
            node_name=name,
            data=deps,
            capabilities=config.capabilities,
            definition=self,
            config=config,
            # pool=self,
            # confirmation_callback=confirmation_callback,
        )

        provider = config.get_provider()
        # set model for provider (the setting should move to provider config soon)
        provider._model = config.get_model()
        sys_prompts: list[str] = []
        sys_prompts.extend(config.system_prompts)
        # Library prompts
        if config.library_system_prompts:
            for prompt_ref in config.library_system_prompts:
                try:
                    content = self.prompt_manager.get_sync(prompt_ref)
                    sys_prompts.append(content)
                except Exception as e:
                    msg = f"Failed to load library prompt {prompt_ref!r} for agent {name}"
                    logger.exception(msg)
                    raise ValueError(msg) from e

        # Create agent with runtime and context
        agent: AnyAgent[TAgentDeps, Any] = Agent[Any](
            runtime=runtime,
            context=context,
            model=config.get_model(),
            provider=provider,
            system_prompt=sys_prompts,
            name=name,
            # name=config.name or name,
        )
        if result_type := self.get_result_type(name):
            return agent.to_structured(result_type)
        return agent

    def get_used_providers(self) -> set[str]:
        """Get all providers configured in this manifest."""
        providers = set[str]()

        for agent_config in self.agents.values():
            match agent_config.provider:
                case "pydantic_ai":
                    providers.add("pydantic_ai")
                case "litellm":
                    providers.add("litellm")
                case BaseProviderConfig():
                    providers.add(agent_config.provider.type)
        return providers

    def register_agent(
        self,
        name: str,
        *,
        model: str | None = None,
        system_prompts: list[str] | None = None,
        description: str | None = None,
        temporary: bool = False,
        **kwargs: Any,
    ) -> AgentConfig:
        """Register a new agent configuration.

        Args:
            name: Name of the new agent
            model: Model to use
            system_prompts: System prompts for the agent
            description: Optional description
            temporary: If True, won't be added to manifest's agents
            **kwargs: Additional AgentConfig fields
        """
        if name in self.agents:
            msg = f"Agent {name} already exists"
            raise ValueError(msg)

        config = AgentConfig(
            name=name,
            model=model,
            system_prompts=system_prompts or [],
            description=description,
            **kwargs,
        )

        if not temporary:
            self.agents[name] = config

        return config

    @classmethod
    def from_file(cls, path: StrPath) -> Self:
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
            # Set identifier as name if not set
            for identifier, config in data["agents"].items():
                if not config.get("name"):
                    config["name"] = identifier
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
        from llmling_agent.delegation import AgentPool

        return AgentPool(manifest=self)

    def get_result_type(self, agent_name: str) -> type[Any] | None:
        """Get the resolved result type for an agent.

        Returns None if no result type is configured.
        """
        agent_config = self.agents[agent_name]
        if not agent_config.result_type:
            return None
        logger.debug("Building response model for %r", agent_config.result_type)
        if isinstance(agent_config.result_type, str):
            response_def = self.responses[agent_config.result_type]
            return response_def.create_model()  # type: ignore
        return agent_config.result_type.create_model()  # type: ignore


if __name__ == "__main__":
    model = {"type": "input"}
    agent_cfg = AgentConfig(name="test_agent", model=model)  # type: ignore
    manifest = AgentsManifest(agents=dict(test_agent=agent_cfg))
    print(manifest.agents["test_agent"].model)
