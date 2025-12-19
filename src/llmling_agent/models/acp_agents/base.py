"""Configuration models for ACP (Agent Client Protocol) agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from exxec import ExecutionEnvironmentStr, get_environment  # noqa: TC002
from exxec.configs import (
    E2bExecutionEnvironmentConfig,
    ExecutionEnvironmentConfig,  # noqa: TC002
)
from pydantic import ConfigDict, Field
from tokonomics.model_discovery import ProviderType  # noqa: TC002

from llmling_agent_config.nodes import NodeConfig


if TYPE_CHECKING:
    from exxec import ExecutionEnvironment


class BaseACPAgentConfig(NodeConfig):
    """Base configuration for all ACP agents.

    Provides common fields and the interface for building commands.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-icon": "octicon:terminal-16",
            "x-doc-title": "ACP Agent Configuration",
        }
    )

    cwd: str | None = Field(
        default=None,
        title="Working Directory",
        examples=["/path/to/project", ".", "/home/user/myproject"],
    )
    """Working directory for the session."""

    env: dict[str, str] = Field(
        default_factory=dict,
        title="Environment Variables",
        examples=[{"PATH": "/usr/local/bin:/usr/bin", "DEBUG": "1"}],
    )
    """Environment variables to set."""

    execution_environment: Annotated[
        ExecutionEnvironmentStr | ExecutionEnvironmentConfig,
        Field(
            default="local",
            title="Execution Environment",
            examples=[
                "docker",
                E2bExecutionEnvironmentConfig(template="python-sandbox"),
            ],
        ),
    ] = "local"
    """Execution environment config for ACP client operations (filesystem, terminals)."""

    allow_file_operations: bool = Field(default=True, title="Allow File Operations")
    """Whether to allow file read/write operations."""

    allow_terminal: bool = Field(default=True, title="Allow Terminal")
    """Whether to allow terminal operations."""

    requires_tool_confirmation: Literal["never", "always"] = Field(
        default="always", title="Tool confirmation mode"
    )
    """Whether to automatically grant all permission requests."""

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        raise NotImplementedError

    def get_args(self) -> list[str]:
        """Get command arguments."""
        raise NotImplementedError

    def get_execution_environment(self) -> ExecutionEnvironment:
        """Create execution environment from config."""
        if isinstance(self.execution_environment, str):
            return get_environment(self.execution_environment)
        return self.execution_environment.get_provider()

    @property
    def model_providers(self) -> list[ProviderType]:
        """Return the model providers used by this ACP agent.

        Override in subclasses to specify which providers the agent uses.
        Used for intelligent model discovery and fallback configuration.
        """
        return []


class ACPAgentConfig(BaseACPAgentConfig):
    """Configuration for a custom ACP agent with explicit command.

    Use this for ACP servers that don't have a preset, or when you need
    full control over the command and arguments.

    Example:
        ```yaml
        agents:
          custom_agent:
            type: acp
            command: my-acp-server
            args: ["--mode", "coding"]
            cwd: /path/to/project
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Custom ACP Agent Configuration"})

    type: Literal["acp"] = Field("acp", init=False)
    """Discriminator for custom ACP agent."""

    command: str = Field(
        ...,
        title="Command",
        examples=["claude-code-acp", "aider", "my-custom-acp"],
    )
    """Command to spawn the ACP server."""

    args: list[str] = Field(
        default_factory=list,
        title="Arguments",
        examples=[["--mode", "coding"], ["--debug", "--verbose"]],
    )
    """Arguments to pass to the command."""

    providers: list[ProviderType] = Field(
        default_factory=list,
        title="Providers",
        examples=[["openai", "anthropic"], ["gemini"]],
    )
    """Model providers this agent can use."""

    @property
    def model_providers(self) -> list[ProviderType]:
        """Return configured providers for custom ACP agents."""
        return list(self.providers)

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        return self.command

    def get_args(self) -> list[str]:
        """Get command arguments."""
        return self.args
