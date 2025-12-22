"""Configuration models for ACP (Agent Client Protocol) agents."""

from __future__ import annotations

from collections.abc import Sequence
import os
import tempfile
from typing import TYPE_CHECKING, Annotated, Any, Literal

from exxec import ExecutionEnvironmentStr, get_environment  # noqa: TC002
from exxec.configs import (
    E2bExecutionEnvironmentConfig,
    ExecutionEnvironmentConfig,  # noqa: TC002
)
from pydantic import ConfigDict, Field
from tokonomics.model_discovery import ProviderType  # noqa: TC002

from agentpool_config.nodes import NodeConfig
from agentpool_config.system_prompts import PromptConfig  # noqa: TC001


if TYPE_CHECKING:
    from exxec import ExecutionEnvironment

    from agentpool.prompts.manager import PromptManager


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

    type: Literal["acp"] = Field("acp", init=False)
    """Top-level discriminator for agent type."""

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
    """Execution environment config for the agent's own toolsets."""

    client_execution_environment: Annotated[
        ExecutionEnvironmentStr | ExecutionEnvironmentConfig | None,
        Field(
            default=None,
            title="Client Execution Environment",
            examples=[
                "local",
                "docker",
                E2bExecutionEnvironmentConfig(template="python-sandbox"),
            ],
        ),
    ] = None
    """Execution environment for handling subprocess requests (filesystem, terminals).

    When the ACP subprocess requests file/terminal operations, this environment
    determines where those operations execute. Falls back to execution_environment
    if not set.

    Use cases:
    - None (default): Use same env as toolsets (execution_environment)
    - "local": Subprocess operates on its own local filesystem
    - Remote config: Subprocess operates in a specific remote environment
    """

    allow_file_operations: bool = Field(default=True, title="Allow File Operations")
    """Whether to allow file read/write operations."""

    allow_terminal: bool = Field(default=True, title="Allow Terminal")
    """Whether to allow terminal operations."""

    requires_tool_confirmation: Literal["never", "always"] = Field(
        default="always", title="Tool confirmation mode"
    )
    """Whether to automatically grant all permission requests."""

    system_prompt: str | Sequence[str | PromptConfig] | None = Field(
        default=None,
        title="System Prompt",
        examples=[
            "You are a helpful coding assistant.",
            ["Always write tests.", "Focus on Python."],
        ],
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/agentpool/YAML%20Configuration/system_prompts_configuration/"
        },
    )
    """System prompt for the agent. Can be a string or list of strings/prompt configs.

    Support varies by agent:
    - Claude: passed via --system-prompt
    - Auggie: passed via --instruction-file
    - Stakpak: passed via --system-prompt-file
    - Others: may not support system prompts

    Docs: https://phil65.github.io/agentpool/YAML%20Configuration/system_prompts_configuration/
    """

    def get_command(self) -> str:
        """Get the command to spawn the ACP server."""
        raise NotImplementedError

    async def get_args(self, prompt_manager: PromptManager | None = None) -> list[str]:
        """Get command arguments."""
        raise NotImplementedError

    async def render_system_prompt(
        self,
        prompt_manager: PromptManager | None = None,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Render system prompt to a single string.

        Resolves library references and renders templates.

        Args:
            prompt_manager: Optional prompt manager for resolving library references
            context: Optional context for template rendering

        Returns:
            Rendered system prompt string, or None if no prompt configured
        """
        from toprompt import render_prompt

        from agentpool_config.system_prompts import (
            FilePromptConfig,
            FunctionPromptConfig,
            LibraryPromptConfig,
            StaticPromptConfig,
        )

        if self.system_prompt is None:
            return None

        context = context or {"name": self.name}
        prompt_list = (
            [self.system_prompt] if isinstance(self.system_prompt, str) else self.system_prompt
        )

        rendered_parts: list[str] = []
        for prompt in prompt_list:
            match prompt:
                case str() as content:
                    rendered_parts.append(render_prompt(content, {"agent": context}))
                case StaticPromptConfig(content=content):
                    rendered_parts.append(render_prompt(content, {"agent": context}))
                case FilePromptConfig(path=path, variables=variables):
                    from pathlib import Path

                    template_path = Path(path)
                    if not template_path.is_absolute() and self.config_file_path:
                        base_path = Path(self.config_file_path).parent
                        template_path = base_path / path
                    template_content = template_path.read_text("utf-8")
                    template_ctx = {"agent": context, **variables}
                    rendered_parts.append(render_prompt(template_content, template_ctx))
                case LibraryPromptConfig(reference=reference):
                    if prompt_manager:
                        resolved = await prompt_manager.get_from(reference)
                        rendered_parts.append(render_prompt(resolved, {"agent": context}))
                    else:
                        # Fallback: include reference marker
                        rendered_parts.append(f"[LIBRARY:{reference}]")
                case FunctionPromptConfig(function=function, arguments=arguments):
                    content = function(**arguments)
                    rendered_parts.append(render_prompt(content, {"agent": context}))

        return "\n\n".join(rendered_parts) if rendered_parts else None

    async def write_system_prompt_file(
        self,
        prompt_manager: PromptManager | None = None,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Write system prompt to a temporary file.

        Creates a temp file in the system temp directory that will be
        cleaned up on system restart.

        Args:
            prompt_manager: Optional prompt manager for resolving library references
            context: Optional context for template rendering

        Returns:
            Path to the temp file, or None if no prompt configured
        """
        content = await self.render_system_prompt(prompt_manager, context)
        if not content:
            return None

        fd, path = tempfile.mkstemp(prefix="agentpool_prompt_", suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def get_execution_environment(self) -> ExecutionEnvironment:
        """Create execution environment from config."""
        if isinstance(self.execution_environment, str):
            return get_environment(self.execution_environment)
        return self.execution_environment.get_provider()

    def get_client_execution_environment(self) -> ExecutionEnvironment | None:
        """Create client execution environment from config.

        Returns None if not configured (caller should fall back to main env).
        """
        if self.client_execution_environment is None:
            return None
        if isinstance(self.client_execution_environment, str):
            return get_environment(self.client_execution_environment)
        return self.client_execution_environment.get_provider()

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
            provider: custom
            command: my-acp-server
            args: ["--mode", "coding"]
            cwd: /path/to/project
        ```
    """

    model_config = ConfigDict(json_schema_extra={"title": "Custom ACP Agent Configuration"})

    provider: Literal["custom"] = Field("custom", init=False)
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

    async def get_args(self, prompt_manager: PromptManager | None = None) -> list[str]:
        """Get command arguments."""
        _ = prompt_manager  # Custom agents use explicit args
        return self.args
