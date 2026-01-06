"""Models for agentpool standalone tool configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from exxec_config import ExecutionEnvironmentConfig  # noqa: TC002
from pydantic import ConfigDict, Field

from agentpool_config.tools import BaseToolConfig


if TYPE_CHECKING:
    from agentpool.tools.base import Tool


class BashToolConfig(BaseToolConfig):
    """Configuration for bash command execution tool.

    Example:
        ```yaml
        tools:
          - type: bash
            timeout: 30.0
            output_limit: 10000
            requires_confirmation: true
            environment:
              type: mock
              deterministic_ids: true
        ```
    """

    model_config = ConfigDict(title="Bash Tool")

    type: Literal["bash"] = Field("bash", init=False)
    """Bash command execution tool."""

    timeout: float | None = Field(
        default=None,
        examples=[30.0, 60.0, 120.0],
        title="Command timeout",
    )
    """Command timeout in seconds. None means no timeout."""

    output_limit: int | None = Field(
        default=None,
        examples=[10000, 50000, 100000],
        title="Output limit",
    )
    """Maximum bytes of output to return."""

    environment: ExecutionEnvironmentConfig | None = Field(
        default=None,
        title="Execution environment",
    )
    """Execution environment for command execution. Falls back to agent's env if not set."""

    def get_tool(self) -> Tool:
        """Convert config to BashTool instance."""
        from agentpool.tool_impls.bash import create_bash_tool

        env = self.environment.get_provider() if self.environment else None
        return create_bash_tool(
            env=env,
            timeout=self.timeout,
            output_limit=self.output_limit,
            name=self.name or "bash",
            description=self.description or "Execute a shell command and return the output.",
            requires_confirmation=self.requires_confirmation,
        )


class AgentCliToolConfig(BaseToolConfig):
    """Configuration for agent CLI tool.

    Example:
        ```yaml
        tools:
          - type: agent_cli
        ```
    """

    model_config = ConfigDict(title="Agent CLI Tool")

    type: Literal["agent_cli"] = Field("agent_cli", init=False)
    """Agent CLI tool."""

    def get_tool(self) -> Tool:
        """Convert config to AgentCliTool instance."""
        from agentpool.tool_impls.agent_cli import create_agent_cli_tool

        return create_agent_cli_tool(
            name=self.name or "run_agent_cli_command",
            description=self.description or "Execute an internal agent management command.",
            requires_confirmation=self.requires_confirmation,
        )


class AskUserToolConfig(BaseToolConfig):
    """Configuration for user interaction tool.

    Example:
        ```yaml
        tools:
          - type: ask_user
        ```
    """

    model_config = ConfigDict(title="Ask User Tool")

    type: Literal["ask_user"] = Field("ask_user", init=False)
    """User interaction tool."""

    def get_tool(self) -> Tool:
        """Convert config to AskUserTool instance."""
        from agentpool.tool_impls.ask_user import create_ask_user_tool

        return create_ask_user_tool(
            name=self.name or "ask_user",
            description=self.description or "Ask the user a clarifying question.",
            requires_confirmation=self.requires_confirmation,
        )


class ExecuteCodeToolConfig(BaseToolConfig):
    """Configuration for Python code execution tool.

    Example:
        ```yaml
        tools:
          - type: execute_code
            requires_confirmation: true
            environment:
              type: mock
              deterministic_ids: true
        ```
    """

    model_config = ConfigDict(title="Execute Code Tool")

    type: Literal["execute_code"] = Field("execute_code", init=False)
    """Python code execution tool."""

    environment: ExecutionEnvironmentConfig | None = Field(
        default=None,
        title="Execution environment",
    )
    """Execution environment for code execution. Falls back to agent's env if not set."""

    def get_tool(self) -> Tool:
        """Convert config to ExecuteCodeTool instance."""
        from agentpool.tool_impls.execute_code import create_execute_code_tool

        env = self.environment.get_provider() if self.environment else None
        return create_execute_code_tool(
            env=env,
            name=self.name or "execute_code",
            description=self.description or "Execute Python code and return the result.",
            requires_confirmation=self.requires_confirmation,
        )


# Union type for agentpool tool configs
AgentpoolToolConfig = (
    AgentCliToolConfig | AskUserToolConfig | BashToolConfig | ExecuteCodeToolConfig
)
