"""Configuration models for Claude Code agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field

from llmling_agent_config.nodes import BaseAgentConfig
from llmling_agent_config.output_types import StructuredResponseConfig  # noqa: TC001
from llmling_agent_config.toolsets import ToolsetConfig


if TYPE_CHECKING:
    from llmling_agent.resource_providers import ResourceProvider


PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]


class ClaudeCodeAgentConfig(BaseAgentConfig):
    """Configuration for Claude Code agents.

    Claude Code agents use the Claude Agent SDK to interact with Claude Code CLI,
    enabling file operations, terminal access, and code editing capabilities.

    Example:
        ```yaml
        claude_code_agents:
          coder:
            cwd: /path/to/project
            model: claude-sonnet-4-5
            allowed_tools:
              - Read
              - Write
              - Bash
            system_prompt: "You are a helpful coding assistant."
            max_turns: 10

          planner:
            cwd: /path/to/project
            permission_mode: plan
            max_thinking_tokens: 10000
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "title": "Claude Code Agent Configuration",
            "x-icon": "simple-icons:anthropic",
        }
    )

    type: Literal["claude_code"] = Field(default="claude_code", init=False)
    """Top-level discriminator for agent type."""

    cwd: str | None = Field(
        default=None,
        title="Working Directory",
        examples=["/path/to/project", ".", "/home/user/myproject"],
    )
    """Working directory for Claude Code operations."""

    model: str | None = Field(
        default=None,
        title="Model",
        examples=["claude-sonnet-4-5", "claude-opus-4", "claude-haiku-3-5"],
    )
    """Model to use for this agent. Defaults to Claude's default model."""

    allowed_tools: list[str] | None = Field(
        default=None,
        title="Allowed Tools",
        examples=[["Read", "Write", "Bash"], ["Read", "Grep", "Glob"]],
    )
    """List of tool names the agent is allowed to use.

    If not specified, all tools are available (subject to permission_mode).
    Common tools: Read, Write, Edit, Bash, Glob, Grep, Task, WebFetch, etc.
    """

    disallowed_tools: list[str] | None = Field(
        default=None,
        title="Disallowed Tools",
        examples=[["Bash", "Write"], ["Task"]],
    )
    """List of tool names the agent is NOT allowed to use.

    Takes precedence over allowed_tools if both are specified.
    """

    system_prompt: str | None = Field(
        default=None,
        title="System Prompt",
        examples=["You are a helpful coding assistant focused on Python."],
    )
    """Custom system prompt to use instead of the default."""

    max_turns: int | None = Field(
        default=None,
        title="Max Turns",
        ge=1,
        examples=[5, 10, 20],
    )
    """Maximum number of conversation turns before stopping."""

    max_thinking_tokens: int | None = Field(
        default=None,
        title="Max Thinking Tokens",
        ge=1000,
        examples=[5000, 10000, 50000],
    )
    """Maximum tokens for extended thinking mode.

    When set, enables Claude's extended thinking capability for more
    complex reasoning tasks.
    """

    permission_mode: PermissionMode | None = Field(
        default=None,
        title="Permission Mode",
        examples=["default", "acceptEdits", "plan", "bypassPermissions"],
    )
    """Permission handling mode:

    - "default": Ask for permission on each tool use
    - "acceptEdits": Auto-accept file edits but ask for other operations
    - "plan": Plan-only mode, no execution
    - "bypassPermissions": Skip all permission checks (use with caution)
    """

    output_type: str | StructuredResponseConfig | None = Field(
        default=None,
        examples=["json_response", "code_output"],
        title="Response type",
    )
    """Optional structured output type for responses.

    Can be either a reference to a response defined in manifest.responses,
    or an inline StructuredResponseConfig.
    """

    toolsets: list[ToolsetConfig] = Field(
        default_factory=list,
        title="Toolsets",
        examples=[
            [
                {"type": "subagent"},
                {"type": "agent_management"},
            ],
        ],
    )
    """Toolsets to expose to this Claude Code agent via MCP bridge.

    These toolsets will be started as an in-process MCP server and made
    available to Claude Code. This allows Claude Code to use internal
    llmling-agent toolsets like subagent delegation, agent management, etc.

    The toolsets are exposed using the Claude SDK's native MCP support,
    which passes the FastMCP server instance directly without HTTP overhead.
    """

    def get_toolset_providers(self) -> list[ResourceProvider]:
        """Get resource providers for all configured toolsets.

        Returns:
            List of initialized ResourceProvider instances
        """
        return [toolset.get_provider() for toolset in self.toolsets]
