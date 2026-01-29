"""Pydantic models for Claude Code server info structures."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class ClaudeCodeBasemodel(BaseModel):
    """Base model."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class ClaudeCodeModelInfo(ClaudeCodeBasemodel):
    """Information about an available AI model from Claude Code."""

    value: str
    """Model identifier used in API calls (e.g., "default", "opus", "haiku")."""

    display_name: str
    """Human-readable display name (e.g., "Opus", "Default (recommended)")."""

    description: str
    """Full description including capabilities and pricing."""


class ClaudeCodeCommandInfo(ClaudeCodeBasemodel):
    """Information about an available slash command from Claude Code."""

    name: str
    """Command name without the / prefix (e.g., "compact", "review")."""

    description: str
    """Full description of what the command does."""

    argument_hint: str
    """Usage hint for command arguments (may be empty string)."""


class ClaudeCodeAccountInfo(ClaudeCodeBasemodel):
    """Account information from Claude Code."""

    email: str | None = None
    """User email address."""

    subscription_type: str | None = None
    """Subscription type (e.g., "Claude API")."""

    token_source: str | None = None
    """Where tokens come from (e.g., "claude.ai")."""

    api_key_source: str | None = None
    """Where API key comes from (e.g., "ANTHROPIC_API_KEY")."""


class ClaudeCodeServerInfo(ClaudeCodeBasemodel):
    """Complete server initialization info from Claude Code.

    This is returned by the Claude Code server during initialization and contains
    all available capabilities including models, commands, output styles, and
    account information.
    """

    models: list[ClaudeCodeModelInfo] = Field(default_factory=list)
    """List of available AI models."""

    commands: list[ClaudeCodeCommandInfo] = Field(default_factory=list)
    """List of available slash commands."""

    output_style: str = Field(default="default")
    """Current output style setting."""

    available_output_styles: list[str] = Field(default_factory=list)
    """List of all available output styles."""

    account: ClaudeCodeAccountInfo | None = Field(default=None)
    """Account and authentication information."""
