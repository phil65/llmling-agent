"""Pydantic models for Claude Code server info structures."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClaudeCodeModelInfo(BaseModel):
    """Information about an available AI model from Claude Code.

    Attributes:
        value: Model identifier used in API calls (e.g., "default", "opus", "haiku")
        display_name: Human-readable display name (e.g., "Opus", "Default (recommended)")
        description: Full description including capabilities and pricing
    """

    value: str = Field(..., description="Model identifier for API calls")
    display_name: str = Field(..., alias="displayName", description="Human-readable name")
    description: str = Field(..., description="Full description with pricing info")


class ClaudeCodeCommandInfo(BaseModel):
    """Information about an available slash command from Claude Code.

    Attributes:
        name: Command name without the / prefix (e.g., "compact", "review")
        description: Full description of what the command does
        argument_hint: Usage hint for command arguments (may be empty string)
    """

    name: str = Field(..., description="Command name without / prefix")
    description: str = Field(..., description="What the command does")
    argument_hint: str = Field(..., alias="argumentHint", description="Usage hint for arguments")


class ClaudeCodeAccountInfo(BaseModel):
    """Account information from Claude Code.

    Attributes:
        token_source: Where tokens come from (e.g., "claude.ai")
        api_key_source: Where API key comes from (e.g., "ANTHROPIC_API_KEY")
    """

    token_source: str = Field(..., alias="tokenSource", description="Token source")
    api_key_source: str = Field(..., alias="apiKeySource", description="API key source")


class ClaudeCodeServerInfo(BaseModel):
    """Complete server initialization info from Claude Code.

    This is returned by the Claude Code server during initialization and contains
    all available capabilities including models, commands, output styles, and
    account information.

    Attributes:
        models: List of available AI models
        commands: List of available slash commands
        output_style: Current output style setting
        available_output_styles: List of all available output styles
        account: Account and authentication information
    """

    models: list[ClaudeCodeModelInfo] = Field(
        default_factory=list, description="Available AI models"
    )
    commands: list[ClaudeCodeCommandInfo] = Field(
        default_factory=list, description="Available slash commands"
    )
    output_style: str = Field(default="default", description="Current output style")
    available_output_styles: list[str] = Field(
        default_factory=list, description="All available output styles"
    )
    account: ClaudeCodeAccountInfo | None = Field(default=None, description="Account information")

    model_config = {"populate_by_name": True}
