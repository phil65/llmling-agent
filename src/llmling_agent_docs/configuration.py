"""YAML Configuration section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("YAML Configuration")


@nav.route.page("Manifest Overview", icon="octicon:file-code-16")
def _(page: mk.MkPage):
    """Complete manifest structure and organization."""
    page += mk.MkTemplate("docs/config_file/manifest.md")


@nav.route.page("Provider Configuration", icon="octicon:plug-16")
def _(page: mk.MkPage):
    """Agent provider setup and options."""
    page += mk.MkTemplate("docs/config_file/provider_config.md")


@nav.route.page("Model Configuration", icon="octicon:cpu-16")
def _(page: mk.MkPage):
    """Language model setup and configuration."""
    page += mk.MkTemplate("docs/config_file/model_config.md")


@nav.route.page("Team Configuration", icon="fluent:people-team-toolbox-24-regular")
def _(page: mk.MkPage):
    """Team configuration and setup."""
    page += mk.MkTemplate("docs/config_file/team_config.md")


# @nav.route.page("Capabilities", icon="octicon:shield-check-16")
# def _(page: mk.MkPage):
#     """Agent capabilities and permissions."""
#     page += mk.MkTemplate("docs/config_file/capabilities_config.md")


@nav.route.page("Worker Configuration", icon="octicon:people-16")
def _(page: mk.MkPage):
    """Worker agent setup and management."""
    page += mk.MkTemplate("docs/config_file/worker_config.md")


@nav.route.page("Tool Configuration", icon="octicon:tools-16")
def _(page: mk.MkPage):
    """Tool registration and configuration."""
    page += mk.MkTemplate("docs/config_file/tool_config.md")


@nav.route.page("Toolset Configuration", icon="octicon:package-16")
def _(page: mk.MkPage):
    """Toolset setup and management."""
    page += mk.MkTemplate("docs/config_file/toolset_config.md")


@nav.route.page("Response Types", icon="octicon:reply-16")
def _(page: mk.MkPage):
    """Structured response type configuration."""
    page += mk.MkTemplate("docs/config_file/response_config.md")


@nav.route.page("Knowledge Sources", icon="octicon:database-16")
def _(page: mk.MkPage):
    """Knowledge source configuration and management."""
    page += mk.MkTemplate("docs/config_file/knowledge_config.md")


@nav.route.page("Connections & Message Forwarding", icon="octicon:arrow-right-16")
def _(page: mk.MkPage):
    """Message routing and connection configuration."""
    page += mk.MkTemplate("docs/config_file/connection_config.md")


@nav.route.page("Conditions", icon="octicon:git-branch-16")
def _(page: mk.MkPage):
    """Condition configurations."""
    page += mk.MkTemplate("docs/config_file/condition_config.md")


@nav.route.page("System prompts / Prompt library", icon="octicon:book-16")
def _(page: mk.MkPage):
    """Prompt library."""
    page += mk.MkTemplate("docs/config_file/prompt_config.md")


@nav.route.page("Storage Configuration", icon="octicon:database-16")
def _(page: mk.MkPage):
    """Database and storage setup."""
    page += mk.MkTemplate("docs/config_file/storage_config.md")


@nav.route.page("Event Configuration", icon="octicon:broadcast-16")
def _(page: mk.MkPage):
    """Event handling and trigger setup."""
    page += mk.MkTemplate("docs/config_file/events_config.md")


@nav.route.page("MCP Server Setup", icon="octicon:server-16")
def _(page: mk.MkPage):
    """MCP server configuration and integration."""
    page += mk.MkTemplate("docs/config_file/mcp_config.md")


@nav.route.page("Session Configuration", icon="octicon:history-16")
def _(page: mk.MkPage):
    """Session management and configuration."""
    page += mk.MkTemplate("docs/config_file/session_config.md")


@nav.route.page("Task Configuration", icon="octicon:tasklist-16")
def _(page: mk.MkPage):
    """Task definition and configuration."""
    page += mk.MkTemplate("docs/config_file/task_config.md")


@nav.route.page("Environment Setup", icon="octicon:tools-16")
def _(page: mk.MkPage):
    """Environment configuration for tools and resources."""
    page += mk.MkTemplate("docs/config_file/env_config.md")


@nav.route.page("Inheritance", icon="octicon:git-merge-16")
def _(page: mk.MkPage):
    """Configuration inheritance system."""
    page += mk.MkTemplate("docs/config_file/inheritance.md")


if __name__ == "__main__":
    print(nav.to_markdown())
