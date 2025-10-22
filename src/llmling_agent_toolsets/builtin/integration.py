"""Provider for integration tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic_ai.tools import RunContext

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling_agent.tools.skills import SkillsRegistry
    from llmling_agent_config.mcp_server import MCPServerConfig


async def add_local_mcp_server(  # noqa: D417
    ctx: AgentContext,
    name: str,
    command: str,
    args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Add a local MCP server via stdio transport.

    Args:
        name: Unique name for the MCP server
        command: Command to execute for the server
        args: Command arguments
        env_vars: Environment variables to pass to the server

    Returns:
        Confirmation message about the added server
    """
    from llmling_agent_config.mcp_server import StdioMCPServerConfig

    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    env = env_vars or {}
    config = StdioMCPServerConfig(name=name, command=command, args=args or [], env=env)
    ctx.agent.mcp.add_server_config(config)
    await ctx.agent.mcp.setup_server(config)

    return f"Added local MCP server {name!r} with command: {command}"


async def add_remote_mcp_server(  # noqa: D417
    ctx: AgentContext,
    name: str,
    url: str,
    transport: Literal["sse", "streamable-http"] = "streamable-http",
) -> str:
    """Add a remote MCP server via HTTP-based transport.

    Args:
        name: Unique name for the MCP server
        url: Server URL endpoint
        transport: HTTP transport type to use (http is preferred)

    Returns:
        Confirmation message about the added server
    """
    from llmling_agent_config.mcp_server import (
        SSEMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )

    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    match transport:
        case "sse":
            config: MCPServerConfig = SSEMCPServerConfig(name=name, url=url)
        case "streamable-http":
            config = StreamableHTTPMCPServerConfig(name=name, url=url)

    ctx.agent.mcp.add_server_config(config)
    await ctx.agent.mcp.setup_server(config)

    return f"Added remote MCP server '{name}' at {url} using {transport} transport"


async def load_skill(ctx: AgentContext, skill_name: str) -> str:  # noqa: D417
    """Load a Claude Code Skill and return its instructions.

    Args:
        skill_name: Name of the skill to load

    Returns:
        The full skill instructions for execution
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    registry = ctx.agent.skills_registry
    await registry.discover_skills()

    try:
        skill = registry.get(skill_name)
        instructions = skill.load_instructions()

        # Format the skill content for Claude to follow
    except Exception as e:  # noqa: BLE001
        return f"Failed to load skill '{skill_name}': {e}"
    else:
        return f"""
<command-message>The "{skill_name}" skill is loading</command-message>

# {skill.name}

{instructions}

---
Skill loaded from: {skill.source}
Skill directory: {skill.skill_path}
"""


class IntegrationTools(ResourceProvider):
    """Provider for integration tools."""

    def __init__(
        self, name: str = "integrations", skills_registry: SkillsRegistry | None = None
    ):
        super().__init__(name)
        self.skills_registry = skills_registry

    async def get_tools(self) -> list[Tool]:
        """Get integration tools with dynamic skill tool."""
        tools = [
            Tool.from_callable(add_local_mcp_server, source="builtin", category="other"),
            Tool.from_callable(add_remote_mcp_server, source="builtin", category="other"),
        ]

        # Add skill loading tool if registry is available
        if self.skills_registry:
            await self.skills_registry.discover_skills()

            # Create skill tool with dynamic description including available skills
            base_desc = """Load a Claude Code Skill and return its instructions.

This tool provides access to Claude Code Skills - specialized workflows and techniques
for handling specific types of tasks. When you need to use a skill, call this tool
with the skill name.

Available skills:"""

            if self.skills_registry.is_empty:
                description = base_desc + "\n(No skills found in configured directories)"
            else:
                skills_list = []
                for skill_name in self.skills_registry.list_items():
                    skill = self.skills_registry.get(skill_name)
                    skills_list.append(f"- {skill.name}: {skill.description}")
                description = base_desc + "\n" + "\n".join(skills_list)

            skill_tool = Tool.from_callable(
                load_skill,
                source="builtin",
                category="read",
                description_override=description,
            )
            tools.append(skill_tool)

        return tools
