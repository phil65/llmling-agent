"""Provider for skills and commands tools."""

from __future__ import annotations

from agentpool.agents.context import AgentContext  # noqa: TC001
from agentpool.resource_providers import StaticResourceProvider


BASE_DESC = """Load a Claude Code Skill and return its instructions.

This tool provides access to Claude Code Skills - specialized workflows and techniques
for handling specific types of tasks. When you need to use a skill, call this tool
with the skill name.

Available skills:"""


async def load_skill(ctx: AgentContext, skill_name: str) -> str:
    """Load a Claude Code Skill and return its instructions.

    Args:
        ctx: Agent context providing access to pool and skills
        skill_name: Name of the skill to load

    Returns:
        The full skill instructions for execution
    """
    if not ctx.pool:
        return "No agent pool available - skills require pool context"

    skills = ctx.pool.skills.list_skills()
    if not skills:
        return "No skills available."
    skill = next((s for s in skills if s.name == skill_name), None)
    if not skill:
        available = ", ".join(s.name for s in skills)
        return f"Skill {skill_name!r} not found. Available skills: {available}"

    try:
        instructions = ctx.pool.skills.get_skill_instructions(skill_name)
    except Exception as e:  # noqa: BLE001
        return f"Failed to load skill {skill_name!r}: {e}"
    return f"# {skill.name}\n{instructions}\nSkill directory: {skill.skill_path}"


async def list_skills(ctx: AgentContext) -> str:
    """List all available skills.

    Returns:
        Formatted list of available skills with descriptions
    """
    if not ctx.pool:
        return "No agent pool available - skills require pool context"
    skills = ctx.pool.skills.list_skills()
    if not skills:
        return "No skills available"
    lines = ["Available skills:", ""]
    lines.extend(f"- **{skill.name}**: {skill.description}" for skill in skills)
    return "\n".join(lines)


class _StringOutputWriter:
    """Output writer that captures output to a string buffer."""

    def __init__(self) -> None:
        from io import StringIO

        self._buffer = StringIO()

    async def print(self, message: str) -> None:
        """Write a message to the buffer."""
        self._buffer.write(message)
        self._buffer.write("\n")

    def getvalue(self) -> str:
        """Get the captured output."""
        return self._buffer.getvalue()


async def run_command(ctx: AgentContext, command: str) -> str:  # noqa: D417
    """Execute an internal command.

    This provides access to the agent's internal CLI for management operations.
    Use `help` to see all available commands, or `help <command>` for details.

    Commands cover:
    - Agent/team management: create-agent, create-team, list-agents
    - Tool management: list-tools, register-tool, enable-tool, disable-tool
    - MCP servers: add-mcp-server, add-remote-mcp-server, list-mcp-servers
    - Connections: connect, disconnect, connections
    - Workers: add-worker, remove-worker, list-workers
    - And more...

    Args:
        command: The command to execute (e.g., "help", "list-tools", "create-agent myagent").
                 Leading slash is optional.

    Returns:
        Command output or error message
    """
    from slashed import CommandContext

    if not ctx.agent.command_store:
        return "No command store available"

    # Remove leading slash if present (slashed expects command name without /)
    cmd = command.lstrip("/")

    # Create output capture
    output = _StringOutputWriter()

    # Create CommandContext with output capture and AgentContext as data
    cmd_ctx = CommandContext(
        output=output,
        data=ctx,
        command_store=ctx.agent.command_store,
    )

    try:
        await ctx.agent.command_store.execute_command(cmd, cmd_ctx)
        result = output.getvalue()
    except Exception as e:  # noqa: BLE001
        return f"Command failed: {e}"
    else:
        return result if result else "Command executed successfully."


class SkillsTools(StaticResourceProvider):
    """Provider for skills and commands tools.

    Provides tools to:
    - Discover and load skills from the pool's skills registry
    - Execute internal commands via the agent's command system

    Skills are discovered from configured directories (e.g., ~/.claude/skills/,
    .claude/skills/).

    Commands provide access to management operations like creating agents,
    managing tools, connecting nodes, etc. Use run_command("/help") to discover
    available commands.
    """

    def __init__(self, name: str = "skills") -> None:
        super().__init__(name=name)
        self._tools = [
            self.create_tool(load_skill, category="read", read_only=True, idempotent=True),
            self.create_tool(list_skills, category="read", read_only=True, idempotent=True),
            self.create_tool(
                run_command,
                category="other",
                read_only=False,
                idempotent=False,
            ),
        ]
