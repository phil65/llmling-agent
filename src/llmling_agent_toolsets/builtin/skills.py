"""Provider for skills tools."""

from __future__ import annotations

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import StaticResourceProvider


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


class SkillsTools(StaticResourceProvider):
    """Provider for Claude Code Skills tools.

    Provides tools to discover and load skills from the pool's skills registry.
    Skills are discovered from configured directories (e.g., ~/.claude/skills/,
    .claude/skills/).

    The pool manages skill discovery; this toolset just provides access to them.
    """

    def __init__(self, name: str = "skills") -> None:
        super().__init__(name=name)
        self._tools = [
            self.create_tool(load_skill, category="read", read_only=True, idempotent=True),
            self.create_tool(list_skills, category="read", read_only=True, idempotent=True),
        ]
