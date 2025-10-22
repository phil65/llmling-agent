"""Provider for integration tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling_agent.tools.skills import SkillsRegistry


class IntegrationTools(ResourceProvider):
    """Provider for integration tools."""

    def __init__(
        self, name: str = "integrations", skills_registry: SkillsRegistry | None = None
    ):
        super().__init__(name)
        self.skills_registry = skills_registry

    async def get_tools(self) -> list[Tool]:
        """Get integration tools with dynamic skill tool."""
        from llmling_agent_tools import capability_tools

        tools = [
            Tool.from_callable(
                capability_tools.add_local_mcp_server,
                source="builtin",
                category="other",
            ),
            Tool.from_callable(
                capability_tools.add_remote_mcp_server,
                source="builtin",
                category="other",
            ),
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
                capability_tools.load_skill,
                source="builtin",
                category="read",
                description_override=description,
            )
            tools.append(skill_tool)

        return tools
