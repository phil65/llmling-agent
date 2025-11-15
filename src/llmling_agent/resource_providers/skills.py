"""Skills resource provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.skills.registry import SkillsRegistry
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from fsspec.utils import Sequence
    from upath.types import JoinablePathLike

    from llmling_agent.skills.skill import Skill


logger = get_logger(__name__)

BASE_DESC = """Load a Claude Code Skill and return its instructions.

This tool provides access to Claude Code Skills - specialized workflows and techniques
for handling specific types of tasks. When you need to use a skill, call this tool
with the skill name.

Available skills:"""


async def load_skill(ctx: AgentContext, skill_name: str) -> str:  # noqa: D417
    """Load a Claude Code Skill and return its instructions.

    Args:
        skill_name: Name of the skill to load

    Returns:
        The full skill instructions for execution
    """
    # Get skills provider from agent's resource providers
    skills_provider = None
    for provider in ctx.agent.tools.providers:
        if isinstance(provider, SkillsResourceProvider):
            skills_provider = provider
            break

    if not skills_provider:
        return "No skills provider found in agent configuration"

    try:
        instructions = await skills_provider.get_skill_instructions(skill_name)
        skill = next(
            (s for s in await skills_provider.get_skills() if s.name == skill_name),
            None,
        )

        if not skill:
            return f"Skill {skill_name!r} not found"

        # Format the skill content for Claude to follow
        return f"""
<command-message>The "{skill_name}" skill is loading</command-message>

# {skill.name}

{instructions}

---
Skill directory: {skill.skill_path}
"""  # noqa: TRY300
    except Exception as e:  # noqa: BLE001
        return f"Failed to load skill {skill_name!r}: {e}"


class SkillsResourceProvider(ResourceProvider):
    """Resource provider for Claude Code Skills."""

    def __init__(
        self,
        registry: SkillsRegistry | None = None,
        skills_dirs: Sequence[JoinablePathLike] | None = None,
        name: str = "skills",
        owner: str | None = None,
    ):
        """Initialize the skills provider.

        Args:
            registry: Existing skills registry to use
            skills_dirs: Directories to search for skills (if no registry provided)
            name: Name for this provider
            owner: Owner of this provider
        """
        super().__init__(name, owner)
        self.registry = registry or SkillsRegistry(skills_dirs)

    async def __aenter__(self) -> Self:
        """Initialize skills provider and discover skills."""
        await self.registry.discover_skills()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Cleanup skills provider resources."""
        # Skills are file-based, no persistent connections to clean up

    async def get_skills(self) -> list[Skill]:
        """Get all available skills (already discovered in __aenter__)."""
        return [self.registry.get(name) for name in self.registry.list_items()]

    async def get_skill_instructions(self, skill_name: str) -> str:
        """Get full instructions for a specific skill."""
        return self.registry.get_skill_instructions(skill_name)

    async def get_tools(self) -> list[Tool]:
        """Get skills loading tool with dynamic description."""
        skills = await self.get_skills()

        if not skills:
            # No tool if no skills available
            return []

        skills_list = [f"- {s.name}: {s.description}" for s in skills]
        return [
            Tool.from_callable(
                load_skill,
                source="skills",
                category="read",
                description_override=BASE_DESC + "\n" + "\n".join(skills_list),
            )
        ]

    async def refresh(self) -> None:
        """Force rediscovery of skills."""
        await self.registry.discover_skills()


if __name__ == "__main__":
    import asyncio

    async def main():
        async with SkillsResourceProvider() as provider:
            await provider.refresh()
            print(await provider.get_tools())

    asyncio.run(main())
