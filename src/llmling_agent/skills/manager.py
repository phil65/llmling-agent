"""Skills manager for pool-wide management."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Self

from upath import UPath

from llmling_agent.log import get_logger
from llmling_agent.resource_providers.skills import SkillsResourceProvider
from llmling_agent.skills.registry import SkillsRegistry


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


logger = get_logger(__name__)


class SkillsManager:
    """Manages skills discovery and distributes skills provider to agents."""

    def __init__(
        self,
        name: str = "skills",
        owner: str | None = None,
        skills_dirs: list[JoinablePathLike] | None = None,
    ) -> None:
        """Initialize the skills manager.

        Args:
            name: Name for this manager
            owner: Owner of this manager
            skills_dirs: Directories to search for skills
        """
        self.name = name
        self.owner = owner
        self.registry = SkillsRegistry(skills_dirs)
        self.provider = SkillsResourceProvider(self.registry, name=name, owner=owner)
        self.exit_stack = AsyncExitStack()
        self._initialized = False

    def __repr__(self) -> str:
        skill_count = len(self.registry.list_items()) if self._initialized else "?"
        return f"SkillsManager(name={self.name!r}, skills={skill_count})"

    async def __aenter__(self) -> Self:
        """Initialize the skills manager."""
        try:
            # Initialize the provider through its async context manager
            await self.exit_stack.enter_async_context(self.provider)
            self._initialized = True
            count = len(self.registry.list_items())
            logger.info("Skills manager initialized", name=self.name, skill_count=count)
        except Exception as e:
            msg = "Failed to initialize skills manager"
            logger.exception(msg, name=self.name, error=e)
            raise
        return self

    async def __aexit__(self, *args: object) -> None:
        """Clean up the skills manager."""
        await self.exit_stack.aclose()

    def get_skills_provider(self) -> SkillsResourceProvider:
        """Get the skills resource provider for agents."""
        return self.provider

    async def refresh_skills(self) -> None:
        """Refresh skills discovery."""
        await self.provider.refresh()
        skill_count = len(self.registry.list_items())
        logger.info("Skills refreshed", name=self.name, skill_count=skill_count)

    def add_skills_directory(self, path: JoinablePathLike) -> None:
        """Add a new skills directory to search."""
        if path not in self.registry.skills_dirs:
            self.registry.skills_dirs.append(UPath(path))
            logger.info("Added skills directory", path=str(path))
