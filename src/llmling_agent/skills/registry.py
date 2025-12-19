"""Claude Code Skills registry with auto-discovery."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar

from fsspec import AbstractFileSystem
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.asyn_wrapper import (
    AsyncFileSystemWrapper,
)
from upathtools import is_directory
from upathtools.helpers import to_upath, upath_to_fs

from llmling_agent.log import get_logger
from llmling_agent.skills.skill import Skill
from llmling_agent.tools.exceptions import ToolError
from llmling_agent.utils.baseregistry import BaseRegistry


if TYPE_CHECKING:
    from collections.abc import Sequence

    from upathtools import JoinablePathLike, UPath


SKILL_NAME_LIMIT = 64
SKILL_DESCRIPTION_LIMIT = 1024
logger = get_logger(__name__)


class SkillsRegistry(BaseRegistry[str, Skill]):
    """Registry for Claude Code Skills with auto-discovery."""

    DEFAULT_SKILL_PATHS: ClassVar = ["~/.claude/skills/", ".claude/skills/"]

    def __init__(self, skills_dirs: Sequence[JoinablePathLike] | None = None) -> None:
        """Initialize with custom skill directories or auto-detect."""
        super().__init__()
        if skills_dirs:
            self.skills_dirs = [to_upath(i) for i in skills_dirs or []]
        else:
            self.skills_dirs = [to_upath(i) for i in self.DEFAULT_SKILL_PATHS or []]

    async def discover_skills(self) -> None:
        """Scan filesystem and register all found skills.

        Args:
            filesystem: Optional async filesystem to use. If None, will use upath_to_fs()
                       to get appropriate filesystem for each skills directory.
        """
        for skills_dir in self.skills_dirs:
            await self.register_skills_from_path(skills_dir)

    async def register_skills_from_path(
        self,
        skills_dir: JoinablePathLike | AbstractFileSystem,
        base_path: str | None = None,
        **storage_options: Any,
    ) -> None:
        """Register skills from a given path.

        Args:
            skills_dir: Path to the directory containing skills, or filesystem instance.
            base_path: When skills_dir is a filesystem, the path within that filesystem
                      to look for skills. Defaults to root_marker if not specified.
            storage_options: Additional options to pass to the filesystem.
        """
        if isinstance(skills_dir, AbstractFileSystem):
            fs = skills_dir
            if not isinstance(fs, AsyncFileSystem):
                fs = AsyncFileSystemWrapper(fs)
            search_path = base_path if base_path is not None else fs.root_marker
        else:
            fs = upath_to_fs(skills_dir, **storage_options)
            search_path = fs.root_marker

        try:
            # List entries in skills directory
            entries = await fs._ls(search_path, detail=True)
        except FileNotFoundError:
            logger.warning("Skills directory not found", path=search_path)
            return
        # Filter for directories that might contain skills
        skill_dirs = [
            entry
            for entry in entries
            if await is_directory(fs, entry["name"], entry_type=entry.get("type"))
        ]
        if not skill_dirs:
            logger.info("No skills found", skills_dir=search_path)
            return
        logger.info("Found skills", skills=skill_dirs, skills_dir=search_path)
        for skill_entry in skill_dirs:
            skill_name = skill_entry["name"].lstrip("./")
            # Construct full path for skill directory
            if search_path and search_path != fs.root_marker:
                skill_dir_path = to_upath(f"{search_path}/{skill_name}")
                skill_md_path = f"{search_path}/{skill_name}/SKILL.md"
            else:
                skill_dir_path = to_upath(skill_name)
                skill_md_path = f"{skill_name}/SKILL.md"
            try:
                await fs._cat_file(skill_md_path)
            except FileNotFoundError:
                continue

            try:
                skill = self._parse_skill(skill_dir_path)
                self.register(skill.name, skill, replace=True)
            except Exception as e:  # noqa: BLE001
                # Log but don't fail discovery for one bad skill
                print(f"Warning: Failed to parse skill at {skill_dir_path}: {e}")

    def _parse_skill(self, skill_dir: JoinablePathLike) -> Skill:
        """Parse a SKILL.md file and extract metadata."""
        skill_file = to_upath(skill_dir) / "SKILL.md"
        content = skill_file.read_text("utf-8")

        # Extract YAML frontmatter
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not frontmatter_match:
            msg = f"No YAML frontmatter found in {skill_file}"
            raise ToolError(msg)
        import yamling

        try:
            metadata = yamling.load_yaml(frontmatter_match.group(1))
        except yamling.YAMLError as e:
            msg = f"Invalid YAML frontmatter in {skill_file}: {e}"
            raise ToolError(msg) from e

        # Validate required fields
        if not isinstance(metadata, dict):
            msg = f"YAML frontmatter must be a dictionary in {skill_file}"
            raise ToolError(msg)

        name = metadata.get("name")
        description = metadata.get("description")

        if not name:
            msg = f"Missing 'name' field in {skill_file}"
            raise ToolError(msg)
        if not description:
            msg = f"Missing 'description' field in {skill_file}"
            raise ToolError(msg)

        # Validate limits
        if len(name) > SKILL_NAME_LIMIT:
            msg = f"{skill_file}: Skill name exceeds {SKILL_NAME_LIMIT} chars"
            raise ToolError(msg)
        if len(description) > SKILL_DESCRIPTION_LIMIT:
            msg = f"{skill_file}: Skill description exceeds {SKILL_DESCRIPTION_LIMIT} chars"
            raise ToolError(msg)

        return Skill(name=name, description=description, skill_path=to_upath(skill_dir))

    @property
    def _error_class(self) -> type[ToolError]:
        """Error class to use for this registry."""
        return ToolError

    def _validate_item(self, item: Any) -> Skill:
        """Validate and possibly transform item before registration."""
        if not isinstance(item, Skill):
            msg = f"Expected Skill instance, got {type(item)}"
            raise ToolError(msg)
        return item

    def get_skill_instructions(self, skill_name: str) -> str:
        """Lazy load full instructions for a skill."""
        skill = self.get(skill_name)
        return skill.load_instructions()


if __name__ == "__main__":
    import asyncio
    import os

    from upathtools import UPath

    from llmling_agent.log import configure_logging

    configure_logging()

    async def main() -> None:
        reg = SkillsRegistry()
        p = UPath(
            "github://",
            token=os.getenv("GITHUB_TOKEN"),
            username="phil65",
            org="anthropics",
            repo="skills",
        )
        print("Repository contents:")
        print([f.name for f in p.iterdir()][:5])  # Show first 5 items

        await reg.register_skills_from_path(
            p,
            token=os.getenv("GITHUB_TOKEN"),
            username="phil65",
            org="anthropics",
            repo="skills",
        )
        print(f"Found {len(reg)} skills:")
        for name, skill in reg.items():
            print(f"  - {name}: {skill.description[:60]}...")

    asyncio.run(main())
