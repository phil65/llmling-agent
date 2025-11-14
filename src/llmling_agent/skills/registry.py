"""Claude Code Skills registry with auto-discovery."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar

from upath import UPath
from upathtools.helpers import upath_to_fs

from llmling_agent.skills.skill import Skill
from llmling_agent.tools.exceptions import ToolError
from llmling_agent.utils.baseregistry import BaseRegistry


if TYPE_CHECKING:
    from collections.abc import Sequence

    from fsspec.asyn import AsyncFileSystem
    from upath.types import JoinablePathLike


SKILL_NAME_LIMIT = 64
SKILL_DESCRIPTION_LIMIT = 1024


class SkillsRegistry(BaseRegistry[str, Skill]):
    """Registry for Claude Code Skills with auto-discovery."""

    DEFAULT_SKILL_PATHS: ClassVar = ["~/.claude/skills/", ".claude/skills/"]

    def __init__(self, skills_dirs: Sequence[JoinablePathLike] | None = None) -> None:
        """Initialize with custom skill directories or auto-detect."""
        super().__init__()
        if skills_dirs:
            self.skills_dirs = [UPath(i) for i in skills_dirs or []]
        else:
            self.skills_dirs = [UPath(i) for i in self.DEFAULT_SKILL_PATHS or []]

    async def discover_skills(self) -> None:
        """Scan filesystem and register all found skills.

        Args:
            filesystem: Optional async filesystem to use. If None, will use upath_to_fs()
                       to get appropriate filesystem for each skills directory.
        """
        for skills_dir in self.skills_dirs:
            fs = upath_to_fs(skills_dir)
            try:
                # List entries in skills directory
                entries = await fs._ls("/", detail=True)
            except FileNotFoundError:
                continue

            # Filter for directories that might contain skills
            skill_dirs = [entry for entry in entries if entry.get("type") == "directory"]

            for skill_entry in skill_dirs:
                skill_dir_path = UPath(skill_entry["name"])
                skill_file_path = skill_dir_path / "SKILL.md"
                try:
                    await fs._cat(str(skill_file_path))
                except FileNotFoundError:
                    continue

                try:
                    skill = await self._parse_skill(skill_dir_path, skills_dir, fs)
                    self.register(skill.name, skill, replace=True)
                except Exception as e:  # noqa: BLE001
                    # Log but don't fail discovery for one bad skill
                    print(f"Warning: Failed to parse skill at {skill_dir_path}: {e}")

    async def _parse_skill(
        self,
        skill_dir: JoinablePathLike,
        source_dir: JoinablePathLike,
        filesystem: AsyncFileSystem,
    ) -> Skill:
        """Parse a SKILL.md file and extract metadata."""
        skill_file = UPath(skill_dir) / "SKILL.md"
        skill_content = await filesystem._cat(str(skill_file))
        content = (
            skill_content.decode("utf-8")
            if isinstance(skill_content, bytes)
            else str(skill_content)
        )

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
            msg = (
                f"{skill_file}: Skill description exceeds {SKILL_DESCRIPTION_LIMIT} chars"
            )
            raise ToolError(msg)

        return Skill(
            name=name,
            description=description,
            skill_path=UPath(skill_dir),
            source=UPath(source_dir),
        )

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

    async def main():
        reg = SkillsRegistry()
        await reg.discover_skills()
        print(dict(reg))

    asyncio.run(main())
