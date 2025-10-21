"""Claude Code Skills registry with auto-discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, ClassVar

from llmling.core.baseregistry import BaseRegistry
import yaml

from .exceptions import ToolError


SKILL_NAME_LIMIT = 64
SKILL_DESCRIPTION_LIMIT = 1024


@dataclass
class Skill:
    """A Claude Code Skill with metadata and lazy-loaded instructions."""

    name: str
    description: str
    skill_path: Path
    source: Path  # Directory where skill was discovered
    instructions: str | None = None

    def load_instructions(self) -> str:
        """Lazy load full instructions from SKILL.md."""
        if self.instructions is None:
            skill_file = self.skill_path / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text(encoding="utf-8")
                # Split on first --- after frontmatter
                parts = content.split("---", 2)
                if len(parts) >= 3:  # noqa: PLR2004
                    self.instructions = parts[2].strip()
                else:
                    self.instructions = ""
            else:
                self.instructions = ""
        return self.instructions


class SkillsRegistry(BaseRegistry[str, Skill]):
    """Registry for Claude Code Skills with auto-discovery."""

    # Default skill discovery paths (can be overridden by subclasses)
    DEFAULT_SKILL_PATHS: ClassVar = [
        "~/.claude/skills",  # Global user skills
        ".claude/skills",  # Project-local skills (walks up tree)
    ]

    def __init__(self, skills_dirs: list[Path] | None = None) -> None:
        """Initialize with custom skill directories or auto-detect."""
        super().__init__()
        self.skills_dirs = skills_dirs or self._get_default_skills_dirs()

    def _get_default_skills_dirs(self) -> list[Path]:
        """Get skill directories from class attribute paths."""
        dirs = []

        for path_pattern in self.DEFAULT_SKILL_PATHS:
            if path_pattern.startswith("~/"):
                # Expand home directory
                resolved_path = Path.home() / path_pattern[2:]
                if resolved_path.exists():
                    dirs.append(resolved_path)
            elif not path_pattern.startswith("/"):
                # Relative path - walk up directory tree
                cwd = Path.cwd()
                for parent in [cwd, *list(cwd.parents)]:
                    candidate = parent / path_pattern
                    if candidate.exists():
                        dirs.append(candidate)
                        break
            else:
                # Absolute path
                absolute_path = Path(path_pattern)
                if absolute_path.exists():
                    dirs.append(absolute_path)

        return dirs

    async def discover_skills(self) -> None:
        """Scan filesystem and register all found skills."""
        for skills_dir in self.skills_dirs:
            if not skills_dir.exists():
                continue

            for skill_dir in skills_dir.iterdir():
                if not skill_dir.is_dir():
                    continue

                skill_file = skill_dir / "SKILL.md"
                if not skill_file.exists():
                    continue

                try:
                    skill = self._parse_skill(skill_dir, skills_dir)
                    self.register(skill.name, skill, replace=True)
                except Exception as e:  # noqa: BLE001
                    # Log but don't fail discovery for one bad skill
                    print(f"Warning: Failed to parse skill at {skill_dir}: {e}")

    def _parse_skill(self, skill_dir: Path, source_dir: Path) -> Skill:
        """Parse a SKILL.md file and extract metadata."""
        skill_file = skill_dir / "SKILL.md"
        content = skill_file.read_text(encoding="utf-8")

        # Extract YAML frontmatter
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not frontmatter_match:
            msg = f"No YAML frontmatter found in {skill_file}"
            raise ToolError(msg)

        try:
            metadata = yaml.safe_load(frontmatter_match.group(1))
        except yaml.YAMLError as e:
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
            skill_path=skill_dir,
            source=source_dir,
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
