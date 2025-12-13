"""Claude Code Skill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from upathtools import UPath


@dataclass
class Skill:
    """A Claude Code Skill with metadata and lazy-loaded instructions."""

    name: str
    description: str
    skill_path: UPath
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
