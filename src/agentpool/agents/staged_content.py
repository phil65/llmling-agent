"""Staged content."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai import UserPromptPart

from agentpool.log import get_logger


if TYPE_CHECKING:
    from pydantic_ai import SystemPromptPart


logger = get_logger(__name__)


@dataclass
class StagedContent:
    """Buffer for prompt parts to be injected into the next agent call.

    This allows commands (like /fetch-repo, /git-diff) to stage content that will
    be automatically included in the next prompt sent to the agent.
    """

    _parts: list[SystemPromptPart | UserPromptPart] = field(default_factory=list)

    def add(self, parts: list[SystemPromptPart | UserPromptPart]) -> None:
        """Add prompt parts to the staging area."""
        self._parts.extend(parts)

    def add_text(self, content: str) -> None:
        """Add text content to the staging area as a UserPromptPart."""
        self._parts.append(UserPromptPart(content=content))

    async def consume(self) -> list[SystemPromptPart | UserPromptPart]:
        """Return all staged parts and clear the buffer."""
        parts = self._parts.copy()
        self._parts.clear()
        return parts

    async def consume_as_text(self) -> str | None:
        """Return all staged content as a single string and clear the buffer.

        Returns:
            Combined text content, or None if nothing staged.
        """
        if not self._parts:
            return None
        texts = [part.content for part in self._parts if isinstance(part.content, str)]
        self._parts.clear()
        content = "\n\n".join(texts) if texts else None
        return f"\n<context>\n{content}\n</context>\n" if content else None

    def __len__(self) -> int:
        """Return count of staged parts."""
        return len(self._parts)
