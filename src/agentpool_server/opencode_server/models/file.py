"""File operation models."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class FileNode(OpenCodeBaseModel):
    """File or directory node."""

    name: str
    path: str
    type: Literal["file", "directory"]
    size: int | None = None


class FileContent(OpenCodeBaseModel):
    """File content response."""

    path: str
    content: str
    encoding: str = "utf-8"


class FileStatus(OpenCodeBaseModel):
    """File status (for VCS)."""

    path: str
    status: str  # modified, added, deleted, etc.


class TextWrapper(OpenCodeBaseModel):
    """Wrapper for text content."""

    text: str


class SubmatchInfo(OpenCodeBaseModel):
    """Submatch information."""

    match: TextWrapper
    start: int
    end: int

    @classmethod
    def create(cls, text: str, start: int, end: int) -> Self:
        return cls(match=TextWrapper(text=text), start=start, end=end)


class FindMatch(BaseModel):
    """Text search match."""

    path: TextWrapper
    lines: TextWrapper
    line_number: int  # these here are snake_case in the API, so we inherit from BaseModel
    absolute_offset: int
    submatches: list[SubmatchInfo] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        path: str,
        lines: str,
        line_number: int,
        absolute_offset: int,
        submatches: list[SubmatchInfo] | None = None,
    ) -> FindMatch:
        return cls(
            path=TextWrapper(text=path),
            lines=TextWrapper(text=lines),
            line_number=line_number,
            absolute_offset=absolute_offset,
            submatches=submatches or [],
        )


class Symbol(OpenCodeBaseModel):
    """Code symbol."""

    name: str
    kind: str
    path: str
    line: int
    character: int
