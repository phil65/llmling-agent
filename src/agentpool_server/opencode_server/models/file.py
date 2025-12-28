"""File operation models."""

from typing import Literal

from pydantic import Field

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


class FindMatch(OpenCodeBaseModel):
    """Text search match."""

    path: TextWrapper
    lines: TextWrapper
    line_number: int = Field(alias="line_number")
    absolute_offset: int = Field(alias="absolute_offset")
    submatches: list[SubmatchInfo] = Field(default_factory=list)


class Symbol(OpenCodeBaseModel):
    """Code symbol."""

    name: str
    kind: str
    path: str
    line: int
    character: int
