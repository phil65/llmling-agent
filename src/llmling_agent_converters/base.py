"""Base class for converters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import mimetypes
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


class DocumentConverter(ABC):
    """Base class for document converters."""

    def convert_file(self, path: JoinablePathLike) -> str:
        """Convert document file to markdown."""
        from upathtools import to_upath

        path_obj = to_upath(path)
        content = path_obj.read_bytes()
        return self.convert_content(content, mimetypes.guess_type(str(path))[0])

    @abstractmethod
    def convert_content(self, content: Any, mime_type: str | None = None) -> str:
        """Convert content to markdown."""

    @abstractmethod
    def supports_file(self, path: JoinablePathLike) -> bool:
        """Check if converter can handle this file type."""

    @abstractmethod
    def supports_content(self, content: Any, mime_type: str | None = None) -> bool:
        """Check if converter can handle this content type."""
