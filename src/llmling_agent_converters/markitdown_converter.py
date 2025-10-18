"""MarkItDown converter."""

from __future__ import annotations

from functools import cached_property
import tempfile
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent_config.converters import MarkItDownConfig
from llmling_agent_converters.base import DocumentConverter


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


logger = get_logger(__name__)


class MarkItDownConverter(DocumentConverter):
    """Converter using MarkItDown for document conversion."""

    def __init__(self, config: MarkItDownConfig | None = None):
        self.config = config or MarkItDownConfig()

    @cached_property
    def converter(self):
        from markitdown import MarkItDown  # pyright: ignore

        return MarkItDown()

    def supports_file(self, path: JoinablePathLike) -> bool:
        """Accept any file - MarkItDown is good at detecting formats."""
        return True

    def supports_content(self, content: Any, mime_type: str | None = None) -> bool:
        """Only supports files, not raw content."""
        return False

    def convert_content(self, content: Any, mime_type: str | None = None) -> str:
        """Convert content to markdown via temporary file."""
        try:
            # Determine appropriate file extension
            suffix = ".txt"
            if mime_type:
                import mimetypes

                if ext := mimetypes.guess_extension(mime_type):
                    suffix = ext

            # Write content to temporary file
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                if isinstance(content, str):
                    tmp.write(content.encode())
                elif isinstance(content, bytes):
                    tmp.write(content)
                else:
                    tmp.write(str(content).encode())
                tmp.flush()

                result = self.converter.convert(tmp.name)
                return result.text_content

        except Exception as e:
            msg = "Failed to convert content"
            logger.exception(msg)
            raise ValueError(msg) from e

    def convert_file(self, path: JoinablePathLike) -> str:
        """Convert using MarkItDown's file-based interface."""
        try:
            from upathtools import to_upath

            path_obj = to_upath(path)

            # Direct handling for local paths and http(s) URLs
            if path_obj.protocol in {"", "file", "http", "https"}:
                result = self.converter.convert(path_obj.path)
            else:
                # For other protocols, use temporary file
                with tempfile.NamedTemporaryFile(suffix=path_obj.suffix) as tmp:
                    tmp.write(path_obj.read_bytes())
                    tmp.flush()
                    result = self.converter.convert(tmp.name)

        except Exception as e:
            msg = f"Failed to convert file {path}"
            logger.exception(msg)
            raise ValueError(msg) from e
        else:
            return result.text_content
