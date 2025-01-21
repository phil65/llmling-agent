from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from llmling_agent.models.converters import (
    ConversionConfig,
    MarkItDownConfig,
    PlainConverterConfig,
)
from llmling_agent.utils.async_read import read_path


if TYPE_CHECKING:
    from os import PathLike

    from llmling_agent_converters.base import DocumentConverter


class ConversionManager:
    """Manages document conversion using configured providers.

    In order to not make things super complex, all Converters will be implemented as sync.
    The manager will handle async I/O and thread pooling.
    """

    def __init__(self, config: ConversionConfig | list[DocumentConverter]):
        if isinstance(config, list):
            self.config = ConversionConfig()
            self._converters = config
        else:
            self.config = config
            self._converters = self._setup_converters()
        self._executor = ThreadPoolExecutor(max_workers=3)

    def __del__(self):
        self._executor.shutdown(wait=False)

    def _setup_converters(self) -> list[DocumentConverter]:
        """Create converter instances from config."""
        converters: list[DocumentConverter] = []
        for cfg in self.config.providers or []:
            if not cfg.enabled:
                continue
            match cfg:
                case MarkItDownConfig():
                    from llmling_agent_converters.markitdown_converter import (
                        MarkItDownConverter,
                    )

                    converters.append(MarkItDownConverter(cfg))
                case PlainConverterConfig():
                    from llmling_agent_converters.plain_converter import PlainConverter

                    converters.append(PlainConverter(cfg))
        # Always add PlainConverter as fallback
        # if it gets configured by user, that one gets preference.
        converters.append(PlainConverter(PlainConverterConfig()))
        return converters

    async def convert_file(self, path: str | PathLike[str]) -> str:
        """Convert file using first supporting converter."""
        loop = asyncio.get_running_loop()
        content = await read_path(path, "rb")

        for converter in self._converters:
            # Run support check in thread pool
            supports = await loop.run_in_executor(
                self._executor, converter.supports_file, path
            )
            if not supports:
                continue
            # Run conversion in thread pool
            import mimetypes

            return await loop.run_in_executor(
                self._executor,
                converter.convert_content,
                content,
                mimetypes.guess_type(str(path))[0],
            )
        return str(content)

    async def convert_content(self, content: Any, mime_type: str | None = None) -> str:
        """Convert content using first supporting converter."""
        loop = asyncio.get_running_loop()

        for converter in self._converters:
            # Run support check in thread pool
            supports = await loop.run_in_executor(
                self._executor, converter.supports_content, content, mime_type
            )
            if not supports:
                continue

            # Run conversion in thread pool
            return await loop.run_in_executor(
                self._executor, converter.convert_content, content, mime_type
            )

        return str(content)  # Fallback for unsupported content

    def convert_file_sync(self, path: str | PathLike[str]) -> str:
        """Sync wrapper for convert_file."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.convert_file(path))
        except RuntimeError:
            # No running loop - create new one
            return asyncio.run(self.convert_file(path))

    def convert_content_sync(self, content: Any, mime_type: str | None = None) -> str:
        """Sync wrapper for convert_content."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.convert_content(content, mime_type))
        except RuntimeError:
            # No running loop - create new one
            return asyncio.run(self.convert_content(content, mime_type))


if __name__ == "__main__":
    from llmling_agent.models.converters import ConversionConfig

    config = ConversionConfig(
        providers=[
            MarkItDownConfig(enabled=True),
            PlainConverterConfig(enabled=True),
        ]
    )
    manager = ConversionManager(config)
    print(manager.convert_file_sync("pyproject.toml"))
