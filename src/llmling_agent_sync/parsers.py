"""Parsers for extracting sync metadata from different file types."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol

import yaml


if TYPE_CHECKING:
    from llmling_agent_sync.models import SyncMetadata


_BLOCK_PATTERN = re.compile(r"^# /// sync\s*\n((?:^#[^\n]*\n)*?)^# ///$", re.MULTILINE)


class MetadataParser(Protocol):
    """Protocol for file-type specific metadata extraction."""

    extensions: tuple[str, ...]
    """File extensions this parser handles."""

    def parse(self, content: str) -> SyncMetadata | None:
        """Extract sync metadata from file content."""
        ...

    def update(self, content: str, metadata: SyncMetadata) -> str:
        """Update/inject metadata back into file content."""
        ...


class PythonSyncParser:
    """Parser for Python files using PEP723-style blocks.

    Format:
        # /// sync
        # agent = "doc_sync_agent"
        # dependencies = ["src/models/*.py"]
        # urls = ["https://docs.example.com"]
        # [context]
        # key = "value"
        # ///
    """

    extensions: tuple[str, ...] = (".py", ".pyi")

    def parse(self, content: str) -> SyncMetadata | None:

        match = _BLOCK_PATTERN.search(content)
        if not match:
            return None

        # Strip comment prefixes and parse as TOML-like
        block = match.group(1)
        lines = [line.lstrip("#").strip() for line in block.splitlines()]
        text = "\n".join(lines)

        return self._parse_toml_like(text)

    def _parse_toml_like(self, text: str) -> SyncMetadata:
        """Simple TOML-like parser for the metadata block."""
        import tomllib

        from llmling_agent_sync.models import SyncMetadata

        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return SyncMetadata()

        return SyncMetadata(
            agent=data.get("agent"),
            dependencies=data.get("dependencies", []),
            urls=data.get("urls", []),
            context=data.get("context", {}),
            last_checked=data.get("last_checked"),
        )

    def update(self, content: str, metadata: SyncMetadata) -> str:
        """Update or inject sync block into Python file."""
        new_block = self._format_block(metadata)
        match = _BLOCK_PATTERN.search(content)

        if match:
            return content[: match.start()] + new_block + content[match.end() :]

        # Inject after module docstring or at top
        return self._inject_at_top(content, new_block)

    def _format_block(self, metadata: SyncMetadata) -> str:
        """Format metadata as a sync block."""
        lines = ["# /// sync"]
        if metadata.agent:
            lines.append(f'# agent = "{metadata.agent}"')
        if metadata.dependencies:
            deps = ", ".join(f'"{d}"' for d in metadata.dependencies)
            lines.append(f"# dependencies = [{deps}]")
        if metadata.urls:
            urls = ", ".join(f'"{u}"' for u in metadata.urls)
            lines.append(f"# urls = [{urls}]")
        if metadata.last_checked:
            lines.append(f'# last_checked = "{metadata.last_checked}"')
        if metadata.context:
            lines.append("# [context]")
            for key, value in metadata.context.items():
                lines.append(f'# {key} = "{value}"')

        lines.append("# ///")
        return "\n".join(lines) + "\n"

    def _inject_at_top(self, content: str, block: str) -> str:
        """Inject block after docstring or at very top."""
        # Simple heuristic: after first triple-quoted block if present
        docstring_pattern = re.compile(r'^("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*\n')
        if match := docstring_pattern.match(content):
            pos = match.end()
            return content[:pos] + "\n" + block + content[pos:]
        return block + "\n" + content


class MarkdownSyncParser:
    """Parser for Markdown files using YAML frontmatter.

    Format:
        ---
        sync:
          agent: doc_sync_agent
          dependencies:
            - src/models/*.py
          urls:
            - https://docs.example.com
          context:
            key: value
        ---
    """

    extensions: tuple[str, ...] = (".md", ".mdx")
    _FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    def parse(self, content: str) -> SyncMetadata | None:
        from llmling_agent_sync.models import SyncMetadata

        match = self._FRONTMATTER_PATTERN.match(content)
        if not match:
            return None

        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None

        if not frontmatter or "sync" not in frontmatter:
            return None

        sync_data = frontmatter["sync"]
        return SyncMetadata(
            agent=sync_data.get("agent"),
            dependencies=sync_data.get("dependencies", []),
            urls=sync_data.get("urls", []),
            context=sync_data.get("context", {}),
            last_checked=sync_data.get("last_checked"),
        )

    def update(self, content: str, metadata: SyncMetadata) -> str:
        """Update or inject sync metadata in frontmatter."""
        sync_dict = self._metadata_to_dict(metadata)
        if match := self._FRONTMATTER_PATTERN.match(content):
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                frontmatter = {}

            frontmatter["sync"] = sync_dict
            new_frontmatter = yaml.dump(frontmatter, default_flow_style=False)
            return f"---\n{new_frontmatter}---\n" + content[match.end() :]

        # No frontmatter exists, create one
        new_frontmatter = yaml.dump({"sync": sync_dict}, default_flow_style=False)
        return f"---\n{new_frontmatter}---\n\n" + content

    def _metadata_to_dict(self, metadata: SyncMetadata) -> dict[str, Any]:
        """Convert metadata to dict for YAML serialization."""
        result: dict[str, Any] = {}
        if metadata.agent:
            result["agent"] = metadata.agent
        if metadata.dependencies:
            result["dependencies"] = metadata.dependencies
        if metadata.urls:
            result["urls"] = metadata.urls
        if metadata.last_checked:
            result["last_checked"] = metadata.last_checked
        if metadata.context:
            result["context"] = metadata.context

        return result


# Registry of built-in parsers
BUILTIN_PARSERS: list[MetadataParser] = [PythonSyncParser(), MarkdownSyncParser()]


def get_parser_for_file(path: str) -> MetadataParser | None:
    """Get appropriate parser for a file path."""
    for parser in BUILTIN_PARSERS:
        if path.endswith(parser.extensions):
            return parser
    return None
