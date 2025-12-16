"""Helper functions for running examples in different environments."""

from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self, Union, get_args, get_origin


if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator


EXAMPLES_DIR = Path("src/llmling_agent_docs/examples")


def is_pyodide() -> bool:
    """Check if code is running in a Pyodide environment."""
    try:
        from js import Object  # type: ignore[import-not-found] # noqa: F401

        return True  # noqa: TRY300
    except ImportError:
        return False


def get_config_path(module_path: str | None = None, filename: str = "config.yml") -> Path:
    """Get the configuration file path based on environment.

    Args:
        module_path: Optional __file__ from the calling module (ignored in Pyodide)
        filename: Name of the config file (default: config.yml)

    Returns:
        Path to the configuration file
    """
    if is_pyodide():
        return Path(filename)
    if module_path is None:
        raise ValueError("module_path is required in non-Pyodide environment")
    return Path(module_path).parent / filename


def run[T](coro: Awaitable[T]) -> T:
    """Run a coroutine in both normal Python and Pyodide environments."""
    try:
        # Check if we're in an event loop
        asyncio.get_running_loop()
        # If we are, run until complete
        return asyncio.get_event_loop().run_until_complete(coro)
    except RuntimeError:
        # No running event loop, create one
        return asyncio.run(coro)  # type: ignore


@dataclass
class Example:
    """Represents a LLMling-agent example with its metadata."""

    name: str
    path: Path
    title: str
    description: str
    icon: str = "octicon:code-16"

    @property
    def files(self) -> list[Path]:
        """Get all Python and YAML files (excluding __init__.py)."""
        return [
            f
            for f in self.path.glob("**/*.*")
            if f.suffix in {".py", ".yml"} and not f.name.startswith("__")
        ]

    @property
    def docs(self) -> Path | None:
        """Get docs.md file if it exists."""
        docs = self.path / "docs.md"
        return docs if docs.exists() else None

    @classmethod
    def from_directory(cls, path: Path) -> Self | None:
        """Create Example from directory if it's a valid example."""
        if not path.is_dir() or path.name.startswith("__"):
            return None

        init_file = path / "__init__.py"
        if not init_file.exists():
            return None

        # Load the module to get variables
        namespace: dict[str, str] = {}
        with init_file.open() as f:
            exec(f.read(), namespace)

        # Get metadata with defaults
        title = namespace.get("TITLE", path.name.replace("_", " ").title())
        icon = namespace.get("ICON", "octicon:code-16")
        description = namespace.get("__doc__", "")

        return cls(
            name=path.name,
            path=path,
            title=title,
            description=description,
            icon=icon,
        )


def iter_examples(root: Path | str | None = None) -> Iterator[Example]:
    """Iterate over all available examples.

    Args:
        root: Optional root directory (defaults to llmling_agent_docs/examples)
    """
    root = Path(root) if root else EXAMPLES_DIR

    for path in sorted(root.iterdir()):
        if example := Example.from_directory(path):
            yield example


def get_example(name: str, root: Path | str | None = None) -> Example:
    """Get a specific example by name."""
    for example in iter_examples(root):
        if example.name == name:
            return example
    raise KeyError(f"Example {name!r} not found")


def get_discriminator_values(union_type: Any) -> dict[str, type]:
    """Extract discriminator values from a discriminated union.

    Args:
        union_type: A Union type (possibly wrapped in Annotated)

    Returns:
        Dict mapping discriminator values to their model classes
    """
    # Unwrap Annotated if present
    origin = get_origin(union_type)
    if origin is Annotated:
        union_type = get_args(union_type)[0]
        origin = get_origin(union_type)

    # Verify it's a Union
    if origin not in (Union, types.UnionType):
        msg = f"Expected Union type, got: {union_type}"
        raise TypeError(msg)

    # Get all types in the union
    union_args = get_args(union_type)

    # Extract discriminator values from each model
    result: dict[str, type] = {}
    for model_cls in union_args:
        if model_cls is type(None):
            continue

        # Get the 'type' field which serves as discriminator
        if hasattr(model_cls, "model_fields") and "type" in model_cls.model_fields:
            field = model_cls.model_fields["type"]
            if field.default is not None:
                result[field.default] = model_cls

    return result


def discriminator_to_filename(discriminator: str) -> str:
    """Convert discriminator value to expected doc filename.

    Examples:
        "file_access" -> "file-access"
        "vfs" -> "vfs"
        "agent_management" -> "agent-management"
    """
    return discriminator.replace("_", "-")


def check_docs_for_union(
    union_type: Any,
    docs_dir: Path | str,
    *,
    index_filename: str = "index",
) -> tuple[dict[str, type], set[str]]:
    """Check that all union members have corresponding doc files.

    Args:
        union_type: A discriminated Union type
        docs_dir: Directory containing the doc files
        index_filename: Filename to ignore (default: "index")

    Returns:
        Tuple of (missing_docs, extra_docs) where:
        - missing_docs: Dict of discriminator -> model class for undocumented types
        - extra_docs: Set of doc filenames without corresponding union member

    Example:
        ```python
        from llmling_agent_config.toolsets import ToolsetConfig
        missing, extra = check_docs_for_union(
            ToolsetConfig,
            Path("docs/configuration/toolsets"),
        )
        ```
    """
    docs_dir = Path(docs_dir)

    # Get discriminator values from union
    discriminators = get_discriminator_values(union_type)

    # Get doc files (filename without .md, excluding index)
    doc_files = {f.stem for f in docs_dir.glob("*.md") if f.stem != index_filename}

    # Convert discriminators to expected filenames
    expected_files = {discriminator_to_filename(d): d for d in discriminators}

    # Find mismatches
    missing_docs = {
        orig_discriminator: discriminators[orig_discriminator]
        for filename, orig_discriminator in expected_files.items()
        if filename not in doc_files
    }

    extra_docs = doc_files - set(expected_files.keys())

    return missing_docs, extra_docs


if __name__ == "__main__":
    # Example usage:
    for ex in iter_examples():
        print(f"\n{ex.title} ({ex.name})")
        print(f"Icon: {ex.icon}")
        print(f"Files: {len(ex.files)}")
        if ex.docs:
            print("Has docs.md")
        print(f"Description: {ex.description.splitlines()[0]}")
