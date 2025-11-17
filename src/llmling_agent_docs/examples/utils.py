"""Helper functions for running examples in different environments."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self


if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator


EXAMPLES_DIR = Path("src/llmling_agent_docs/examples")


def is_pyodide() -> bool:
    """Check if code is running in a Pyodide environment."""
    try:
        from js import Object  # noqa: F401  # pyright: ignore[reportMissingImports]

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
        msg = "module_path is required in non-Pyodide environment"
        raise ValueError(msg)
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
    msg = f"Example {name!r} not found"
    raise KeyError(msg)


if __name__ == "__main__":
    # Example usage:
    for ex in iter_examples():
        print(f"\n{ex.title} ({ex.name})")
        print(f"Icon: {ex.icon}")
        print(f"Files: {len(ex.files)}")
        if ex.docs:
            print("Has docs.md")
        print(f"Description: {ex.description.splitlines()[0]}")
