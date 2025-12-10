"""Jinja filters for llmling-agent documentation."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote


if TYPE_CHECKING:
    from collections.abc import Sequence
    import os

    from jinjarope import Environment


def pydantic_playground_url(
    files: Mapping[str, str] | Sequence[str | os.PathLike[str]],
    active_index: int = 0,
) -> str:
    """Generate a Pydantic Playground URL from files.

    Args:
        files: Either a mapping of filenames to content, or a sequence of file paths
        active_index: Index of the file to show as active (default: 0)

    Returns:
        URL to Pydantic Playground with files pre-loaded
    """
    match files:
        case Mapping():
            file_data: list[dict[str, str | int]] = [
                {"name": name, "content": content} for name, content in files.items()
            ]
        case [str() | Path(), *_] | []:
            file_data = []
            for path in files:
                file_path = Path(path)
                file_data.append({
                    "name": file_path.name,
                    "content": file_path.read_text("utf-8"),
                })
        case _:
            msg = f"Unsupported files type: {type(files)}"
            raise TypeError(msg)

    # Mark active file
    if file_data and 0 <= active_index < len(file_data):
        file_data[active_index]["activeIndex"] = 1

    json_str = json.dumps(file_data)
    encoded = quote(json_str)
    return f"https://pydantic.run/new?files={encoded}"


def pydantic_playground_iframe(
    files: Mapping[str, str] | Sequence[str | os.PathLike[str]],
    width: str = "100%",
    height: str = "800px",
    active_index: int = 0,
) -> str:
    """Generate an iframe HTML element for Pydantic Playground.

    Args:
        files: Either a mapping of filenames to content, or a sequence of file paths
        width: Width of the iframe
        height: Height of the iframe
        active_index: Index of the file to show as active

    Returns:
        HTML iframe element
    """
    url = pydantic_playground_url(files, active_index)
    return (
        f'<iframe src="{url}" width="{width}" height="{height}" '
        f'frameborder="0" style="border: 1px solid #ccc; border-radius: 4px;"></iframe>'
    )


def pydantic_playground_link(
    files: Mapping[str, str] | Sequence[str | os.PathLike[str]],
    title: str = "Open in Pydantic Playground",
    active_index: int = 0,
    as_button: bool = True,
) -> str:
    """Generate a markdown link to Pydantic Playground.

    Args:
        files: Either a mapping of filenames to content, or a sequence of file paths
        title: Link text
        active_index: Index of the file to show as active
        as_button: Whether to style as a button

    Returns:
        Markdown link
    """
    url = pydantic_playground_url(files, active_index)
    button_class = "{.md-button}" if as_button else ""
    return f"[{title}]({url}){button_class}"


def pydantic_playground(
    files: Mapping[str, str] | Sequence[str | os.PathLike[str]],
    width: str = "100%",
    height: str = "800px",
    active_index: int = 0,
    show_link: bool = True,
    link_title: str = "Open in Pydantic Playground",
) -> str:
    """Generate both iframe and link for Pydantic Playground.

    Args:
        files: Either a mapping of filenames to content, or a sequence of file paths
        width: Width of the iframe
        height: Height of the iframe
        active_index: Index of the file to show as active
        show_link: Whether to show a link below the iframe
        link_title: Text for the link

    Returns:
        HTML with iframe and optional link
    """
    parts = [pydantic_playground_iframe(files, width, height, active_index)]
    if show_link:
        parts.append("")
        parts.append(pydantic_playground_link(files, link_title, active_index))
    return "\n".join(parts)


def setup_jinjarope_filters(env: Environment) -> None:
    """Set up jinjarope filters for llmling-agent.

    This is called via the jinjarope.environment entry point.

    Args:
        env: The jinjarope environment to add filters to
    """
    env.filters["pydantic_playground_url"] = pydantic_playground_url
    env.filters["pydantic_playground_iframe"] = pydantic_playground_iframe
    env.filters["pydantic_playground_link"] = pydantic_playground_link
    env.filters["pydantic_playground"] = pydantic_playground
