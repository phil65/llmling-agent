"""Jinja filters for llmling-agent documentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jinja2


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    import os

    from mknodes.jinja.nodeenvironment import NodeEnvironment


@jinja2.pass_environment
def pydantic_playground_url(
    env: NodeEnvironment,
    files: Mapping[str, str] | Sequence[str | os.PathLike[str]],
    active_index: int = 0,
) -> str:
    """Generate a Pydantic Playground URL from files.

    Args:
        env: The jinja environment (passed automatically)
        files: Either a mapping of filenames to content, or a sequence of file paths
        active_index: Index of the file to show as active (default: 0)

    Returns:
        URL to Pydantic Playground with files pre-loaded
    """
    import mknodes as mk

    link = mk.MkLink.for_pydantic_playground(files, active_index=active_index)
    return str(link.target)


@jinja2.pass_environment
def pydantic_playground_iframe(
    env: NodeEnvironment,
    files: Mapping[str, str] | Sequence[str | os.PathLike[str]],
    width: int = 1200,
    height: int = 900,
    active_index: int = 0,
) -> str:
    """Generate an MkIFrame for Pydantic Playground.

    Args:
        env: The jinja environment (passed automatically)
        files: Either a mapping of filenames to content, or a sequence of file paths
        width: Width of the iframe
        height: Height of the iframe
        active_index: Index of the file to show as active

    Returns:
        MkIFrame node
    """
    import mknodes as mk

    link = mk.MkLink.for_pydantic_playground(files, active_index=active_index)
    return str(mk.MkIFrame(str(link.target), width=width, height=height, parent=env.node))


@jinja2.pass_environment
def pydantic_playground_link(
    env: NodeEnvironment,
    files: Mapping[str, str] | Sequence[str | os.PathLike[str]],
    title: str = "Open in Pydantic Playground",
    active_index: int = 0,
    as_button: bool = True,
) -> str:
    """Generate an MkLink to Pydantic Playground.

    Args:
        env: The jinja environment (passed automatically)
        files: Either a mapping of filenames to content, or a sequence of file paths
        title: Link text
        active_index: Index of the file to show as active
        as_button: Whether to style as a button

    Returns:
        MkLink node
    """
    import mknodes as mk

    link = mk.MkLink.for_pydantic_playground(
        files, title=title, active_index=active_index, parent=env.node
    )
    if as_button:
        link.as_button = True
    return str(link)


@jinja2.pass_environment
def pydantic_playground(
    env: NodeEnvironment,
    files: Mapping[str, str] | Sequence[str | os.PathLike[str]],
    width: int = 1200,
    height: int = 900,
    active_index: int = 0,
    show_link: bool = True,
    link_title: str = "Open in Pydantic Playground",
) -> str:
    """Generate both iframe and link for Pydantic Playground.

    Args:
        env: The jinja environment (passed automatically)
        files: Either a mapping of filenames to content, or a sequence of file paths
        width: Width of the iframe
        height: Height of the iframe
        active_index: Index of the file to show as active
        show_link: Whether to show a link below the iframe
        link_title: Text for the link

    Returns:
        MkContainer with iframe and optional link
    """
    import mknodes as mk

    link = mk.MkLink.for_pydantic_playground(files, active_index=active_index)
    iframe = mk.MkIFrame(str(link.target), width=width, height=height)

    if show_link:
        button_link = mk.MkLink.for_pydantic_playground(
            files, title=link_title, active_index=active_index
        )
        button_link.as_button = True
        container = mk.MkContainer([iframe, button_link], parent=env.node)
        return str(container)
    container = mk.MkContainer([iframe], parent=env.node)
    return str(container)
