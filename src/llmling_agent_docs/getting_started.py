"""Getting Started section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("Getting Started")


@nav.route.page("Welcome to LLMling-Agent", hide=["toc"], is_homepage=True)
def _(page: mk.MkPage) -> None:
    page += mk.MkTemplate("docs/home.md")


@nav.route.page("Installation", icon="octicon:download-16")
def _(page: mk.MkPage) -> None:
    """Installation instructions."""
    page += mk.MkTemplate("docs/getting_started/installation.md")


@nav.route.page("Basic Concepts", icon="octicon:book-16")
def _(page: mk.MkPage) -> None:
    """Core concepts and architecture overview."""
    page += mk.MkTemplate("docs/getting_started/basic_concepts.md")


@nav.route.page("Quickstart", icon="octicon:rocket-16")
def _(page: mk.MkPage) -> None:
    """Quick introduction to using LLMling-agent."""
    page += mk.MkTemplate("docs/getting_started/quickstart.md")


@nav.route.page("Creating Agents", icon="octicon:tools-16")
def _(page: mk.MkPage) -> None:
    """Guide to creating and configuring agents."""
    page += mk.MkTemplate("docs/agents/create_agents.md")


@nav.route.page("Dependencies", hide="toc", icon="database")
def _(page: mk.MkPage) -> None:
    page += mk.MkDependencyTable(layout="badge")
    page += mk.MkPipDepTree(direction="LR")


if __name__ == "__main__":
    print(nav.to_markdown())
