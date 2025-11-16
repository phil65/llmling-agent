"""Usage section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("Usage")


@nav.route.page("Running Agents", icon="octicon:play-16")
def _(page: mk.MkPage) -> None:
    """Different ways to run and interact with agents."""
    page += mk.MkTemplate("docs/running_agents.md")


@nav.route.page("Commands", icon="octicon:terminal-16")
def _(page: mk.MkPage) -> None:
    """Command system reference."""
    page += mk.MkTemplate("docs/commands.md")


@nav.route.page("Multi-Agent Orchestration", icon="octicon:project-symlink-16")
def _(page: mk.MkPage) -> None:
    """Working with multiple agents and orchestration patterns."""
    page += mk.MkTemplate("docs/multi_agent.md")


@nav.route.page("Web Interface", icon="octicon:browser-16")
def _(page: mk.MkPage) -> None:
    """Using the web interface."""
    page += mk.MkTemplate("docs/webui.md")


if __name__ == "__main__":
    print(nav.to_markdown())
