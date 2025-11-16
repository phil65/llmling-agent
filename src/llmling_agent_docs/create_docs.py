"""YAML Configuration section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk

from llmling_agent.agent.agent import Agent


nav = mk.MkNav("YAML Configuration")


@nav.route.page("Auto-generated website", icon="octicon:file-code-16")
def _(page: mk.MkPage) -> None:
    """Complete manifest structure and organization."""
    _agent = Agent(model="gpt-5-nano")
