"""ACP resource providers."""

from __future__ import annotations

from llmling_agent.resource_providers import PlanProvider
from llmling_agent_toolsets.fsspec_toolset import FSSpecTools
from .terminal_provider import ACPTerminalProvider

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmling_agent_server.acp_server.session import ACPSession
    from llmling_agent.resource_providers.aggregating import AggregatingResourceProvider


def get_acp_provider(session: ACPSession) -> AggregatingResourceProvider:
    from llmling_agent.resource_providers.aggregating import AggregatingResourceProvider

    providers = [
        PlanProvider(),
        ACPTerminalProvider(session),
        FSSpecTools(session.fs, name=f"acp_fs_{session.session_id}"),
    ]
    return AggregatingResourceProvider(
        providers=providers, name=f"acp_{session.session_id}"
    )


__all__ = ["ACPTerminalProvider"]
