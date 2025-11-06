"""ACP resource providers."""

from __future__ import annotations

from .fs_provider import ACPFileSystemProvider
from .plan_provider import ACPPlanProvider
from .terminal_provider import ACPTerminalProvider

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmling_agent_server.acp_server.session import ACPSession
    from llmling_agent.resource_providers.aggregating import AggregatingResourceProvider


def get_acp_provider(session: ACPSession) -> AggregatingResourceProvider:
    from llmling_agent.resource_providers.aggregating import AggregatingResourceProvider

    providers = [
        ACPPlanProvider(session),
        ACPTerminalProvider(session),
        ACPFileSystemProvider(session),
    ]
    return AggregatingResourceProvider(
        providers=providers, name=f"acp_{session.session_id}"
    )


__all__ = ["ACPFileSystemProvider", "ACPPlanProvider", "ACPTerminalProvider"]
