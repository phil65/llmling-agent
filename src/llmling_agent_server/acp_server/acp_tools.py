"""ACP resource providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers import PlanProvider
from llmling_agent_toolsets.fsspec_toolset import FSSpecTools
from llmling_agent_toolsets.process_toolset import ProcessTools


if TYPE_CHECKING:
    from llmling_agent.resource_providers.aggregating import AggregatingResourceProvider
    from llmling_agent_server.acp_server.session import ACPSession


def get_acp_provider(session: ACPSession) -> AggregatingResourceProvider:
    from llmling_agent.resource_providers.aggregating import AggregatingResourceProvider

    providers = [
        PlanProvider(),
        ProcessTools(session.process_manager, name=f"acp_processes_{session.session_id}"),
        FSSpecTools(session.fs, name=f"acp_fs_{session.session_id}", cwd=session.cwd),
    ]
    return AggregatingResourceProvider(providers=providers, name=f"acp_{session.session_id}")


__all__ = ["get_acp_provider"]
