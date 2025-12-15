"""ACP resource providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from exxec.acp_provider import ACPExecutionEnvironment

from llmling_agent.resource_providers import PlanProvider
from llmling_agent_toolsets.builtin import CodeTools, ExecutionEnvironmentTools
from llmling_agent_toolsets.fsspec_toolset import FSSpecTools


if TYPE_CHECKING:
    from llmling_agent.resource_providers.aggregating import AggregatingResourceProvider
    from llmling_agent_server.acp_server.session import ACPSession


def get_acp_provider(session: ACPSession) -> AggregatingResourceProvider:
    """Create aggregated resource provider with ACP-specific toolsets.

    Args:
        session: The ACP session to create providers for

    Returns:
        AggregatingResourceProvider with execution, filesystem, and code tools
    """
    from llmling_agent.resource_providers.aggregating import AggregatingResourceProvider

    execution_env = ACPExecutionEnvironment(
        fs=session.fs, requests=session.requests, cwd=session.cwd
    )

    providers = [
        PlanProvider(),
        ExecutionEnvironmentTools(env=execution_env, name=f"acp_execution_{session.session_id}"),
        FSSpecTools(execution_env, name=f"acp_fs_{session.session_id}", cwd=session.cwd),
        CodeTools(execution_env, name=f"acp_code_{session.session_id}", cwd=session.cwd),
    ]
    return AggregatingResourceProvider(providers=providers, name=f"acp_{session.session_id}")


__all__ = ["get_acp_provider"]
