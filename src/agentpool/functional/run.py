"""Functional wrappers for Agent usage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack, overload

from anyenv import run_sync
from pydantic_ai import ImageUrl

from agentpool.agents.base_agent import (  # noqa: TC001
    AgentTypeLiteral,
    get_agent_class,
)


if TYPE_CHECKING:
    from agentpool.agents.base_agent import BaseAgentKwargs
    from agentpool.common_types import PromptCompatible


@overload
async def run_agent[TResult](
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    agent_type: AgentTypeLiteral = ...,
    output_type: type[TResult],
    **kwargs: Any,
) -> TResult: ...


@overload
async def run_agent(
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    agent_type: AgentTypeLiteral = ...,
    output_type: None = None,
    **kwargs: Any,
) -> str: ...


async def run_agent(  # type: ignore[misc]
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    agent_type: AgentTypeLiteral = "native",
    output_type: type[Any] | None = None,
    **kwargs: Unpack[BaseAgentKwargs],
) -> Any:
    """Run prompt through any agent type and return result.

    Args:
        prompt: The prompt to run
        image_url: Optional image URL to include with the prompt
        agent_type: Type of agent to use ("native", "acp", "agui", "claude", "codex")
        output_type: Optional structured output type
        **kwargs: Agent configuration (see BaseAgentKwargs)

    Returns:
        The agent's response content
    """
    agent_cls = get_agent_class(agent_type)

    async with agent_cls(**kwargs) as agent:
        # Convert to structured output agent if output_type specified and supported
        if output_type is not None and hasattr(agent, "to_structured"):
            final = agent.to_structured(output_type)
        else:
            final = agent

        if image_url:
            image = ImageUrl(url=image_url)
            result = await final.run(prompt, image)
        else:
            result = await final.run(prompt)
        return result.content


@overload
def run_agent_sync[TResult](
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    agent_type: AgentTypeLiteral = ...,
    output_type: type[TResult],
    **kwargs: Any,
) -> TResult: ...


@overload
def run_agent_sync(
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    agent_type: AgentTypeLiteral = ...,
    output_type: None = None,
    **kwargs: Any,
) -> str: ...


def run_agent_sync(  # type: ignore[misc]
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    agent_type: AgentTypeLiteral = "native",
    output_type: type[Any] | None = None,
    **kwargs: Unpack[BaseAgentKwargs],
) -> Any:
    """Sync wrapper for run_agent.

    Args:
        prompt: The prompt to run
        image_url: Optional image URL to include with the prompt
        agent_type: Type of agent to use ("native", "acp", "agui", "claude", "codex")
        output_type: Optional structured output type
        **kwargs: Agent configuration (see BaseAgentKwargs)

    Returns:
        The agent's response content
    """

    async def _run() -> Any:
        return await run_agent(  # type: ignore[misc]
            prompt,
            image_url,
            agent_type=agent_type,
            output_type=output_type,  # type: ignore[arg-type]
            **kwargs,
        )

    return run_sync(_run())
