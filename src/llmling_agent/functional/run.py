"""Functional wrappers for Agent usage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack, overload

from anyenv import run_sync
from pydantic_ai import ImageUrl

from llmling_agent import Agent


if TYPE_CHECKING:
    from llmling_agent.agents.agent import AgentKwargs
    from llmling_agent.common_types import PromptCompatible


@overload
async def run_agent[TResult](
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    output_type: type[TResult],
    **kwargs: Unpack[AgentKwargs],
) -> TResult: ...


@overload
async def run_agent(
    prompt: PromptCompatible,
    image_url: str | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> str: ...


async def run_agent(
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    output_type: type[Any] | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> Any:
    """Run prompt through agent and return result."""
    async with Agent[Any, str](**kwargs) as agent:
        if image_url:
            image = ImageUrl(url=image_url)
            result = await agent.run(prompt, image, output_type=output_type)
        else:
            result = await agent.run(prompt, output_type=output_type)
        return result.content


@overload
def run_agent_sync[TResult](
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    output_type: type[TResult],
    **kwargs: Unpack[AgentKwargs],
) -> TResult: ...


@overload
def run_agent_sync(
    prompt: PromptCompatible,
    image_url: str | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> str: ...


def run_agent_sync(
    prompt: PromptCompatible,
    image_url: str | None = None,
    *,
    output_type: type[Any] | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> Any:
    """Sync wrapper for run_agent."""
    coro = run_agent(prompt, image_url, output_type=output_type, **kwargs)  # type: ignore
    return run_sync(coro)
