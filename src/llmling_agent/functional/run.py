"""Functional wrappers for Agent usage."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar, Unpack, overload

from llmling_agent import Agent
from llmling_agent.models.content import ImageURLContent


if TYPE_CHECKING:
    import os

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.agent.agent import AgentKwargs


TResult = TypeVar("TResult", default=str)


@overload
async def run_agent[TResult](
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    image_url: str | None = None,
    *,
    result_type: type[TResult],
    **kwargs: Unpack[AgentKwargs],
) -> TResult: ...


@overload
async def run_agent(
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    image_url: str | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> str: ...


async def run_agent(
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    image_url: str | None = None,
    *,
    result_type: type[Any] | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> Any:
    """Run prompt through agent and return result."""
    async with Agent[Any](**kwargs) as agent:
        if image_url:
            image = ImageURLContent(url=image_url)
            result = await agent.run(prompt, image, result_type=result_type)
        else:
            result = await agent.run(prompt, result_type=result_type)
        return result.content


@overload
def run_agent_sync[TResult](
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    image_url: str | None = None,
    *,
    result_type: type[TResult],
    **kwargs: Unpack[AgentKwargs],
) -> TResult: ...


@overload
def run_agent_sync(
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    image_url: str | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> str: ...


def run_agent_sync(
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    image_url: str | None = None,
    *,
    result_type: type[Any] | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> Any:
    """Sync wrapper for run_agent."""
    coro = run_agent(prompt, image_url, result_type=result_type, **kwargs)  # type: ignore
    return asyncio.run(coro)
