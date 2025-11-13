"""Tool wrapping utilities for pydantic-ai integration."""

from __future__ import annotations

from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext

from llmling_agent.agent.context import AgentContext
from llmling_agent.tasks.exceptions import (
    ChainAbortedError,
    RunAbortedError,
    ToolSkippedError,
)
from llmling_agent.utils.inspection import execute, get_argument_key


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llmling_agent.tools.base import Tool


def wrap_tool(
    tool: Tool,
    agent_ctx: AgentContext,
) -> Callable[..., Awaitable[Any]]:
    """Wrap tool with confirmation handling.

    Strategy:
    - Tools with RunContext only: Normal pydantic-ai handling
    - Tools with AgentContext only: Treat as regular tools, inject AgentContext
    - Tools with both contexts: Present as RunContext-only to pydantic-ai, inject AgentContext
    - Tools with no context: Normal pydantic-ai handling
    """  # noqa: E501
    fn = tool.callable
    run_ctx_key = get_argument_key(fn, RunContext)
    agent_ctx_key = get_argument_key(fn, AgentContext)
    # Check if we have separate RunContext and AgentContext parameters
    if run_ctx_key and agent_ctx_key:
        # Dual context - present RunContext-only signature to pydantic-ai
        # pydantic-ai will not see AgentContext parameter
        async def wrapped(ctx: RunContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                kwargs[agent_ctx_key] = agent_ctx
                return await execute(fn, ctx, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

        # Hide AgentContext parameter from pydantic-ai's signature analysis
        wrapped.__signature__ = get_signature_without_argument(fn, agent_ctx_key)  # type: ignore

    elif run_ctx_key:
        # RunContext only - normal pydantic-ai handling, just passthrough
        async def wrapped(ctx: RunContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                return await execute(fn, ctx, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

    elif agent_ctx_key:
        # AgentContext only - treat as regular tool, inject AgentContext
        async def wrapped(*args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                kwargs[agent_ctx_key] = agent_ctx
                return await execute(fn, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

        # Hide AgentContext parameter from pydantic-ai's signature analysis
        wrapped.__signature__ = get_signature_without_argument(fn, agent_ctx_key)  # type: ignore

    else:
        # No context - regular tool
        async def wrapped(*args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                return await execute(fn, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

    wraps(fn)(wrapped)  # pyright: ignore
    wrapped.__doc__ = tool.description
    wrapped.__name__ = tool.name
    return wrapped


async def _handle_confirmation_result(result: str, name: str) -> None:
    """Handle non-allow confirmation results."""
    match result:
        case "skip":
            msg = f"Tool {name} execution skipped"
            raise ToolSkippedError(msg)
        case "abort_run":
            msg = "Run aborted by user"
            raise RunAbortedError(msg)
        case "abort_chain":
            msg = "Agent chain aborted by user"
            raise ChainAbortedError(msg)


def get_signature_without_argument(fn: Callable, key: str) -> inspect.Signature:
    # Hide AgentContext parameter from pydantic-ai's signature analysis
    sig = inspect.signature(fn)
    new_params = [p for p in sig.parameters.values() if p.name != key]
    return sig.replace(parameters=new_params)  # type: ignore
