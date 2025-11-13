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
    run_ctx_key = get_argument_key(tool.callable, RunContext)
    agent_ctx_key = get_argument_key(tool.callable, AgentContext)
    # Check if we have separate RunContext and AgentContext parameters
    if run_ctx_key and agent_ctx_key and run_ctx_key != agent_ctx_key:
        # Dual context - present RunContext-only signature to pydantic-ai
        # pydantic-ai will not see AgentContext parameter
        async def wrapped(ctx: RunContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                kwargs[agent_ctx_key] = agent_ctx
                return await execute(tool.callable, ctx, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

        # Hide AgentContext parameter from pydantic-ai's signature analysis
        sig = inspect.signature(tool.callable)
        new_params = [p for p in sig.parameters.values() if p.name != agent_ctx_key]
        wrapped.__signature__ = sig.replace(parameters=new_params)  # type: ignore

    elif run_ctx_key:
        # RunContext only - normal pydantic-ai handling
        async def wrapped(ctx: RunContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                return await execute(tool.callable, ctx, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

    elif agent_ctx_key:
        # AgentContext only - treat as regular tool, inject context
        async def wrapped(*args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                kwargs[agent_ctx_key] = agent_ctx
                return await execute(tool.callable, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

        # Hide AgentContext parameter from pydantic-ai's signature analysis
        sig = inspect.signature(tool.callable)
        new_params = [p for p in sig.parameters.values() if p.name != agent_ctx_key]
        wrapped.__signature__ = sig.replace(parameters=new_params)  # type: ignore

    else:
        # No context - regular tool
        async def wrapped(*args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                return await execute(tool.callable, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

    wraps(tool.callable)(wrapped)  # pyright: ignore
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
