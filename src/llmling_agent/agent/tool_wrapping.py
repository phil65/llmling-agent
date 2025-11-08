"""Tool wrapping utilities for pydantic-ai integration."""

from __future__ import annotations

from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llmling_agent.tools.base import Tool

from pydantic_ai import RunContext

from llmling_agent.agent.context import AgentContext
from llmling_agent.tasks.exceptions import (
    ChainAbortedError,
    RunAbortedError,
    ToolSkippedError,
)
from llmling_agent.utils.inspection import execute, get_argument_key


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
    original_tool = tool.callable
    has_run_ctx = get_argument_key(original_tool, RunContext)
    has_agent_ctx = get_argument_key(original_tool, AgentContext)

    # Check if we have separate RunContext and AgentContext parameters
    has_dual_contexts = has_run_ctx and has_agent_ctx and has_run_ctx != has_agent_ctx

    if has_dual_contexts:
        # Dual context tool - present RunContext-only signature to pydantic-ai
        agent_ctx_param = has_agent_ctx
        assert agent_ctx_param

        async def wrapped(ctx: RunContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                kwargs[agent_ctx_param] = agent_ctx
                return await execute(original_tool, ctx, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

        # Hide AgentContext parameter from pydantic-ai's signature analysis
        sig = inspect.signature(original_tool)
        new_params = [p for p in sig.parameters.values() if p.name != agent_ctx_param]
        wrapped.__signature__ = sig.replace(parameters=new_params)  # type: ignore

    elif has_run_ctx:
        # RunContext only - normal pydantic-ai handling
        async def wrapped(ctx: RunContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                return await execute(original_tool, ctx, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

    elif has_agent_ctx:
        # AgentContext only - treat as regular tool, inject context
        async def wrapped(*args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                kwargs[has_agent_ctx] = agent_ctx
                return await execute(original_tool, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

        # Hide AgentContext parameter from pydantic-ai's signature analysis
        sig = inspect.signature(original_tool)
        new_params = [p for p in sig.parameters.values() if p.name != has_agent_ctx]
        wrapped.__signature__ = sig.replace(parameters=new_params)  # type: ignore

    else:
        # No context - regular tool
        async def wrapped(*args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                return await execute(original_tool, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

    wraps(original_tool)(wrapped)  # pyright: ignore
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
