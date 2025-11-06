"""Tool wrapping utilities for pydantic-ai integration."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llmling_agent.tools.base import Tool

from pydantic_ai import RunContext

# Import the types from where they actually are
from llmling_agent.agent.context import AgentContext
from llmling_agent.tasks.exceptions import (
    ChainAbortedError,
    RunAbortedError,
    ToolSkippedError,
)
from llmling_agent.utils.inspection import execute, has_argument_type


def wrap_tool(
    tool: Tool,
    agent_ctx: AgentContext,
) -> Callable[..., Awaitable[Any]]:
    """Wrap tool with confirmation handling.

    We wrap the tool to intercept pydantic-ai's tool calls and add our confirmation
    logic before the actual execution happens. The actual tool execution (including
    moving sync functions to threads) is handled by pydantic-ai.

    Current situation is: We only get all infos for tool calls for functions with
    RunContext. In order to migitate this, we "fallback" to the AgentContext, which
    at least provides some information.
    """
    original_tool = tool.callable
    if has_argument_type(original_tool, RunContext):

        async def wrapped(ctx: RunContext[AgentContext], *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            # if agent_ctx.report_progress:
            #     await agent_ctx.report_progress(ctx.run_step, None)
            match result:
                case "allow":
                    return await execute(original_tool, ctx, *args, **kwargs)
                case "skip":
                    msg = f"Tool {tool.name} execution skipped"
                    raise ToolSkippedError(msg)
                case "abort_run":
                    msg = "Run aborted by user"
                    raise RunAbortedError(msg)
                case "abort_chain":
                    msg = "Agent chain aborted by user"
                    raise ChainAbortedError(msg)

    elif has_argument_type(original_tool, AgentContext):

        async def wrapped(ctx: AgentContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            match result:
                case "allow":
                    return await execute(original_tool, agent_ctx, *args, **kwargs)
                case "skip":
                    msg = f"Tool {tool.name} execution skipped"
                    raise ToolSkippedError(msg)
                case "abort_run":
                    msg = "Run aborted by user"
                    raise RunAbortedError(msg)
                case "abort_chain":
                    msg = "Agent chain aborted by user"
                    raise ChainAbortedError(msg)

    else:

        async def wrapped(*args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            match result:
                case "allow":
                    return await execute(original_tool, *args, **kwargs)
                case "skip":
                    msg = f"Tool {tool.name} execution skipped"
                    raise ToolSkippedError(msg)
                case "abort_run":
                    msg = "Run aborted by user"
                    raise RunAbortedError(msg)
                case "abort_chain":
                    msg = "Agent chain aborted by user"
                    raise ChainAbortedError(msg)

    wraps(original_tool)(wrapped)  # pyright: ignore
    wrapped.__doc__ = tool.description
    wrapped.__name__ = tool.name
    return wrapped
