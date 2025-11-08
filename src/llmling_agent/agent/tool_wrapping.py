"""Tool wrapping utilities for pydantic-ai integration."""

from __future__ import annotations

from functools import wraps
import inspect
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
from llmling_agent.utils.inspection import execute, get_argument_key


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
    has_run_ctx = get_argument_key(original_tool, RunContext)
    has_agent_ctx = get_argument_key(original_tool, AgentContext)

    # Check if we have separate RunContext and AgentContext parameters
    # vs RunContext[AgentContext] (which would match both but is a single param)
    has_separate_contexts = False
    if has_run_ctx and has_agent_ctx and has_run_ctx != has_agent_ctx:
        has_separate_contexts = True

    if has_separate_contexts:
        # Tool needs both contexts - create wrapper with pydantic-ai compatible signature
        # The original tool signature violates pydantic-ai's constraint that RunContext
        # can only be the first parameter, so we create a wrapper that presents the
        # correct signature to pydantic-ai and injects AgentContext internally

        # Get the original function's parameter info to reconstruct the call
        sig = inspect.signature(original_tool)
        params = list(sig.parameters.values())

        # Find parameter names for the contexts
        run_ctx_param = has_run_ctx
        agent_ctx_param = has_agent_ctx

        # Create new signature that only shows RunContext as first param
        # All other params (excluding AgentContext) will be preserved
        new_params = []
        for param in params:
            if param.name == run_ctx_param:
                # Keep RunContext as first parameter
                new_params.append(param)
            elif param.name == agent_ctx_param:
                # Skip AgentContext - it will be injected by wrapper
                continue
            else:
                # Keep all other parameters
                new_params.append(param)

        async def wrapped(ctx: RunContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                # Inject AgentContext by parameter name and call original tool
                call_kwargs = kwargs.copy()
                call_kwargs[agent_ctx_param] = agent_ctx
                return await execute(original_tool, ctx, *args, **call_kwargs)
            return await _handle_confirmation_result(result, tool, original_tool)

        # Update the wrapper's signature to hide AgentContext from pydantic-ai
        wrapped.__signature__ = sig.replace(parameters=new_params)

    # Create wrapper based on context requirements
    elif has_run_ctx:
        # Tool needs only RunContext
        async def wrapped(ctx: RunContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            # if agent_ctx.report_progress:
            #     await agent_ctx.report_progress(ctx.run_step, None)
            return await _handle_confirmation_result(
                result, tool, original_tool, ctx, *args, **kwargs
            )

    elif has_agent_ctx:
        # Tool needs only AgentContext
        async def wrapped(ctx: AgentContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            return await _handle_confirmation_result(
                result, tool, original_tool, agent_ctx, *args, **kwargs
            )

    else:
        # Tool needs no context
        async def wrapped(*args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            return await _handle_confirmation_result(
                result, tool, original_tool, *args, **kwargs
            )

    wraps(original_tool)(wrapped)  # pyright: ignore
    wrapped.__doc__ = tool.description
    wrapped.__name__ = tool.name
    return wrapped


async def _handle_confirmation_result(
    result: str,
    tool: Tool,
    original_tool: Callable,
    *args,
    **kwargs,
) -> Any:
    """Handle confirmation result and execute tool or raise appropriate exception."""
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
