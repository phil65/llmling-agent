"""Tool wrapping utilities for pydantic-ai integration."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext

from llmling_agent.agent.context import AgentContext
from llmling_agent.tasks import ChainAbortedError, RunAbortedError, ToolSkippedError
from llmling_agent.utils.inspection import execute, get_argument_key
from llmling_agent.utils.signatures import create_modified_signature


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llmling_agent.tools.base import Tool


def wrap_tool(tool: Tool, agent_ctx: AgentContext) -> Callable[..., Awaitable[Any]]:
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

    if run_ctx_key or agent_ctx_key:
        # Tool has RunContext and/or AgentContext
        async def wrapped(ctx: RunContext, *args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                # Populate AgentContext with RunContext data if needed
                if agent_ctx.data is None:
                    agent_ctx.data = ctx.deps

                if agent_ctx_key:  # inject AgentContext
                    kwargs[agent_ctx_key] = agent_ctx

                if run_ctx_key:
                    # Pass RunContext to original function
                    return await execute(fn, ctx, *args, **kwargs)
                # Don't pass RunContext to original function since it didn't expect it
                return await execute(fn, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

    else:
        # Tool has no context - normal function call
        async def wrapped(*args, **kwargs):  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                return await execute(fn, *args, **kwargs)
            return await _handle_confirmation_result(result, tool.name)

    # Apply wraps first
    wraps(fn)(wrapped)  # pyright: ignore
    wrapped.__doc__ = tool.description
    wrapped.__name__ = tool.name

    # Modify signature for pydantic-ai: hide AgentContext, add RunContext if needed
    # Must be done AFTER wraps to prevent overwriting
    if agent_ctx_key and not run_ctx_key:
        # Tool has AgentContext only - make it appear to have RunContext to pydantic-ai
        new_sig = create_modified_signature(
            fn, remove=agent_ctx_key, inject={"ctx": RunContext}
        )
        wrapped.__signature__ = new_sig  # type: ignore
        # Update annotations to remove AgentContext and add RunContext
        wrapped.__annotations__ = {
            name: param.annotation for name, param in new_sig.parameters.items()
        } | {"return": new_sig.return_annotation}
    elif agent_ctx_key and run_ctx_key:
        # Tool has both contexts - hide AgentContext from pydantic-ai
        new_sig = create_modified_signature(fn, remove=agent_ctx_key)
        wrapped.__signature__ = new_sig  # type: ignore
        # Update annotations to remove AgentContext
        wrapped.__annotations__ = {
            name: param.annotation for name, param in new_sig.parameters.items()
        } | {"return": new_sig.return_annotation}

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
