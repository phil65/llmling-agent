"""Tool wrapping utilities for pydantic-ai integration."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
import inspect
import time
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext

from llmling_agent.agents.context import AgentContext
from llmling_agent.tasks import ChainAbortedError, RunAbortedError, ToolSkippedError
from llmling_agent.utils.inspection import execute, get_argument_key
from llmling_agent.utils.signatures import create_modified_signature, update_signature


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llmling_agent.hooks import AgentHooks
    from llmling_agent.tools.base import Tool


def wrap_tool[TReturn](  # noqa: PLR0915
    tool: Tool[TReturn],
    agent_ctx: AgentContext,
    hooks: AgentHooks | None = None,
) -> Callable[..., Awaitable[TReturn | None]]:
    """Wrap tool with confirmation handling and hooks.

    Strategy:
    - Tools with RunContext only: Normal pydantic-ai handling
    - Tools with AgentContext only: Treat as regular tools, inject AgentContext
    - Tools with both contexts: Present as RunContext-only to pydantic-ai, inject AgentContext
    - Tools with no context: Normal pydantic-ai handling

    Args:
        tool: The tool to wrap.
        agent_ctx: Agent context for confirmation handling and dependency injection.
        hooks: Optional AgentHooks for pre/post tool execution hooks.
    """
    fn = tool.callable
    run_ctx_key = get_argument_key(fn, RunContext)
    agent_ctx_key = get_argument_key(fn, AgentContext)

    # Validate parameter order if RunContext is present
    if run_ctx_key:
        param_names = list(inspect.signature(fn).parameters.keys())
        run_ctx_index = param_names.index(run_ctx_key)
        if run_ctx_index != 0:
            msg = f"Tool {tool.name!r}: RunContext param {run_ctx_key!r} must come first."
            raise ValueError(msg)

    async def _execute_with_hooks(
        execute_fn: Callable[..., Awaitable[TReturn]],
        tool_input: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> TReturn | None:
        """Execute tool with pre/post hooks."""
        # Pre-tool hooks
        if hooks:
            pre_result = await hooks.run_pre_tool_hooks(
                agent_name=agent_ctx.node_name,
                tool_name=tool.name,
                tool_input=tool_input,
                conversation_id=None,  # Could be passed through if needed
            )
            if pre_result.get("decision") == "deny":
                reason = pre_result.get("reason", "Blocked by pre-tool hook")
                msg = f"Tool {tool.name} blocked: {reason}"
                raise ToolSkippedError(msg)

            # Apply modified input if provided
            if modified := pre_result.get("modified_input"):
                kwargs.update(modified)

        # Execute the tool
        start_time = time.perf_counter()
        result = await execute_fn(*args, **kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Post-tool hooks
        if hooks:
            await hooks.run_post_tool_hooks(
                agent_name=agent_ctx.node_name,
                tool_name=tool.name,
                tool_input=tool_input,
                tool_output=result,
                duration_ms=duration_ms,
                conversation_id=None,
            )

        return result

    if run_ctx_key or agent_ctx_key:
        # Tool has RunContext and/or AgentContext
        async def wrapped(ctx: RunContext, *args: Any, **kwargs: Any) -> TReturn | None:  # pyright: ignore
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                # Populate AgentContext with RunContext data if needed
                if agent_ctx.data is None:
                    agent_ctx.data = ctx.deps

                if agent_ctx_key:  # inject AgentContext
                    # Create per-call copy with tool execution fields (avoids race condition)
                    call_ctx = replace(
                        agent_ctx,
                        tool_name=ctx.tool_name,
                        tool_call_id=ctx.tool_call_id,
                        tool_input=kwargs.copy(),
                    )
                    kwargs[agent_ctx_key] = call_ctx

                tool_input = kwargs.copy()
                if run_ctx_key:
                    # Pass RunContext to original function
                    return await _execute_with_hooks(
                        lambda *a, **kw: execute(fn, ctx, *a, **kw),
                        tool_input,
                        *args,
                        **kwargs,
                    )
                # Don't pass RunContext to original function since it didn't expect it
                return await _execute_with_hooks(
                    lambda *a, **kw: execute(fn, *a, **kw),
                    tool_input,
                    *args,
                    **kwargs,
                )
            await _handle_confirmation_result(result, tool.name)
            return None

    else:
        # Tool has no context - normal function call
        async def wrapped(*args: Any, **kwargs: Any) -> TReturn | None:  # type: ignore[misc]
            result = await agent_ctx.handle_confirmation(tool, kwargs)
            if result == "allow":
                tool_input = kwargs.copy()
                return await _execute_with_hooks(
                    lambda *a, **kw: execute(fn, *a, **kw),
                    tool_input,
                    *args,
                    **kwargs,
                )
            await _handle_confirmation_result(result, tool.name)
            return None

    # Apply wraps first
    wraps(fn)(wrapped)  # pyright: ignore
    # Python 3.14: functools.wraps copies __annotate__ but not __annotations__.
    # Any subsequent assignment to __annotations__ destroys __annotate__ (PEP 649).
    # Restore from original to preserve deferred annotation evaluation.
    # TODO: probably review all wraps() calls in the codebase.
    wrapped.__annotations__ = fn.__annotations__
    wrapped.__doc__ = tool.description
    wrapped.__name__ = tool.name
    # Modify signature for pydantic-ai: hide AgentContext, add RunContext if needed
    # Must be done AFTER wraps to prevent overwriting
    if agent_ctx_key and not run_ctx_key:
        # Tool has AgentContext only - make it appear to have RunContext to pydantic-ai
        new_sig = create_modified_signature(fn, remove=agent_ctx_key, inject={"ctx": RunContext})
        update_signature(wrapped, new_sig)
    elif agent_ctx_key and run_ctx_key:
        # Tool has both contexts - hide AgentContext from pydantic-ai
        new_sig = create_modified_signature(fn, remove=agent_ctx_key)
        update_signature(wrapped, new_sig)
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
