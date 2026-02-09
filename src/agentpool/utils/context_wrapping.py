"""Context wrapping utilities for instruction functions.

This module provides utilities to wrap instruction functions with appropriate
context injection for pydantic-ai compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentpool.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai import RunContext

    from agentpool.prompts.instructions import InstructionFunc

from agentpool.utils.inspection import (
    execute,
    get_argument_key,
    get_fn_name,
    get_fn_qualname,
)


logger = get_logger(__name__)


def wrap_instruction(
    fn: InstructionFunc,
    *,
    fallback: str = "",
) -> Callable[[RunContext[Any]], Awaitable[str]]:
    """Wrap an instruction function for pydantic-ai compatibility.

    This utility adapts instruction functions to pydantic-ai's expected
    signature: (RunContext) -> str. It automatically detects and injects
    appropriate context(s) based on function signature.

    Supports four patterns:
    1. No context: () -> str
    2. AgentContext only: (AgentContext) -> str
    3. RunContext only: (RunContext) -> str
    4. Both contexts: (AgentContext, RunContext) -> str

    Args:
        fn: The instruction function to wrap
        fallback: Fallback string if execution fails

    Returns:
        Wrapped async function: (RunContext) -> str

    Examples:
        No context:
            def simple() -> str:
                return "Be helpful"

            wrapped = wrap_instruction(simple)

        AgentContext only:
            async def with_agent(ctx: AgentContext) -> str:
                return f"User: {ctx.deps.user_name}"

            wrapped = wrap_instruction(with_agent)

        Both contexts:
            async def with_both(agent_ctx: AgentContext, run_ctx: RunContext) -> str:
                return f"User {agent_ctx.deps.name} using {run_ctx.model.model_name}"

            wrapped = wrap_instruction(with_both)

        Accessing AgentContext from RunContext:
            Note: RunContext.deps is AgentContext

            async def from_run_context(ctx: RunContext) -> str:
                agent_ctx: AgentContext = ctx.deps  # Access AgentContext via deps
                return f"User: {agent_ctx.data.user_name}"

            wrapped = wrap_instruction(from_run_context)
    """
    from pydantic_ai import RunContext

    from agentpool.agents.context import AgentContext

    # Detect which contexts function expects
    agent_ctx_key = get_argument_key(fn, AgentContext)
    run_ctx_key = get_argument_key(fn, RunContext)
    fn_name = get_fn_name(fn)

    async def wrapper(run_ctx: RunContext[Any]) -> str:
        """Wrapped function for pydantic-ai."""
        try:
            kwargs: dict[str, Any] = {}

            # Inject AgentContext if expected
            if agent_ctx_key:
                kwargs[agent_ctx_key] = run_ctx.deps

            # Inject RunContext if expected
            if run_ctx_key:
                kwargs[run_ctx_key] = run_ctx

            # Execute with detected context
            if kwargs:
                return await execute(fn, **kwargs)
            return await execute(fn)

        except Exception:
            # Log error and return fallback
            logger.exception(
                "Instruction execution failed",
                function=fn_name,
            )
            return fallback

    # Preserve function metadata for debugging
    wrapper.__name__ = fn_name
    wrapper.__qualname__ = get_fn_qualname(fn)

    return wrapper
