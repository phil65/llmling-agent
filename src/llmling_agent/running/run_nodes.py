"""Automatic agent function execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from llmling_agent.running.executor import discover_functions, execute_functions


if TYPE_CHECKING:
    from upathtools import JoinablePathLike

    from llmling_agent.models.manifest import AgentsManifest


async def run_nodes_async(
    config: JoinablePathLike | AgentsManifest,
    *,
    module: str | None = None,
    functions: list[str] | None = None,
    inputs: dict[str, Any] | None = None,
    parallel: bool = False,
) -> dict[str, Any]:
    """Execute node functions with dependency handling.

    Args:
        config: Node configuration (path or manifest)
        module: Optional module to discover functions from
        functions: Optional list of function names to run (auto-discovers if None)
        inputs: Optional input values for function parameters
        parallel: Whether to run independent functions in parallel

    Returns:
        Dict mapping function names to their results

    Example:
        ```python
        @node_function
        async def analyze(analyzer: Agent) -> str:
            return await analyzer.run("...")

        results = await run_nodes_async("agents.yml")
        print(results["analyze"])
        ```
    """
    from llmling_agent import AgentPool

    # Find functions to run
    if module:
        discovered = discover_functions(module)
    else:
        # Use calling module
        import inspect

        frame = inspect.currentframe()
        while frame:
            if frame.f_globals.get("__name__") != __name__:
                break
            frame = frame.f_back
        if not frame:
            msg = "Could not determine calling module"
            raise RuntimeError(msg)
        discovered = discover_functions(frame.f_globals["__file__"])

    if functions:
        discovered = [f for f in discovered if f.name in functions]

    # Run with pool
    async with AgentPool(config) as pool:
        return await execute_functions(discovered, pool, inputs=inputs, parallel=parallel)


def run_nodes(config: JoinablePathLike | AgentsManifest, **kwargs: Any) -> dict[str, Any]:
    """Run node functions synchronously.

    Convenience wrapper around run_nodes_async for sync contexts.
    See run_nodes_async for full documentation.

    Args:
        config: Agent configuration path
        **kwargs: Arguments to pass to run_nodes_async

    Returns:
        Dict mapping function names to their results
    """
    return asyncio.run(run_nodes_async(config, **kwargs))
