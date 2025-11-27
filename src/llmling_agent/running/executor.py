"""Function execution management."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, get_type_hints

from llmling_agent.log import get_logger
from llmling_agent.running import with_nodes


if TYPE_CHECKING:
    import os

    from llmling_agent import AgentPool
    from llmling_agent.running.discovery import NodeFunction


logger = get_logger(__name__)


class ExecutionError(Exception):
    """Raised when function execution fails."""


def _validate_path(path: str | os.PathLike[str]) -> Path:
    if str(path) == "__main__":
        # Get the actual file being run

        path = sys.modules["__main__"].__file__  # type: ignore
        if not path:
            msg = "Could not determine main module file"
            raise ValueError(msg)
    path_obj = Path(path)
    if not path_obj.exists():
        msg = f"Module not found: {path}"
        raise ValueError(msg)
    return path_obj


def discover_functions(path: str | os.PathLike[str]) -> list[NodeFunction]:
    """Find all node functions in a module.

    Args:
        path: Path to Python module file

    Returns:
        List of discovered node functions

    Raises:
        ImportError: If module cannot be imported
        ValueError: If path is invalid
    """
    path_obj = _validate_path(path)
    # Import module
    spec = importlib.util.spec_from_file_location(path_obj.stem, path_obj)
    if not spec or not spec.loader:
        msg = f"Could not load module: {path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find decorated functions
    return [i for name, i in inspect.getmembers(module) if hasattr(i, "_node_function")]


def _sort_functions(functions: list[NodeFunction]) -> list[NodeFunction]:
    """Sort functions by order and dependencies.

    Args:
        functions: Functions to sort

    Returns:
        Sorted list of functions

    Raises:
        ValueError: If there are circular dependencies
    """
    # First by explicit order
    ordered = sorted(functions, key=lambda f: f.name)

    # Then resolve dependencies
    result = []
    seen = set()
    in_progress = set()

    def add_function(func: NodeFunction) -> None:
        if func.name in seen:
            return
        if func.name in in_progress:
            msg = f"Circular dependency detected: {func.name}"
            raise ValueError(msg)

        in_progress.add(func.name)
        # Add dependencies first
        for dep in func.depends_on:
            dep_fn = next((f for f in ordered if f.name == dep), None)
            if not dep_fn:
                msg = f"Missing dependency {dep} for {func.name}"
                raise ValueError(msg)
            add_function(dep_fn)

        result.append(func)
        in_progress.remove(func.name)
        seen.add(func.name)

    for func in ordered:
        add_function(func)

    return result


def _group_parallel(
    sorted_funcs: list[NodeFunction],
) -> list[list[NodeFunction]]:
    """Group functions that can run in parallel."""
    if not sorted_funcs:
        return []

    # Group by dependency signature
    by_deps: dict[tuple[str, ...], list[NodeFunction]] = {}

    for func in sorted_funcs:
        # Use tuple of sorted deps as key for consistent grouping
        key = tuple(sorted(func.depends_on))
        if key not in by_deps:
            by_deps[key] = []
        by_deps[key].append(func)

    # Convert to list of groups, maintaining order
    groups = []
    seen_funcs: set[str] = set()

    for func in sorted_funcs:
        key = tuple(sorted(func.depends_on))
        if func.name not in seen_funcs:
            group = by_deps[key]
            groups.append(group)
            seen_funcs.update(f.name for f in group)
    names = [[f.name for f in g] for g in groups]
    logger.debug("Grouped functions into groups", num_funcs=len(sorted_funcs), group_names=names)
    return groups


async def execute_single(
    func: NodeFunction,
    pool: AgentPool,
    available_results: dict[str, Any],
    inputs: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """Execute a single function.

    Args:
        func: Function to execute
        pool: Agent pool for injection
        available_results: Results from previous functions
        inputs: Optional input overrides

    Returns:
        Tuple of (function name, result)

    Raises:
        ExecutionError: If execution fails
    """
    logger.debug("Executing function", name=func.name)
    try:
        kwargs = func.default_inputs.copy()
        if inputs:
            kwargs.update(inputs)

        # Get type hints for the function
        hints = get_type_hints(func.func)

        # Add and validate dependency results
        for dep in func.depends_on:
            if dep not in available_results:
                msg = f"Missing result from {dep}"
                raise ExecutionError(msg)  # noqa: TRY301

            value = available_results[dep]
            if dep in hints:  # If parameter is type hinted
                _validate_value_type(value, hints[dep], func.name, dep)
            kwargs[dep] = value

        # Execute with node injection
        wrapped = with_nodes(pool)(func.func)
        result = await wrapped(**kwargs)

        # Validate return type if hinted
        if "return" in hints:
            _validate_value_type(result, hints["return"], func.name, "return")
    except Exception as e:
        msg = f"Error executing {func.name}: {e}"
        raise ExecutionError(msg) from e
    else:
        return func.name, result


def _validate_dependency_types(functions: list[NodeFunction]) -> None:
    """Validate that dependency types match return types."""
    # Get return types for all functions
    return_types = {}
    for func in functions:
        hints = get_type_hints(func.func)
        if "return" in hints:
            return_types[func.name] = hints["return"]

    # Check each function's dependencies
    for func in functions:
        hints = get_type_hints(func.func)
        for dep in func.depends_on:
            # Only validate if both dependency return type AND parameter are typed
            if dep in hints and dep in return_types:
                expected_type = hints[dep]
                provided_type = return_types[dep]
                if expected_type != provided_type:
                    msg = (
                        f"Type mismatch in {func.name}: "
                        f"dependency {dep!r} is typed as {expected_type}, "
                        f"but {dep} returns {provided_type}"
                    )
                    raise TypeError(msg)


def _validate_value_type(value: Any, expected_type: type, func_name: str, param_name: str) -> None:
    """Validate that a value matches its expected type."""
    if not isinstance(value, expected_type):
        msg = (
            f"Type error in {func_name}: parameter {param_name!r} "
            f"expected {expected_type.__name__}, got {type(value).__name__}"
        )
        raise TypeError(msg)


async def execute_functions(
    functions: list[NodeFunction],
    pool: AgentPool,
    inputs: dict[str, Any] | None = None,
    parallel: bool = False,
) -> dict[str, Any]:
    """Execute discovered functions in the right order."""
    msg = "Executing functions"
    logger.info(msg, num_functions=len(functions), parallel=parallel)
    results: dict[str, Any] = {}
    # Sort by order/dependencies
    sorted_funcs = _sort_functions(functions)
    _validate_dependency_types(sorted_funcs)

    if parallel:
        # Group functions that can run in parallel
        groups = _group_parallel(sorted_funcs)
        for i, group in enumerate(groups):
            logger.debug(
                "Executing parallel group",
                group=i + 1,
                num_groups=len(groups),
                names=[f.name for f in group],
            )

            # Ensure previous results are available
            logger.debug("Available results", results=sorted(results))
            # Run group in parallel
            tasks = [execute_single(func, pool, results, inputs) for func in group]
            group_results = await asyncio.gather(*tasks)
            # Update results after group completes
            results.update(dict(group_results))
            logger.debug("Group complete", num=i + 1)
            # Add small delay between groups to ensure timing separation
            if i < len(groups) - 1:
                await asyncio.sleep(0.02)  # 20ms between groups
    else:
        # Execute sequentially
        for func in sorted_funcs:
            name, result = await execute_single(func, pool, results, inputs)
            results[name] = result

    return results
