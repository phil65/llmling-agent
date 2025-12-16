"""Tool to chain multiple function calls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, assert_never

from pydantic import Field
from pydantic_ai import ModelRetry
from schemez import Schema

from llmling_agent.agents.context import AgentContext  # noqa: TC001


class ErrorStrategy(StrEnum):
    """Strategy for handling errors in the pipeline."""

    STOP = "stop"  # Stop pipeline on error
    SKIP = "skip"  # Skip failed step, continue with previous result
    DEFAULT = "default"  # Use provided default value
    RETRY = "retry"  # Retry the step


class StepCondition(Schema):
    """Condition for conditional execution."""

    field: str  # Field to check in result
    operator: Literal["eq", "gt", "lt", "contains", "exists"]
    value: Any = None

    def evaluate_with_value(self, value: Any) -> bool:
        """Evaluate this condition against a value.

        Args:
            value: The value to evaluate against the condition.

        Returns:
            bool: True if the condition is met, False otherwise.
        """
        field_value = value.get(self.field) if isinstance(value, dict) else value

        match self.operator:
            case "eq":
                return bool(field_value == self.value)
            case "gt":
                return bool(field_value > self.value)
            case "lt":
                return bool(field_value < self.value)
            case "contains":
                try:
                    return self.value in field_value  # type: ignore[operator]
                except TypeError:
                    return False
            case "exists":
                return field_value is not None
            case _ as unreachable:
                assert_never(unreachable)


@dataclass
class StepResult:
    """Result of a pipeline step execution."""

    success: bool
    result: Any
    error: Exception | None = None
    retries: int = 0
    duration: float = 0.0


# Type alias for step results during execution
type StepResults = dict[str, StepResult]


class PipelineStep(Schema):
    """Single step in a tool pipeline."""

    tool: str
    input_kwarg: str = "text"
    keyword_args: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None  # Optional step name for referencing
    condition: StepCondition | None = None  # Conditional execution
    error_strategy: ErrorStrategy = ErrorStrategy.STOP
    default_value: Any = None  # Used with ErrorStrategy.DEFAULT
    max_retries: int = 0
    retry_delay: float = 1.0
    timeout: float | None = None
    depends_on: list[str] = Field(default_factory=list)  # Step dependencies


class Pipeline(Schema):
    """A pipeline of tool operations."""

    input: str | dict[str, Any]
    steps: list[PipelineStep]
    mode: Literal["sequential", "parallel"] = "sequential"
    max_parallel: int = 5  # Max concurrent steps
    collect_metrics: bool = False  # Collect execution metrics


async def _execute_step(
    ctx: AgentContext,
    step: PipelineStep,
    input_value: Any,
    results: StepResults,
) -> StepResult:
    """Execute a single pipeline step."""
    start_time = asyncio.get_event_loop().time()
    retries = 0

    while True:
        try:
            # Check condition if any
            if step.condition and not step.condition.evaluate_with_value(input_value):
                return StepResult(success=True, result=input_value, duration=0)

            tool_info = await ctx.agent.tools.get_tool(step.tool)  # Get the tool
            if isinstance(input_value, dict):  # Prepare kwargs
                kwargs = {**input_value, **step.keyword_args}
            else:
                kwargs = {step.input_kwarg: input_value, **step.keyword_args}

            # Execute with timeout if specified
            if step.timeout:
                fut = tool_info.execute(ctx, **kwargs)
                result = await asyncio.wait_for(fut, timeout=step.timeout)
            else:
                result = await tool_info.execute(ctx, **kwargs)

            duration = asyncio.get_event_loop().time() - start_time
            return StepResult(success=True, result=result, duration=duration)

        except Exception as exc:
            match step.error_strategy:
                case ErrorStrategy.STOP:
                    raise

                case ErrorStrategy.SKIP:
                    duration = asyncio.get_event_loop().time() - start_time
                    return StepResult(
                        success=False,
                        result=input_value,
                        error=exc,
                        duration=duration,
                    )

                case ErrorStrategy.DEFAULT:
                    duration = asyncio.get_event_loop().time() - start_time
                    return StepResult(
                        success=False,
                        result=step.default_value,
                        error=exc,
                        duration=duration,
                    )

                case ErrorStrategy.RETRY:
                    retries += 1
                    if retries <= step.max_retries:
                        await asyncio.sleep(step.retry_delay)
                        continue
                    raise  # Max retries exceeded


async def _execute_sequential(ctx: AgentContext, pipeline: Pipeline, results: StepResults) -> Any:
    """Execute steps sequentially."""
    current = pipeline.input
    for step in pipeline.steps:
        result = await _execute_step(ctx, step, current, results)
        if step.name:
            results[step.name] = result
        current = result.result
    return current


async def _execute_parallel(ctx: AgentContext, pipeline: Pipeline, results: StepResults) -> Any:
    """Execute independent steps in parallel."""
    semaphore = asyncio.Semaphore(pipeline.max_parallel)

    async def run_step(step: PipelineStep) -> None:
        async with semaphore:
            # Wait for dependencies
            for dep in step.depends_on:
                while dep not in results:
                    await asyncio.sleep(0.1)

            # Get input from dependency or pipeline input
            input_value = results[step.depends_on[-1]].result if step.depends_on else pipeline.input
            result = await _execute_step(ctx, step, input_value, results)
            if step.name:
                results[step.name] = result

    # Create tasks for all steps
    tasks = [run_step(step) for step in pipeline.steps]
    await asyncio.gather(*tasks)
    # Return last result
    return results[name].result if (name := pipeline.steps[-1].name) else None


async def chain_tools(
    ctx: AgentContext,
    input_data: str | dict[str, Any],
    steps: list[dict[str, Any]],
    mode: Literal["sequential", "parallel"] = "sequential",
    max_parallel: int = 5,
    collect_metrics: bool = False,
) -> Any:
    """Execute multiple tool operations in sequence or parallel.

    WHEN TO USE THIS TOOL:
    - Use this when you can plan multiple operations confidently in advance
    - Use this for common sequences you've successfully used before
    - Use this to reduce interaction rounds for known operation patterns
    - Use this when all steps are independent of intermediate results

    DO NOT USE THIS TOOL:
    - When you need to inspect intermediate results
    - When next steps depend on analyzing previous results
    - When you're unsure about the complete sequence
    - When you need to handle errors at each step individually

    Args:
        ctx: Agent context for tool execution
        input_data: Initial input for the pipeline
        steps: List of step configurations, each containing:
            - tool: Name of the tool to execute
            - input_kwarg: Keyword argument name for input (default: "text")
            - keyword_args: Additional keyword arguments
            - name: Optional step name for referencing
            - condition: Optional execution condition
            - error_strategy: How to handle errors ("stop", "skip", "default", "retry")
            - default_value: Value to use with "default" error strategy
            - max_retries: Maximum retry attempts
            - retry_delay: Delay between retries in seconds
            - timeout: Step timeout in seconds
            - depends_on: List of step names this depends on
        mode: Execution mode - "sequential" or "parallel"
        max_parallel: Maximum concurrent steps for parallel mode
        collect_metrics: Whether to collect execution metrics

    Examples:
        # Sequential processing
        await chain_tools(
            ctx,
            input_data="main.py",
            steps=[
                {"tool": "load_resource", "input_kwarg": "name"},
                {"tool": "analyze_code", "input_kwarg": "code"},
                {"tool": "format_output", "input_kwarg": "text"}
            ]
        )

        # Parallel processing with dependencies
        await chain_tools(
            ctx,
            input_data="test.py",
            mode="parallel",
            steps=[
                {"tool": "load_resource", "input_kwarg": "name", "name": "load"},
                {"tool": "analyze_code", "input_kwarg": "code", "depends_on": ["load"]},
                {"tool": "count_tokens", "input_kwarg": "text", "depends_on": ["load"]}
            ]
        )
    """
    try:
        pipeline = Pipeline(
            input=input_data,
            steps=[PipelineStep.model_validate(step) for step in steps],
            mode=mode,
            max_parallel=max_parallel,
            collect_metrics=collect_metrics,
        )
    except Exception as e:
        msg = f"Invalid pipeline configuration: {e}"
        raise ModelRetry(msg) from e
    results: StepResults = {}

    try:
        match pipeline.mode:
            case "sequential":
                return await _execute_sequential(ctx, pipeline, results)
            case "parallel":
                return await _execute_parallel(ctx, pipeline, results)
    except Exception as e:
        msg = f"Failed to execute pipeline: {e}"
        raise ModelRetry(msg) from e
