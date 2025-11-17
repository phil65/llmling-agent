"""Statement-by-statement execution with automatic progress tracking."""

from __future__ import annotations

import ast
import asyncio
import inspect
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent.common_types import ProgressCallback


def count_statements(code: str) -> int:
    """Count total statements in the code."""
    try:
        tree = ast.parse(code)
        return len(tree.body)
    except SyntaxError:
        return 0


class ProgressTrackingExecutor:
    """Execute Python code statement-by-statement with automatic progress reporting."""

    def __init__(
        self, progress_callback: ProgressCallback | None = None, step_delay: float = 0.01
    ) -> None:
        """Initialize executor.

        Args:
            progress_callback: Optional callback for progress reporting
            step_delay: Delay between statement executions (for async behavior)
        """
        self.globals: dict[str, Any] = {"__builtins__": __builtins__}
        self.locals: dict[str, Any] = {}
        self.progress_callback: ProgressCallback | None = progress_callback
        self.step_delay = step_delay
        self.total_statements = 0
        self.current_statement = 0
        self.start_time: float | None = None

    def update_namespace(self, namespace: dict[str, Any]) -> None:
        """Update the execution namespace with additional globals."""
        self.globals.update(namespace)

    async def execute_with_progress(self, code: str) -> Any:
        """Execute code with automatic progress reporting.

        Args:
            code: Python source code to execute

        Returns:
            Result of code execution
        """
        self.total_statements = count_statements(code)
        self.current_statement = 0
        self.start_time = time.time()

        if self.total_statements == 0:
            return "No statements to execute"

        try:
            tree = ast.parse(code)

            for i, node in enumerate(tree.body):
                self.current_statement = i + 1

                # Report progress if callback available
                if self.progress_callback:
                    node_type = type(node).__name__
                    message = f"Executing {node_type.lower()} statement"

                    await self.progress_callback(
                        float(self.current_statement),
                        float(self.total_statements),
                        message,
                    )

                # Execute the statement
                compiled = compile(
                    ast.Module(body=[node], type_ignores=[]), f"<statement_{i}>", "exec"
                )
                exec(compiled, self.globals, self.locals)

                # Small delay for async behavior
                if self.step_delay > 0:
                    await asyncio.sleep(self.step_delay)

            # Try to get result from main() function or return success
            if "main" in self.locals and callable(self.locals["main"]):
                result = self.locals["main"]()
                if inspect.iscoroutine(result):
                    result = await result
                return result
        except Exception as e:
            # Report error with current progress
            if self.progress_callback:
                await self.progress_callback(
                    float(self.current_statement),
                    float(self.total_statements),
                    f"Error: {str(e)[:50]}...",
                )
            raise
        else:
            return "Code executed successfully"

    async def execute_statements(
        self, code: str
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Execute code statement by statement, yielding each executed statement.

        This method is useful for external progress tracking or debugging.

        Yields:
            (statement_source, metadata) tuples
        """
        self.total_statements = count_statements(code)
        self.current_statement = 0
        self.start_time = time.time()

        if self.total_statements == 0:
            return

        tree = ast.parse(code)
        for i, node in enumerate(tree.body):
            self.current_statement = i + 1
            statement_code = ast.unparse(node)
            # Create metadata
            elapsed_time = time.time() - (self.start_time or time.time())
            metadata = {
                "statement_index": i,
                "total_statements": self.total_statements,
                "node_type": type(node).__name__,
                "elapsed_seconds": elapsed_time,
                "locals_before": dict(self.locals),
            }

            # Execute the statement
            try:
                compiled = compile(
                    ast.Module(body=[node], type_ignores=[]), f"<statement_{i}>", "exec"
                )
                exec(compiled, self.globals, self.locals)
                metadata["status"] = "success"
            except Exception as e:  # noqa: BLE001
                metadata["status"] = "error"
                metadata["error"] = str(e)

            metadata["locals_after"] = dict(self.locals)
            yield statement_code, metadata

            if self.step_delay > 0:
                await asyncio.sleep(self.step_delay)


if __name__ == "__main__":
    from pydantic_ai import RunContext  # noqa: TC002

    from llmling_agent import Agent, log
    from llmling_agent.agent import AgentContext  # noqa: TC001
    from llmling_agent.agent.events import create_queuing_progress_handler

    log.configure_logging()

    async def run_me(run_context: RunContext, ctx: AgentContext):
        """Test function using the unified progress system."""
        code = """
async def main():
    x = 10
    y = 20
    z = x + y
    print(f"Result: {z}")
    for i in range(3):
        print(f"Loop: {i}")
    return f"Completed calculation: {z}"
"""

        # Create contextual progress handler (like provider does)
        queuing_handler = create_queuing_progress_handler(ctx.agent._progress_queue)

        tool_name = run_context.tool_name or "run_me"
        tool_call_id = run_context.tool_call_id or ""
        tool_input = {"python_code": code}

        async def bound_progress(
            progress: float, total: float | None, message: str
        ) -> None:
            await queuing_handler(
                progress, total, message, tool_name, tool_call_id, tool_input
            )

        executor = ProgressTrackingExecutor(bound_progress, step_delay=0.5)
        result = await executor.execute_with_progress(code)
        return result or "Code executed successfully"

    async def main() -> None:
        async with Agent(model="openai:gpt-5-nano", tools=[run_me]) as agent:
            print("🚀 Testing unified progress system...")
            async for event in agent.run_stream("Run run_me and show progress."):
                print(f"Event: {event}")

    asyncio.run(main())
