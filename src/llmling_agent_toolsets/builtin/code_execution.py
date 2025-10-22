"""Provider for code execution tools."""

from __future__ import annotations

import asyncio
import contextlib
import io

from pydantic_ai.tools import RunContext

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


async def execute_python(ctx: AgentContext, code: str) -> str:
    """Execute Python code directly."""
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    try:
        # Capture output
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__})
            return buf.getvalue() or "Code executed successfully"
    except Exception as e:  # noqa: BLE001
        return f"Error executing code: {e}"


async def execute_command(  # noqa: D417
    ctx: AgentContext,
    command: str,
    env: dict[str, str] | None = None,
    output_limit: int | None = None,
) -> str:
    """Execute a shell command.

    Args:
        command: Shell command to execute
        env: Environment variables to add to current environment
        output_limit: Maximum bytes of output to return
    """
    import os

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    try:
        # Prepare environment
        proc_env = dict(os.environ)
        if env:
            proc_env.update(env)

        pipe = asyncio.subprocess.PIPE
        proc = await asyncio.create_subprocess_shell(
            command, stdout=pipe, stderr=pipe, env=proc_env
        )
        stdout, stderr = await proc.communicate()

        # Combine and decode output
        output = stdout.decode() or stderr.decode() or "Command completed"

        # Apply output limit if specified
        if output_limit and len(output.encode()) > output_limit:
            # Truncate from the end to keep most recent output
            truncated_output = output.encode()[-output_limit:].decode(errors="ignore")
            output = f"...[truncated]\n{truncated_output}"
    except Exception as e:  # noqa: BLE001
        return f"Error executing command: {e}"
    else:
        return output


def create_code_execution_tools() -> list[Tool]:
    """Create tools for code execution operations."""
    return [
        Tool.from_callable(execute_python, source="builtin", category="execute"),
        Tool.from_callable(execute_command, source="builtin", category="execute"),
    ]


class CodeExecutionTools(StaticResourceProvider):
    """Provider for code execution tools."""

    def __init__(self, name: str = "code_execution"):
        super().__init__(name=name, tools=create_code_execution_tools())
