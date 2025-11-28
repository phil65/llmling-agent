"""Provider for code execution tools."""

from __future__ import annotations

import contextlib
import io
import os

from anyenv import create_shell_process

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import StaticResourceProvider


async def execute_python(ctx: AgentContext, code: str) -> str:
    """Execute Python code directly."""
    try:
        # Capture output
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__})
            return buf.getvalue() or "Code executed successfully"
    except Exception as e:  # noqa: BLE001
        return f"Error executing code: {e}"


async def execute_command(
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
    try:
        env = dict(os.environ)
        if env:
            env.update(env)
        proc = await create_shell_process(command, stdout="pipe", stderr="pipe", env=env)
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


class CodeExecutionTools(StaticResourceProvider):
    """Provider for code execution tools."""

    def __init__(self, name: str = "code_execution") -> None:
        super().__init__(name=name)
        self._tools = [
            self.create_tool(execute_python, category="execute"),
            self.create_tool(execute_command, category="execute"),
        ]
