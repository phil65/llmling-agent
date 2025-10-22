"""Provider for process management tools."""

from __future__ import annotations

from pydantic_ai.tools import RunContext

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


async def start_process(  # noqa: D417
    ctx: AgentContext,
    command: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    output_limit: int | None = None,
) -> str:
    """Start a command in the background and return immediately with process ID.

    Args:
        command: Command to execute
        args: Command arguments
        cwd: Working directory
        env: Environment variables (added to current env)
        output_limit: Maximum bytes of output to retain

    Returns:
        Process ID for tracking the background process
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        process_id = await ctx.process_manager.start_process(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            output_limit=output_limit,
        )
    except Exception as e:  # noqa: BLE001
        return f"Failed to start process: {e}"
    else:
        return f"Started process: {process_id}"


async def get_process_output(ctx: AgentContext, process_id: str) -> str:  # noqa: D417
    """Get current output from a background process.

    Args:
        process_id: Process identifier from start_background_process

    Returns:
        Current process output (stdout + stderr)
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        output = await ctx.process_manager.get_output(process_id)
        result = f"Process {process_id}:\n"
        if output.stdout:
            result += f"STDOUT:\n{output.stdout}\n"
        if output.stderr:
            result += f"STDERR:\n{output.stderr}\n"
        if output.exit_code is not None:
            result += f"Exit code: {output.exit_code}\n"
        if output.truncated:
            result += "Note: Output was truncated due to size limits\n"
        return result.strip()
    except ValueError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"Error getting process output: {e}"


async def wait_for_process(ctx: AgentContext, process_id: str) -> str:  # noqa: D417
    """Wait for background process to complete and return final output.

    Args:
        process_id: Process identifier from start_process

    Returns:
        Final process output and exit code
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        exit_code = await ctx.process_manager.wait_for_exit(process_id)
        output = await ctx.process_manager.get_output(process_id)

        result = f"Process {process_id} completed with exit code {exit_code}\n"
        if output.stdout:
            result += f"STDOUT:\n{output.stdout}\n"
        if output.stderr:
            result += f"STDERR:\n{output.stderr}\n"
        if output.truncated:
            result += "Note: Output was truncated due to size limits\n"
        return result.strip()
    except ValueError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"Error waiting for process: {e}"


async def kill_process(ctx: AgentContext, process_id: str) -> str:  # noqa: D417
    """Terminate a background process.

    Args:
        process_id: Process identifier from start_process

    Returns:
        Confirmation message
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        await ctx.process_manager.kill_process(process_id)
    except ValueError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"Error killing process: {e}"
    else:
        return f"Process {process_id} has been terminated"


async def release_process(ctx: AgentContext, process_id: str) -> str:  # noqa: D417
    """Release resources for a background process.

    Args:
        process_id: Process identifier from start_process

    Returns:
        Confirmation message
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        await ctx.process_manager.release_process(process_id)
    except ValueError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"Error releasing process: {e}"
    else:
        return f"Process {process_id} resources have been released"


async def list_processes(ctx: AgentContext) -> str:
    """List all active background processes.

    Returns:
        List of process IDs and basic information
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        process_ids = ctx.process_manager.list_processes()
        if not process_ids:
            return "No active processes"

        result = "Active processes:\n"
        for process_id in process_ids:
            try:
                info = await ctx.process_manager.get_process_info(process_id)
                status = (
                    "running" if info["is_running"] else f"exited ({info['exit_code']})"
                )
                args = " ".join(info["args"])
                result += f"- {process_id}: {info['command']} {args} [{status}]\n"
            except Exception as e:  # noqa: BLE001
                result += f"- {process_id}: Error getting info - {e}\n"
        return result.strip()
    except Exception as e:  # noqa: BLE001
        return f"Error listing processes: {e}"


def create_process_management_tools() -> list[Tool]:
    """Create tools for process management operations."""
    return [
        Tool.from_callable(start_process, source="builtin", category="execute"),
        Tool.from_callable(get_process_output, source="builtin", category="execute"),
        Tool.from_callable(wait_for_process, source="builtin", category="execute"),
        Tool.from_callable(kill_process, source="builtin", category="execute"),
        Tool.from_callable(release_process, source="builtin", category="execute"),
        Tool.from_callable(list_processes, source="builtin", category="search"),
    ]


class ProcessManagementTools(StaticResourceProvider):
    """Provider for process management tools."""

    def __init__(self, name: str = "process_management"):
        super().__init__(name=name, tools=create_process_management_tools())
