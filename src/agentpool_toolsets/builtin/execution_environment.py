"""Provider for execution environment tools with event emission."""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent

from agentpool import log
from agentpool.agents.context import AgentContext  # noqa: TC001
from agentpool.resource_providers import ResourceProvider


logger = log.get_logger(__name__)


if TYPE_CHECKING:
    from exxec import ExecutionEnvironment

    from agentpool.tools.base import Tool


class ExecutionEnvironmentTools(ResourceProvider):
    """Provider for execution environment tools.

    Combines code execution and process management capabilities
    using any ExecutionEnvironment backend. Emits events via AgentContext.

    NOTE: The ACP execution environment used handles the Terminal events of the protocol,
    the toolset should deal with the ToolCall events for UI display purposes.
    """

    def __init__(self, env: ExecutionEnvironment | None = None, name: str = "execution") -> None:
        """Initialize execution environment toolset.

        Args:
            env: Execution environment to use (defaults to LocalExecutionEnvironment)
            name: The name of the toolset
        """
        super().__init__(name=name)
        self._env = env

    def get_env(self, agent_ctx: AgentContext) -> ExecutionEnvironment:
        """Get execution environment, falling back to agent's env if not set.

        Args:
            agent_ctx: Agent context to get fallback env from
        """
        if self._env is not None:
            return self._env
        return agent_ctx.agent.env

    async def get_tools(self) -> list[Tool]:
        return [
            # Code execution tools
            self.create_tool(self.execute_code, category="execute"),
            self.create_tool(self.execute_command, category="execute", open_world=True),
            # Process management tools
            self.create_tool(self.start_process, category="execute", open_world=True),
            self.create_tool(
                self.get_process_output, category="execute", read_only=True, idempotent=True
            ),
            self.create_tool(
                self.wait_for_process, category="execute", read_only=True, idempotent=True
            ),
            self.create_tool(self.kill_process, category="execute", destructive=True),
            self.create_tool(self.release_process, category="execute"),
            self.create_tool(
                self.list_processes, category="search", read_only=True, idempotent=True
            ),
        ]

    async def execute_code(self, agent_ctx: AgentContext, code: str) -> str:  # noqa: D417
        """Execute Python code and return the result.

        Args:
            code: Python code to execute
        """
        process_id: str | None = None
        output_parts: list[str] = []
        exit_code: int | None = None
        error_msg: str | None = None
        try:
            async for event in self.get_env(agent_ctx).stream_code(code):
                match event:
                    case ProcessStartedEvent(process_id=pid, command=cmd):
                        process_id = pid  # save for later on.
                        await agent_ctx.events.process_started(pid, cmd, success=True)
                    case OutputEvent(data=data):
                        output_parts.append(data)
                        if process_id:
                            await agent_ctx.events.process_output(process_id, data)
                    case ProcessCompletedEvent(exit_code=code_):
                        exit_code = code_
                        out = "".join(output_parts)
                        if process_id:
                            await agent_ctx.events.process_exit(
                                process_id, exit_code, final_output=out
                            )
                    case ProcessErrorEvent(error=err, exit_code=code_):
                        error_msg = err
                        exit_code = code_
                        if process_id:
                            await agent_ctx.events.process_exit(
                                process_id, exit_code or 1, final_output=err
                            )

            combined_output = "".join(output_parts)

            # Format as plain text for LLM
            if error_msg:
                return f"{combined_output}\n\nError: {error_msg}\nExit code: {exit_code}"

        except Exception as e:  # noqa: BLE001
            error_id = process_id or f"code_{uuid.uuid4().hex[:8]}"
            await agent_ctx.events.process_started(
                error_id, "execute_code", success=False, error=str(e)
            )
            return f"Error executing code: {e}"
        else:
            # Return just output if success, add exit code only if non-zero
            if exit_code and exit_code != 0:
                return f"{combined_output}\n\nExit code: {exit_code}"
            return combined_output

    async def execute_command(  # noqa: PLR0915, D417
        self,
        agent_ctx: AgentContext,
        command: str,
        output_limit: int | None = None,
    ) -> str:
        """Execute a shell command and return the output.

        Args:
            command: Shell command to execute
            output_limit: Maximum bytes of output to return
        """
        # process_id comes from exxec events (is terminal_id when using ACP)
        process_id: str | None = None
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        exit_code: int | None = None
        error_msg: str | None = None
        try:
            async for event in self.get_env(agent_ctx).stream_command(command):
                match event:
                    case ProcessStartedEvent(process_id=pid, command=cmd):
                        process_id = pid
                        if pid:
                            await agent_ctx.events.process_started(pid, cmd, success=True)
                        else:
                            logger.warning("ProcessStartedEvent missing process_id", command=cmd)
                    case OutputEvent(process_id=pid, data=data, stream=stream):
                        if stream == "stderr":
                            stderr_parts.append(data)
                        else:
                            stdout_parts.append(data)
                        if pid:
                            await agent_ctx.events.process_output(pid, data)
                        else:
                            logger.warning("OutputEvent missing process_id", stream=stream)
                    case ProcessCompletedEvent(process_id=pid, exit_code=code_):
                        exit_code = code_
                        combined = "".join(stdout_parts) + "".join(stderr_parts)
                        if pid:
                            await agent_ctx.events.process_exit(
                                pid, exit_code, final_output=combined
                            )
                        else:
                            msg = "ProcessCompletedEvent missing process_id,"
                            logger.warning(msg, exit_code=code_)
                    case ProcessErrorEvent(process_id=pid, error=err, exit_code=code_):
                        error_msg = err
                        exit_code = code_

            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)

            # Apply output limit if specified
            truncated = False
            if output_limit:
                if len(stdout.encode()) > output_limit:
                    out = stdout.encode()[-output_limit:].decode(errors="ignore")
                    stdout = "...[truncated]\n" + out
                    truncated = True
                if len(stderr.encode()) > output_limit:
                    out = stderr.encode()[-output_limit:].decode(errors="ignore")
                    stderr = "...[truncated]\n" + out
                    truncated = True

            # Format as plain text for LLM
            if error_msg:
                output = stdout + stderr if stdout or stderr else ""
                return f"{output}\n\nError: {error_msg}\nExit code: {exit_code}"

        except Exception as e:  # noqa: BLE001
            # Use process_id from events if available, otherwise generate fallback
            error_id = process_id or f"cmd_{uuid.uuid4().hex[:8]}"
            await agent_ctx.events.process_started(error_id, command, success=False, error=str(e))
            return f"Error executing command: {e}"
        else:
            # Combine stdout and stderr for output
            output = stdout
            if stderr:
                output = f"{stdout}\n\nSTDERR:\n{stderr}" if stdout else f"STDERR:\n{stderr}"

            # Add metadata only when relevant
            suffix_parts = []
            if truncated:
                suffix_parts.append("[output truncated]")
            if exit_code and exit_code != 0:
                suffix_parts.append(f"Exit code: {exit_code}")

            if suffix_parts:
                return f"{output}\n\n{' | '.join(suffix_parts)}"
            return output

    async def start_process(  # noqa: D417
        self,
        agent_ctx: AgentContext,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        output_limit: int | None = None,
    ) -> str:
        """Start a command in the background and return process ID.

        Args:
            command: Command to execute
            args: Command arguments
            cwd: Working directory
            env: Environment variables (added to current env)
            output_limit: Maximum bytes of output to retain
        """
        manager = self.get_env(agent_ctx).process_manager
        try:
            process_id = await manager.start_process(
                command=command,
                args=args,
                cwd=cwd,
                env=env,
                output_limit=output_limit,
            )
            await agent_ctx.events.process_started(process_id, command, success=True)

        except Exception as e:  # noqa: BLE001
            await agent_ctx.events.process_started("", command, success=False, error=str(e))
            return f"Failed to start process: {e}"
        else:
            full_cmd = f"{command} {' '.join(args)}" if args else command
            return f"Started background process {process_id}\nCommand: {full_cmd}"

    async def get_process_output(self, agent_ctx: AgentContext, process_id: str) -> str:  # noqa: D417
        """Get current output from a background process.

        Args:
            process_id: Process identifier from start_process
        """
        manager = self.get_env(agent_ctx).process_manager
        try:
            output = await manager.get_output(process_id)
            await agent_ctx.events.process_output(process_id, output.combined or "")

            combined = output.combined or ""
            status = "completed" if output.exit_code is not None else "running"

            # Format as plain text
            suffix_parts = [f"Status: {status}"]
            if output.exit_code is not None:
                suffix_parts.append(f"Exit code: {output.exit_code}")
            if output.truncated:
                suffix_parts.append("[output truncated]")

            return (
                f"{combined}\n\n{' | '.join(suffix_parts)}"
                if combined
                else " | ".join(suffix_parts)
            )
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:  # noqa: BLE001
            return f"Error getting process output: {e}"

    async def wait_for_process(self, agent_ctx: AgentContext, process_id: str) -> str:  # noqa: D417
        """Wait for background process to complete and return final output.

        Args:
            process_id: Process identifier from start_process
        """
        manager = self.get_env(agent_ctx).process_manager
        try:
            exit_code = await manager.wait_for_exit(process_id)
            output = await manager.get_output(process_id)
            await agent_ctx.events.process_exit(process_id, exit_code, final_output=output.combined)
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:  # noqa: BLE001
            return f"Error waiting for process: {e}"
        else:
            combined = output.combined or ""

            # Format as plain text
            suffix_parts = []
            if output.truncated:
                suffix_parts.append("[output truncated]")
            if exit_code != 0:
                suffix_parts.append(f"Exit code: {exit_code}")

            if suffix_parts:
                return f"{combined}\n\n{' | '.join(suffix_parts)}"
            return combined

    async def kill_process(self, agent_ctx: AgentContext, process_id: str) -> str:  # noqa: D417
        """Terminate a background process.

        Args:
            process_id: Process identifier from start_process
        """
        try:
            await self.get_env(agent_ctx).process_manager.kill_process(process_id)
            await agent_ctx.events.process_killed(process_id=process_id, success=True)
        except ValueError as e:
            await agent_ctx.events.process_killed(process_id, success=False, error=str(e))
            return f"Error: {e}"
        except Exception as e:  # noqa: BLE001
            await agent_ctx.events.process_killed(process_id, success=False, error=str(e))
            return f"Error killing process: {e}"
        else:
            return f"Process {process_id} has been terminated"

    async def release_process(self, agent_ctx: AgentContext, process_id: str) -> str:  # noqa: D417
        """Release resources for a background process.

        Args:
            process_id: Process identifier from start_process
        """
        try:
            await self.get_env(agent_ctx).process_manager.release_process(process_id)
            await agent_ctx.events.process_released(process_id=process_id, success=True)
        except ValueError as e:
            await agent_ctx.events.process_released(process_id, success=False, error=str(e))
            return f"Error: {e}"
        except Exception as e:  # noqa: BLE001
            await agent_ctx.events.process_released(process_id, success=False, error=str(e))
            return f"Error releasing process: {e}"
        else:
            return f"Process {process_id} resources have been released"

    async def list_processes(self, agent_ctx: AgentContext) -> str:
        """List all active background processes."""
        env = self.get_env(agent_ctx)
        try:
            process_ids = await env.process_manager.list_processes()
            if not process_ids:
                return "No active background processes"

            lines = [f"Active processes ({len(process_ids)}):"]
            for process_id in process_ids:
                try:
                    info = await env.process_manager.get_process_info(process_id)
                    command = info["command"]
                    args = info.get("args", [])
                    full_cmd = f"{command} {' '.join(args)}" if args else command
                    status = "running" if info.get("is_running", False) else "stopped"
                    exit_code = info.get("exit_code")
                    status_str = (
                        f"{status}" if exit_code is None else f"{status} (exit {exit_code})"
                    )
                    lines.append(f"  - {process_id}: {full_cmd} [{status_str}]")
                except Exception as e:  # noqa: BLE001
                    lines.append(f"  - {process_id}: [error getting info: {e}]")

            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"Error listing processes: {e}"
