"""Provider for execution environment tools with event emission."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import uuid

from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent

from llmling_agent import log
from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import ResourceProvider


logger = log.get_logger(__name__)


if TYPE_CHECKING:
    from exxec import ExecutionEnvironment

    from llmling_agent.tools.base import Tool


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

    async def execute_code(self, agent_ctx: AgentContext, code: str) -> dict[str, Any]:
        """Execute Python code and return the result.

        Args:
            agent_ctx: Agent execution context
            code: Python code to execute
        """
        process_id = f"code_{uuid.uuid4().hex[:8]}"
        output_parts: list[str] = []
        exit_code: int | None = None
        error_msg: str | None = None
        duration: float | None = None
        try:
            async for event in self.get_env(agent_ctx).stream_code(code):
                match event:
                    case ProcessStartedEvent(command=cmd):
                        await agent_ctx.events.process_started(process_id, cmd, success=True)
                    case OutputEvent(data=data):
                        output_parts.append(data)
                        await agent_ctx.events.process_output(process_id, data)
                    case ProcessCompletedEvent(exit_code=code_, duration=dur):
                        exit_code = code_
                        duration = dur
                        out = "".join(output_parts)
                        await agent_ctx.events.process_exit(process_id, exit_code, final_output=out)
                    case ProcessErrorEvent(error=err, exit_code=code_):
                        error_msg = err
                        exit_code = code_
                        await agent_ctx.events.process_exit(
                            process_id, exit_code or 1, final_output=err
                        )

            combined_output = "".join(output_parts)
            if error_msg:
                return {"error": error_msg, "output": combined_output, "exit_code": exit_code}

        except Exception as e:  # noqa: BLE001
            await agent_ctx.events.process_started(
                process_id, "execute_code", success=False, error=str(e)
            )
            return {"error": f"Error executing code: {e}"}
        else:
            return {"output": combined_output, "exit_code": exit_code, "duration": duration}

    async def execute_command(  # noqa: PLR0915
        self,
        agent_ctx: AgentContext,
        command: str,
        output_limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute a shell command and return the output.

        Args:
            agent_ctx: Agent execution context
            command: Shell command to execute
            output_limit: Maximum bytes of output to return
        """
        # process_id comes from exxec events (is terminal_id when using ACP)
        process_id: str | None = None
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        exit_code: int | None = None
        error_msg: str | None = None
        duration: float | None = None
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
                    case ProcessCompletedEvent(process_id=pid, exit_code=code_, duration=dur):
                        exit_code = code_
                        duration = dur
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
            if error_msg:
                return {
                    "error": error_msg,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                }
        except Exception as e:  # noqa: BLE001
            # Use process_id from events if available, otherwise generate fallback
            error_id = process_id or f"cmd_{uuid.uuid4().hex[:8]}"
            await agent_ctx.events.process_started(error_id, command, success=False, error=str(e))
            return {"success": False, "error": f"Error executing command: {e}"}
        else:
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "duration": duration,
                "truncated": truncated,
            }

    async def start_process(
        self,
        agent_ctx: AgentContext,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        output_limit: int | None = None,
    ) -> dict[str, Any]:
        """Start a command in the background and return process ID.

        Args:
            agent_ctx: Agent execution context
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
            return {"error": f"Failed to start process: {e}"}
        else:
            return {
                "process_id": process_id,
                "command": command,
                "args": args or [],
                "cwd": cwd,
                "status": "started",
            }

    async def get_process_output(self, agent_ctx: AgentContext, process_id: str) -> dict[str, Any]:
        """Get current output from a background process.

        Args:
            agent_ctx: Agent execution context
            process_id: Process identifier from start_process
        """
        manager = self.get_env(agent_ctx).process_manager
        try:
            output = await manager.get_output(process_id)
            await agent_ctx.events.process_output(process_id, output.combined or "")
            result: dict[str, Any] = {
                "process_id": process_id,
                "stdout": output.stdout or "",
                "stderr": output.stderr or "",
                "combined": output.combined or "",
                "truncated": output.truncated,
            }
            if output.exit_code is not None:
                result["exit_code"] = output.exit_code
                result["status"] = "completed"
            else:
                result["status"] = "running"
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001
            return {"error": f"Error getting process output: {e}"}
        else:
            return result

    async def wait_for_process(self, agent_ctx: AgentContext, process_id: str) -> dict[str, Any]:
        """Wait for background process to complete and return final output.

        Args:
            agent_ctx: Agent execution context
            process_id: Process identifier from start_process
        """
        manager = self.get_env(agent_ctx).process_manager
        try:
            exit_code = await manager.wait_for_exit(process_id)
            output = await manager.get_output(process_id)
            await agent_ctx.events.process_exit(process_id, exit_code, final_output=output.combined)

        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001
            return {"error": f"Error waiting for process: {e}"}
        else:
            return {
                "process_id": process_id,
                "exit_code": exit_code,
                "status": "completed",
                "stdout": output.stdout or "",
                "stderr": output.stderr or "",
                "combined": output.combined or "",
                "truncated": output.truncated,
            }

    async def kill_process(self, agent_ctx: AgentContext, process_id: str) -> dict[str, Any]:
        """Terminate a background process.

        Args:
            agent_ctx: Agent execution context
            process_id: Process identifier from start_process
        """
        try:
            await self.get_env(agent_ctx).process_manager.kill_process(process_id)
            await agent_ctx.events.process_killed(process_id=process_id, success=True)
        except ValueError as e:
            await agent_ctx.events.process_killed(process_id, success=False, error=str(e))
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001
            await agent_ctx.events.process_killed(process_id, success=False, error=str(e))
            return {"error": f"Error killing process: {e}"}
        else:
            return {
                "process_id": process_id,
                "status": "killed",
                "message": f"Process {process_id} has been terminated",
            }

    async def release_process(self, agent_ctx: AgentContext, process_id: str) -> dict[str, Any]:
        """Release resources for a background process.

        Args:
            agent_ctx: Agent execution context
            process_id: Process identifier from start_process
        """
        try:
            await self.get_env(agent_ctx).process_manager.release_process(process_id)
            await agent_ctx.events.process_released(process_id=process_id, success=True)

        except ValueError as e:
            await agent_ctx.events.process_released(process_id, success=False, error=str(e))
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001
            await agent_ctx.events.process_released(process_id, success=False, error=str(e))
            return {"error": f"Error releasing process: {e}"}
        else:
            return {
                "process_id": process_id,
                "status": "released",
                "message": f"Process {process_id} resources have been released",
            }

    async def list_processes(self, agent_ctx: AgentContext) -> dict[str, Any]:
        """List all active background processes.

        Args:
            agent_ctx: Agent execution context
        """
        env = self.get_env(agent_ctx)
        try:
            process_ids = await env.process_manager.list_processes()
            if not process_ids:
                return {"processes": [], "count": 0, "message": "No active processes"}

            processes = []
            for process_id in process_ids:
                try:
                    info = await env.process_manager.get_process_info(process_id)
                    processes.append({
                        "process_id": process_id,
                        "command": info["command"],
                        "args": info.get("args", []),
                        "cwd": info.get("cwd"),
                        "is_running": info.get("is_running", False),
                        "exit_code": info.get("exit_code"),
                        "created_at": info.get("created_at"),
                    })
                except Exception as e:  # noqa: BLE001
                    processes.append({
                        "process_id": process_id,
                        "error": f"Error getting info: {e}",
                    })

            return {"processes": processes, "count": len(processes)}
        except Exception as e:  # noqa: BLE001
            return {"error": f"Error listing processes: {e}"}
