"""Injectable process management toolset with event emission."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from anyenv.process_manager import ProcessManagerProtocol


class ProcessTools(ResourceProvider):
    """Process management toolset with injectable process manager."""

    def __init__(
        self,
        process_manager: ProcessManagerProtocol,
        name: str = "process_tools",
    ) -> None:
        """Initialize process toolset.

        Args:
            process_manager: Process manager implementation to use
            name: Provider name
        """
        super().__init__(name=name)
        self.process_manager = process_manager
        self._tools: list[Tool] | None = None

    async def get_tools(self) -> list[Tool]:
        """Get process management tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool.from_callable(self._start_process),
            Tool.from_callable(self._get_process_output),
            Tool.from_callable(self._wait_for_process),
            Tool.from_callable(self._kill_process),
            Tool.from_callable(self._release_process),
            Tool.from_callable(self._list_processes),
        ]
        return self._tools

    async def _start_process(
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

        Returns:
            Dictionary with process information or error
        """
        try:
            process_id = await self.process_manager.start_process(
                command=command,
                args=args,
                cwd=cwd,
                env=env,
                output_limit=output_limit,
            )

            # Emit success event
            await agent_ctx.events.process_started(
                process_id=process_id,
                command=command,
                args=args or [],
                cwd=cwd,
                env=env or {},
                output_limit=output_limit,
                success=True,
            )
        except Exception as e:  # noqa: BLE001
            # Emit failure event
            await agent_ctx.events.process_started(
                process_id="",
                command=command,
                args=args or [],
                cwd=cwd,
                env=env or {},
                output_limit=output_limit,
                success=False,
                error=str(e),
            )
            return {"error": f"Failed to start process: {e}"}
        else:
            return {
                "process_id": process_id,
                "command": command,
                "args": args or [],
                "cwd": cwd,
                "status": "started",
            }

    async def _get_process_output(
        self, agent_ctx: AgentContext, process_id: str
    ) -> dict[str, Any]:
        """Get current output from a background process.

        Args:
            agent_ctx: Agent execution context
            process_id: Process identifier from start_process

        Returns:
            Dictionary with process output or error
        """
        try:
            output = await self.process_manager.get_output(process_id)

            # Emit output event
            await agent_ctx.events.process_output(
                process_id=process_id,
                output=output.combined or "",
                stdout=output.stdout,
                stderr=output.stderr,
                truncated=output.truncated,
            )

            result = {
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

    async def _wait_for_process(
        self, agent_ctx: AgentContext, process_id: str
    ) -> dict[str, Any]:
        """Wait for background process to complete and return final output.

        Args:
            agent_ctx: Agent execution context
            process_id: Process identifier from start_process

        Returns:
            Dictionary with final process result or error
        """
        try:
            exit_code = await self.process_manager.wait_for_exit(process_id)
            output = await self.process_manager.get_output(process_id)

            # Emit exit event
            await agent_ctx.events.process_exit(
                process_id=process_id,
                exit_code=exit_code,
                final_output=output.combined,
                truncated=output.truncated,
            )
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

    async def _kill_process(
        self, agent_ctx: AgentContext, process_id: str
    ) -> dict[str, Any]:
        """Terminate a background process.

        Args:
            agent_ctx: Agent execution context
            process_id: Process identifier from start_process

        Returns:
            Dictionary with kill result or error
        """
        try:
            await self.process_manager.kill_process(process_id)

            # Emit kill event
            await agent_ctx.events.process_killed(
                process_id=process_id,
                success=True,
            )
        except ValueError as e:
            # Emit failure event
            await agent_ctx.events.process_killed(
                process_id=process_id,
                success=False,
                error=str(e),
            )
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001
            # Emit failure event
            await agent_ctx.events.process_killed(
                process_id=process_id,
                success=False,
                error=str(e),
            )
            return {"error": f"Error killing process: {e}"}
        else:
            return {
                "process_id": process_id,
                "status": "killed",
                "message": f"Process {process_id} has been terminated",
            }

    async def _release_process(
        self, agent_ctx: AgentContext, process_id: str
    ) -> dict[str, Any]:
        """Release resources for a background process.

        Args:
            agent_ctx: Agent execution context
            process_id: Process identifier from start_process

        Returns:
            Dictionary with release result or error
        """
        try:
            await self.process_manager.release_process(process_id)

            # Emit release event
            await agent_ctx.events.process_released(
                process_id=process_id,
                success=True,
            )
        except ValueError as e:
            # Emit failure event
            await agent_ctx.events.process_released(
                process_id=process_id,
                success=False,
                error=str(e),
            )
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001
            # Emit failure event
            await agent_ctx.events.process_released(
                process_id=process_id,
                success=False,
                error=str(e),
            )
            return {"error": f"Error releasing process: {e}"}
        else:
            return {
                "process_id": process_id,
                "status": "released",
                "message": f"Process {process_id} resources have been released",
            }

    async def _list_processes(self, agent_ctx: AgentContext) -> dict[str, Any]:
        """List all active background processes.

        Args:
            agent_ctx: Agent execution context

        Returns:
            Dictionary with process list or error
        """
        try:
            process_ids = await self.process_manager.list_processes()

            if not process_ids:
                return {
                    "processes": [],
                    "count": 0,
                    "message": "No active processes",
                }

            processes = []
            for process_id in process_ids:
                try:
                    info = await self.process_manager.get_process_info(process_id)
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

            return {
                "processes": processes,
                "count": len(processes),
            }

        except Exception as e:  # noqa: BLE001
            return {"error": f"Error listing processes: {e}"}
