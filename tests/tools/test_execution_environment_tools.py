"""End-to-end tests for execution environment tools."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from anyenv.process_manager import ProcessOutput
from exxec.mock_provider import MockExecutionEnvironment
from exxec.models import ExecutionResult
import pytest

from llmling_agent import Agent, AgentContext
from llmling_agent.agents.events import ToolCallProgressEvent
from llmling_agent.models.agents import AgentConfig
from llmling_agent.models.manifest import AgentsManifest
from llmling_agent_toolsets.builtin.execution_environment import ExecutionEnvironmentTools


if TYPE_CHECKING:
    from llmling_agent.agents.events import RichAgentStreamEvent


def drain_event_queue(agent: Agent) -> list[RichAgentStreamEvent]:
    """Drain all events from the agent's event queue."""
    events: list[RichAgentStreamEvent] = []
    while not agent._event_queue.empty():
        try:
            events.append(agent._event_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return events


def get_progress_events(agent: Agent) -> list[ToolCallProgressEvent]:
    """Get all ToolCallProgressEvent from the agent's queue."""
    events = drain_event_queue(agent)
    return [e for e in events if isinstance(e, ToolCallProgressEvent)]


@pytest.fixture
def test_agent() -> Agent[None]:
    """Create a minimal agent for testing event emission."""
    return Agent(name="test_agent")


@pytest.fixture
def agent_ctx(test_agent: Agent[None]) -> AgentContext:
    """Create a real AgentContext for testing."""
    return AgentContext(
        node=test_agent,
        config=AgentConfig(name="test_agent"),
        definition=AgentsManifest(),
        tool_call_id="test_call_123",
        tool_name="test_tool",
        tool_input={"command": "echo", "args": ["hello"]},
    )


class TestToolsInitialization:
    """Test tools initialization and configuration."""

    def test_custom_env_is_used(self):
        """Test that custom environment is properly set."""
        env = MockExecutionEnvironment()
        tools = ExecutionEnvironmentTools(env=env)
        assert tools._env is env

    def test_default_env_is_created(self):
        """Test that ExecutionEnvironmentTools creates default LocalExecutionEnvironment."""
        tools = ExecutionEnvironmentTools()
        assert tools._env is None  # None means fallback to agent's env


class TestCodeExecution:
    """Test code execution tools."""

    async def test_execute_code_success(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test successful code execution."""
        env = MockExecutionEnvironment(
            code_results={
                "print(42)": ExecutionResult(
                    result=None,
                    duration=0.1,
                    success=True,
                    stdout="42\n",
                    stderr="",
                    exit_code=0,
                ),
            },
        )
        tools = ExecutionEnvironmentTools(env=env)

        result = await tools.execute_code(agent_ctx, "print(42)")
        assert "42" in result["output"]
        assert result["exit_code"] == 0

        # Check events were emitted
        events = get_progress_events(test_agent)
        assert len(events) >= 1

    async def test_execute_code_failure(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test code execution failure."""
        env = MockExecutionEnvironment(
            code_results={
                "print(x)": ExecutionResult(
                    result=None,
                    duration=0.1,
                    success=False,
                    stdout="",
                    stderr="",
                    error="NameError: name 'x' is not defined",
                    error_type="NameError",
                    exit_code=1,
                ),
            },
        )
        tools = ExecutionEnvironmentTools(env=env)

        result = await tools.execute_code(agent_ctx, "print(x)")
        assert "NameError" in result["error"]

        # Check events were emitted
        events = get_progress_events(test_agent)
        assert len(events) >= 1

    async def test_execute_code_exception(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test code execution with exception."""
        env = MockExecutionEnvironment(
            code_exceptions={
                "bad code": RuntimeError("Execution failed"),
            },
        )
        tools = ExecutionEnvironmentTools(env=env)

        result = await tools.execute_code(agent_ctx, "bad code")
        assert "Execution failed" in result["error"]

    async def test_execute_command_success(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test successful command execution."""
        env = MockExecutionEnvironment(
            command_results={
                "echo hello world": ExecutionResult(
                    result=None,
                    duration=0.2,
                    success=True,
                    stdout="hello world\n",
                    stderr="",
                    exit_code=0,
                ),
            },
        )
        tools = ExecutionEnvironmentTools(env=env)

        result = await tools.execute_command(agent_ctx, "echo hello world")
        assert result["stdout"] == "hello world\n"
        assert result["exit_code"] == 0

        # Check events were emitted
        events = get_progress_events(test_agent)
        assert len(events) >= 1

    async def test_execute_command_with_output_limit(
        self, agent_ctx: AgentContext, test_agent: Agent
    ):
        """Test command execution with output truncation."""
        long_output = "x" * 1000
        env = MockExecutionEnvironment(
            command_results={
                "echo": ExecutionResult(
                    result=None,
                    duration=0.1,
                    success=True,
                    stdout=long_output,
                    stderr="",
                    exit_code=0,
                ),
            },
        )
        tools = ExecutionEnvironmentTools(env=env)

        result = await tools.execute_command(agent_ctx, "echo", output_limit=100)
        assert "[truncated]" in result["stdout"]
        assert result["truncated"] is True


class TestProcessLifecycle:
    """Test process lifecycle scenarios."""

    async def test_start_process_success(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test successful process start."""
        env = MockExecutionEnvironment()
        tools = ExecutionEnvironmentTools(env=env)

        result = await tools.start_process(
            agent_ctx,
            command="echo",
            args=["hello", "world"],
            cwd="/tmp",
        )
        assert "process_id" in result
        assert result["command"] == "echo"
        assert result["args"] == ["hello", "world"]
        assert result["status"] == "started"

        # Check event was emitted
        events = get_progress_events(test_agent)
        assert len(events) == 1
        assert "Running: echo" in events[0].title

    async def test_start_process_failure(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test process start failure."""
        env = MockExecutionEnvironment()
        # Inject a failure into the mock process manager
        original_start = env.process_manager.start_process

        async def failing_start(*args, **kwargs):
            raise FileNotFoundError("Command not found")

        env.process_manager.start_process = failing_start
        tools = ExecutionEnvironmentTools(env=env)

        result = await tools.start_process(agent_ctx, command="nonexistent")
        assert "error" in result
        assert "Command not found" in result["error"]

        # Check event was emitted with failure
        events = get_progress_events(test_agent)
        assert len(events) == 1

    async def test_get_process_output_running(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test getting output from running process."""
        env = MockExecutionEnvironment(
            default_process_output=ProcessOutput(
                stdout="output line 1\noutput line 2\n",
                stderr="",
                combined="output line 1\noutput line 2\n",
                exit_code=None,
            ),
        )
        tools = ExecutionEnvironmentTools(env=env)

        # Start a process first
        start_result = await tools.start_process(agent_ctx, command="sleep", args=["10"])
        process_id = start_result["process_id"]
        drain_event_queue(test_agent)  # Clear start event

        result = await tools.get_process_output(agent_ctx, process_id)
        assert result["process_id"] == process_id
        assert result["status"] == "running"
        assert "output line 1" in result["stdout"]

        # Check event was emitted
        events = get_progress_events(test_agent)
        assert len(events) == 1

    async def test_get_process_output_completed(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test getting output from completed process."""
        env = MockExecutionEnvironment(
            default_process_output=ProcessOutput(
                stdout="done\n",
                stderr="",
                combined="done\n",
                exit_code=0,
            ),
        )
        tools = ExecutionEnvironmentTools(env=env)

        # Start a process first
        start_result = await tools.start_process(agent_ctx, command="echo", args=["done"])
        process_id = start_result["process_id"]

        result = await tools.get_process_output(agent_ctx, process_id)
        assert result["status"] == "completed"
        assert result["exit_code"] == 0

    async def test_wait_for_process_success(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test waiting for process completion."""
        env = MockExecutionEnvironment(
            default_process_output=ProcessOutput(
                stdout="Process completed\n",
                stderr="",
                combined="Process completed\n",
                exit_code=0,
            ),
        )
        tools = ExecutionEnvironmentTools(env=env)

        # Start a process first
        start_result = await tools.start_process(agent_ctx, command="echo", args=["done"])
        process_id = start_result["process_id"]
        drain_event_queue(test_agent)  # Clear start event

        result = await tools.wait_for_process(agent_ctx, process_id)
        assert result["process_id"] == process_id
        assert result["exit_code"] == 0
        assert result["status"] == "completed"
        assert "Process completed" in result["stdout"]

        # Check event was emitted
        events = get_progress_events(test_agent)
        assert len(events) == 1
        assert "Process exited" in events[0].title

    async def test_wait_for_process_failure(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test waiting for failed process."""
        env = MockExecutionEnvironment(
            default_process_output=ProcessOutput(
                stdout="",
                stderr="Command failed\n",
                combined="Command failed\n",
                exit_code=1,
            ),
        )
        tools = ExecutionEnvironmentTools(env=env)

        # Start a process first
        start_result = await tools.start_process(agent_ctx, command="false")
        process_id = start_result["process_id"]

        result = await tools.wait_for_process(agent_ctx, process_id)
        assert result["exit_code"] == 1
        assert "Command failed" in result["stderr"]

    async def test_kill_process_success(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test killing a process."""
        env = MockExecutionEnvironment()
        tools = ExecutionEnvironmentTools(env=env)

        # Start a process first
        start_result = await tools.start_process(agent_ctx, command="sleep", args=["100"])
        process_id = start_result["process_id"]
        drain_event_queue(test_agent)  # Clear start event

        result = await tools.kill_process(agent_ctx, process_id)
        assert result["process_id"] == process_id
        assert result["status"] == "killed"

        # Check event was emitted
        events = get_progress_events(test_agent)
        assert len(events) == 1
        assert "Killed process" in events[0].title

    async def test_kill_process_not_found(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test killing nonexistent process."""
        env = MockExecutionEnvironment()
        tools = ExecutionEnvironmentTools(env=env)

        result = await tools.kill_process(agent_ctx, "invalid")
        assert "error" in result

        # Check event was emitted with failure
        events = get_progress_events(test_agent)
        assert len(events) == 1

    async def test_release_process_success(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test releasing process resources."""
        env = MockExecutionEnvironment()
        tools = ExecutionEnvironmentTools(env=env)

        # Start a process first
        start_result = await tools.start_process(agent_ctx, command="echo")
        process_id = start_result["process_id"]
        drain_event_queue(test_agent)  # Clear start event

        result = await tools.release_process(agent_ctx, process_id)
        assert result["process_id"] == process_id
        assert result["status"] == "released"

        # Check event was emitted
        events = get_progress_events(test_agent)
        assert len(events) == 1
        assert "Released process" in events[0].title

    async def test_list_processes_empty(self, agent_ctx: AgentContext):
        """Test listing when no processes running."""
        env = MockExecutionEnvironment()
        tools = ExecutionEnvironmentTools(env=env)

        result = await tools.list_processes(agent_ctx)
        assert result["processes"] == []
        assert result["count"] == 0

    async def test_list_processes_with_results(self, agent_ctx: AgentContext):
        """Test listing active processes."""
        env = MockExecutionEnvironment()
        tools = ExecutionEnvironmentTools(env=env)

        # Start two processes
        await tools.start_process(agent_ctx, command="sleep", args=["60"], cwd="/tmp")
        await tools.start_process(agent_ctx, command="echo", args=["done"], cwd="/home")

        result = await tools.list_processes(agent_ctx)
        assert result["count"] == 2  # noqa: PLR2004
        assert len(result["processes"]) == 2  # noqa: PLR2004
        # Check that processes have expected fields
        for proc in result["processes"]:
            assert "process_id" in proc
            assert "command" in proc
            assert "args" in proc

    async def test_list_processes_partial_failure(self, agent_ctx: AgentContext):
        """Test listing when some process info fails."""
        env = MockExecutionEnvironment()
        tools = ExecutionEnvironmentTools(env=env)

        # Start a process
        await tools.start_process(agent_ctx, command="sleep", args=[])

        # Override get_process_info to fail for the second call
        original_get_info = env.process_manager.get_process_info
        call_count = 0

        async def failing_get_info(process_id: str):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("Can't get info")
            return await original_get_info(process_id)

        # Start another process, then make info fail
        await tools.start_process(agent_ctx, command="echo", args=[])
        env.process_manager.get_process_info = failing_get_info

        result = await tools.list_processes(agent_ctx)
        assert result["count"] == 2  # noqa: PLR2004
        assert result["processes"][0]["command"] == "sleep"
        assert "error" in result["processes"][1]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_truncated_output(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test handling of truncated output."""
        env = MockExecutionEnvironment(
            default_process_output=ProcessOutput(
                stdout="partial output...",
                stderr="",
                combined="partial output...",
                truncated=True,
                exit_code=None,
            ),
        )
        tools = ExecutionEnvironmentTools(env=env)

        # Start a process first
        start_result = await tools.start_process(agent_ctx, command="big_command")
        process_id = start_result["process_id"]

        result = await tools.get_process_output(agent_ctx, process_id)
        assert result["truncated"] is True

    async def test_process_with_signal(self, agent_ctx: AgentContext, test_agent: Agent):
        """Test process terminated by signal."""
        env = MockExecutionEnvironment(
            default_process_output=ProcessOutput(
                stdout="",
                stderr="Killed\n",
                combined="Killed\n",
                exit_code=-15,
                signal="15",
            ),
        )
        tools = ExecutionEnvironmentTools(env=env)

        # Start a process first
        start_result = await tools.start_process(agent_ctx, command="killed_proc")
        process_id = start_result["process_id"]

        result = await tools.wait_for_process(agent_ctx, process_id)
        assert result["exit_code"] == -15  # noqa: PLR2004
