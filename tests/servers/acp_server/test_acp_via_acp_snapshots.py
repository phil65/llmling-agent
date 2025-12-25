"""Snapshot tests for agentpool native Agent through ACP protocol.

This tests the full wire format by spawning agentpool serve-acp as a subprocess
and capturing the session updates received by the ACP client.

This complements the direct harness tests by verifying the protocol layer:
- Direct harness: Tests ACPSession internals → notifications
- This test: Tests full subprocess → JSON-RPC → client event flow
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
import tempfile
from typing import TYPE_CHECKING, Any

from exxec.configs import MockExecutionEnvironmentConfig
from exxec.models import ExecutionResult
import pytest
from syrupy.extensions.json import JSONSnapshotExtension
import yaml

from agentpool.agents.acp_agent import ACPAgent
from agentpool_config.toolsets import ExecutionEnvironmentToolsetConfig


if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

# Skip on Windows due to temp file locking issues with subprocesses
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Windows temp file locking")


@pytest.fixture
def json_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Configure snapshot to use JSON format."""
    return snapshot.use_extension(JSONSnapshotExtension)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_config_file(
    temp_dir: Path,
    tool_name: str,
    tool_args: dict[str, Any],
    toolsets: list[ExecutionEnvironmentToolsetConfig],
) -> Path:
    """Create a YAML config file for the subprocess agent.

    Args:
        temp_dir: Directory to write config file to
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool
        toolsets: List of toolset configs
    """
    agent_config: dict[str, Any] = {
        "type": "native",
        "model": {
            "type": "test",
            "call_tools": [tool_name],
            "tool_args": {tool_name: tool_args},
        },
        "toolsets": [t.model_dump(mode="json") for t in toolsets],
    }

    config = {"agents": {"test_agent": agent_config}}

    config_path = temp_dir / "config.yml"
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return config_path


@dataclass
class ACPViaACPHarness:
    """Test harness for capturing events from agentpool-via-ACP.

    Uses MockExecutionEnvironment with deterministic_ids for stable snapshots,
    matching the approach used by the native agent test harness.
    """

    temp_dir: Path
    recorded_events: list[dict[str, Any]] = field(default_factory=list)

    async def execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        toolsets: list[ExecutionEnvironmentToolsetConfig],
    ) -> list[dict[str, Any]]:
        """Execute a tool via ACP subprocess and capture full event details.

        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments to pass to tool
            toolsets: Toolset configurations
        """
        config_path = create_config_file(
            self.temp_dir,
            tool_name,
            tool_args,
            toolsets,
        )

        self.recorded_events.clear()

        async with ACPAgent(
            name="agentpool_via_acp",
            command="uv",
            args=[
                "run",
                "agentpool",
                "serve-acp",
                "--no-skills",
                str(config_path),
            ],
            cwd=str(self.temp_dir),
        ) as agent:
            async for event in agent.run_stream("Execute the tool"):
                from dataclasses import asdict

                event_dict = asdict(event)
                event_dict["type"] = type(event).__name__
                self.recorded_events.append(event_dict)

        return self.recorded_events


@pytest.fixture
def harness(temp_dir: Path) -> ACPViaACPHarness:
    """Create test harness."""
    return ACPViaACPHarness(temp_dir=temp_dir)


class TestExecuteCommandViaACP:
    """Test execute_command tool through ACP subprocess."""

    async def test_execute_command_simple(
        self,
        harness: ACPViaACPHarness,
        json_snapshot: SnapshotAssertion,
    ):
        """Test simple command execution via ACP with mock environment."""
        mock_env = MockExecutionEnvironmentConfig(
            deterministic_ids=True,
            command_results={
                "echo hello": asdict(
                    ExecutionResult(
                        result=None,
                        stdout="hello\n",
                        stderr="",
                        success=True,
                        exit_code=0,
                        duration=0.01,
                    )
                )
            },
        )
        toolset = ExecutionEnvironmentToolsetConfig(environment=mock_env)

        events = await harness.execute_tool(
            tool_name="execute_command",
            tool_args={"command": "echo hello"},
            toolsets=[toolset],
        )

        # Filter to tool call messages for stable comparison
        tool_events = [
            e for e in events if e["type"] in ("ToolCallStartEvent", "ToolCallProgressEvent")
        ]

        assert tool_events == json_snapshot


class TestExecuteCodeViaACP:
    """Test execute_code tool through ACP subprocess."""

    async def test_execute_code_simple(
        self,
        harness: ACPViaACPHarness,
        json_snapshot: SnapshotAssertion,
    ):
        """Test simple code execution via ACP with mock environment."""
        mock_env = MockExecutionEnvironmentConfig(
            deterministic_ids=True,
            code_results={
                "print('hello')": asdict(
                    ExecutionResult(
                        result=None,
                        stdout="hello\n",
                        stderr="",
                        success=True,
                        exit_code=0,
                        duration=0.01,
                    )
                )
            },
        )
        toolset = ExecutionEnvironmentToolsetConfig(environment=mock_env)

        events = await harness.execute_tool(
            tool_name="execute_code",
            tool_args={"code": "print('hello')"},
            toolsets=[toolset],
        )

        # Filter to tool call messages for stable comparison
        tool_events = [
            e for e in events if e["type"] in ("ToolCallStartEvent", "ToolCallProgressEvent")
        ]

        assert tool_events == json_snapshot


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
