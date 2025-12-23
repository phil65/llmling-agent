"""Snapshot tests for agentpool native Agent through ACP protocol.

This tests the full wire format by spawning agentpool serve-acp as a subprocess
and capturing the session updates received by the ACP client.

This complements the direct harness tests by verifying the protocol layer:
- Direct harness: Tests ACPSession internals → notifications
- This test: Tests full subprocess → JSON-RPC → client event flow
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import pytest
from syrupy.extensions.json import JSONSnapshotExtension
import yaml

from agentpool.agents.acp_agent import ACPAgent


if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


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
    toolsets: list[dict[str, Any]],
) -> Path:
    """Create a YAML config file for the subprocess agent.

    Args:
        temp_dir: Directory to write config file to
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool
        toolsets: List of toolset config dicts (can include environment config)
    """
    agent_config: dict[str, Any] = {
        "type": "native",
        "model": {
            "type": "test",
            "call_tools": [tool_name],
            "tool_args": {tool_name: tool_args},
        },
        "toolsets": toolsets,
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
        toolsets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute a tool via ACP subprocess and capture full event details.

        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments to pass to tool
            toolsets: Toolset configurations (can include environment config)
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
                # Convert event to dict for snapshot
                if hasattr(event, "model_dump"):
                    event_dict = event.model_dump(exclude_none=True)
                else:
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

    @pytest.mark.asyncio
    async def test_execute_command_simple(
        self,
        harness: ACPViaACPHarness,
        json_snapshot: SnapshotAssertion,
    ):
        """Test simple command execution via ACP with mock environment."""
        # Configure mock environment on the toolset itself (not the agent)
        # The execution toolset has its own environment config
        mock_env = {
            "type": "mock",
            "deterministic_ids": True,
            "command_results": {
                "echo hello": {
                    "result": None,
                    "stdout": "hello\n",
                    "stderr": "",
                    "success": True,
                    "exit_code": 0,
                    "duration": 0.01,
                }
            },
        }

        events = await harness.execute_tool(
            tool_name="execute_command",
            tool_args={"command": "echo hello"},
            toolsets=[{"type": "execution", "environment": mock_env}],
        )

        # Filter to tool call messages for stable comparison
        tool_events = [
            e for e in events if e["type"] in ("ToolCallStartEvent", "ToolCallProgressEvent")
        ]

        assert tool_events == json_snapshot


class TestExecuteCodeViaACP:
    """Test execute_code tool through ACP subprocess."""

    @pytest.mark.asyncio
    async def test_execute_code_simple(
        self,
        harness: ACPViaACPHarness,
        json_snapshot: SnapshotAssertion,
    ):
        """Test simple code execution via ACP with mock environment."""
        # Configure mock environment on the toolset itself
        mock_env = {
            "type": "mock",
            "deterministic_ids": True,
            "code_results": {
                "print('hello')": {
                    "result": None,
                    "stdout": "hello\n",
                    "stderr": "",
                    "success": True,
                    "exit_code": 0,
                    "duration": 0.01,
                }
            },
        }

        events = await harness.execute_tool(
            tool_name="execute_code",
            tool_args={"code": "print('hello')"},
            toolsets=[{"type": "execution", "environment": mock_env}],
        )

        # Filter to tool call messages for stable comparison
        tool_events = [
            e for e in events if e["type"] in ("ToolCallStartEvent", "ToolCallProgressEvent")
        ]

        assert tool_events == json_snapshot
