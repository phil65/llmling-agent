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
from typing import Any

import pytest
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.json import JSONSnapshotExtension

from agentpool.agents.acp_agent import ACPAgent


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
    toolsets: list[str],
) -> Path:
    """Create a YAML config file for the subprocess agent."""
    import json

    # Build toolset configs
    toolset_yaml = "\n".join(f"      - type: {t}" for t in toolsets)

    config = f"""
agents:
  test_agent:
    type: native
    model:
      type: test
      call_tools: [{tool_name}]
      tool_args:
        {tool_name}: {json.dumps(tool_args)}
    toolsets:
{toolset_yaml}
"""
    config_path = temp_dir / "config.yml"
    config_path.write_text(config)
    return config_path


@dataclass
class ACPViaACPHarness:
    """Test harness for capturing events from agentpool-via-ACP."""

    temp_dir: Path
    recorded_events: list[dict[str, Any]] = field(default_factory=list)

    async def execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        toolsets: list[str],
    ) -> list[dict[str, Any]]:
        """Execute a tool via ACP subprocess and capture events."""
        # Create config file
        config_path = create_config_file(self.temp_dir, tool_name, tool_args, toolsets)

        self.recorded_events.clear()

        # Run via ACPAgent
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
                # Record event type and key fields
                event_record = {
                    "type": type(event).__name__,
                }

                # Extract relevant fields based on event type
                if hasattr(event, "tool_call_id"):
                    event_record["tool_call_id"] = event.tool_call_id
                if hasattr(event, "tool_name"):
                    event_record["tool_name"] = event.tool_name
                if hasattr(event, "title"):
                    event_record["title"] = event.title
                if hasattr(event, "kind"):
                    event_record["kind"] = event.kind
                if hasattr(event, "status"):
                    event_record["status"] = event.status
                if hasattr(event, "content") and event.content:
                    # Simplify content for snapshot
                    if isinstance(event.content, str):
                        event_record["content"] = event.content
                    elif hasattr(event.content, "__iter__"):
                        event_record["content_count"] = len(list(event.content))

                self.recorded_events.append(event_record)

        return self.recorded_events


@pytest.fixture
def harness(temp_dir: Path) -> ACPViaACPHarness:
    """Create test harness."""
    return ACPViaACPHarness(temp_dir=temp_dir)


class TestReadFileViaACP:
    """Test read_file tool through ACP subprocess."""

    @pytest.mark.asyncio
    async def test_read_file_basic(
        self,
        harness: ACPViaACPHarness,
        temp_dir: Path,
        json_snapshot: SnapshotAssertion,
    ):
        """Test basic file read via ACP."""
        # Create test file
        test_file = temp_dir / "hello.txt"
        test_file.write_text("Hello, World!")

        events = await harness.execute_tool(
            tool_name="read_file",
            tool_args={"path": str(test_file)},
            toolsets=["file_access"],
        )

        # Filter to just the event types for stable comparison
        event_types = [e["type"] for e in events]
        assert event_types == json_snapshot


class TestExecuteCommandViaACP:
    """Test execute_command tool through ACP subprocess."""

    @pytest.mark.asyncio
    async def test_execute_command_simple(
        self,
        harness: ACPViaACPHarness,
        json_snapshot: SnapshotAssertion,
    ):
        """Test simple command execution via ACP."""
        events = await harness.execute_tool(
            tool_name="execute_command",
            tool_args={"command": "echo hello"},
            toolsets=["execution"],
        )

        event_types = [e["type"] for e in events]
        assert event_types == json_snapshot
