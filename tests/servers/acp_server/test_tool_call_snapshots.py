"""Snapshot tests for tool call JSON-RPC messages using full ACPSession flow.

These tests use the ToolCallTestHarness to capture the exact wire format of
JSON-RPC notifications for regression testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from agentpool_config.toolsets import FSSpecToolsetConfig

from .tool_call_harness import ToolCallTestHarness


if TYPE_CHECKING:
    from syrupy import SnapshotAssertion


@pytest.fixture
def snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Use JSON serialization for cleaner snapshots."""
    return snapshot.use_extension(JSONSnapshotExtension)


@pytest.fixture
def harness() -> ToolCallTestHarness:
    """Create a fresh test harness for each test."""
    return ToolCallTestHarness()


class TestReadFileSnapshots:
    """Snapshot tests for read_file tool."""

    async def test_read_file_basic(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test basic file read produces expected notifications."""
        await harness.mock_env.set_file_content("/test/hello.txt", "Hello, World!")

        messages = await harness.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/test/hello.txt"},
            toolsets=[FSSpecToolsetConfig()],
        )

        assert messages == snapshot

    async def test_read_file_with_line_range(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test file read with line/limit produces expected notifications."""
        content = "\n".join(f"Line {i}" for i in range(1, 11))
        await harness.mock_env.set_file_content("/test/lines.txt", content)

        messages = await harness.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/test/lines.txt", "line": 3, "limit": 2},
            toolsets=[FSSpecToolsetConfig()],
        )

        assert messages == snapshot


class TestWriteFileSnapshots:
    """Snapshot tests for write_file tool."""

    async def test_write_file_new(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test writing a new file produces expected notifications."""
        messages = await harness.execute_tool(
            tool_name="write_file",
            tool_args={"path": "/test/new_file.txt", "content": "New content here"},
            toolsets=[FSSpecToolsetConfig()],
        )

        assert messages == snapshot

    async def test_write_file_overwrite(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test overwriting existing file produces expected notifications."""
        await harness.mock_env.set_file_content("/test/existing.txt", "Old content")

        messages = await harness.execute_tool(
            tool_name="write_file",
            tool_args={
                "path": "/test/existing.txt",
                "content": "Updated content",
                "overwrite": True,
            },
            toolsets=[FSSpecToolsetConfig()],
        )

        assert messages == snapshot


if __name__ == "__main__":
    pytest.main(["-v", __file__])
