"""Snapshot tests for tool call JSON-RPC messages using full ACPSession flow.

These tests use TestModelConfig with tool_args to make the agent call specific
tools with predetermined arguments, capturing the exact wire format of JSON-RPC
messages for regression testing.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from exxec import MockExecutionEnvironment
from llmling_models.configs import TestModelConfig
import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from acp import ClientCapabilities
from acp.client.implementations import HeadlessACPClient
from acp.schema import SessionNotification
from agentpool import AgentsManifest
from agentpool.delegation import AgentPool
from agentpool.models.agents import NativeAgentConfig
from agentpool_config.toolsets import FSSpecToolsetConfig
from agentpool_server.acp_server.session import ACPSession


if TYPE_CHECKING:
    from syrupy import SnapshotAssertion


@pytest.fixture
def snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Use JSON serialization for cleaner snapshots."""
    return snapshot.use_extension(JSONSnapshotExtension)


class RecordingACPClient(HeadlessACPClient):
    """HeadlessACPClient that records wire-format messages."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wire_messages: list[dict[str, Any]] = []

    async def session_update(self, params: SessionNotification) -> None:
        """Capture wire format before delegating to parent."""
        dumped = params.model_dump(by_alias=True, exclude_none=True)
        update = dumped.get("update", {})
        update_type = update.get("sessionUpdate", "unknown")
        self.wire_messages.append({
            "type": update_type,
            "payload": update,
        })
        await super().session_update(params)

    def clear_wire_messages(self) -> None:
        """Clear recorded wire messages."""
        self.wire_messages.clear()

    def get_tool_call_messages(self) -> list[dict[str, Any]]:
        """Get only tool call related messages."""
        return [
            msg for msg in self.wire_messages if msg["type"] in ("tool_call", "tool_call_update")
        ]


@pytest.fixture
def mock_env() -> MockExecutionEnvironment:
    """Create mock execution environment."""
    return MockExecutionEnvironment()


@pytest.fixture
def recording_client() -> RecordingACPClient:
    """Create recording ACP client."""
    return RecordingACPClient(
        allow_file_operations=True,
        auto_grant_permissions=True,
    )


@pytest.fixture
def mock_acp_agent():
    """Create a mock ACP agent with tasks manager."""
    from agentpool.utils.tasks import TaskManager

    mock = AsyncMock()
    mock.tasks = TaskManager()
    return mock


def create_test_manifest(
    tool_args: dict[str, dict[str, Any]],
    call_tools: list[str],
) -> AgentsManifest:
    """Create a manifest with TestModelConfig and FSSpecToolset."""
    model_config = TestModelConfig(
        call_tools=call_tools,
        tool_args=tool_args,
    )
    agent_config = NativeAgentConfig(
        name="snapshot_test_agent",
        model=model_config,
        toolsets=[FSSpecToolsetConfig()],  # Uses agent's env by default
    )
    return AgentsManifest(agents={"snapshot_test_agent": agent_config})


def create_text_content_block(text: str):
    """Create a text content block for prompts."""
    from acp.schema import TextContentBlock

    return TextContentBlock(type="text", text=text)


async def create_session(
    manifest: AgentsManifest,
    recording_client: RecordingACPClient,
    mock_env: MockExecutionEnvironment,
    mock_acp_agent: Any,
) -> ACPSession:
    """Create an ACPSession from manifest with mock environment."""
    pool = AgentPool(manifest)

    capabilities = ClientCapabilities(fs=None, terminal=False)
    session = ACPSession(
        session_id="snapshot-test-session",
        agent_pool=pool,
        current_agent_name="snapshot_test_agent",
        cwd=tempfile.gettempdir(),
        client=recording_client,
        acp_agent=mock_acp_agent,
        client_capabilities=capabilities,
    )

    # Override agent.env AFTER session creation (session sets its own acp_env)
    for agent in pool.agents.values():
        agent.env = mock_env

    return session


class TestReadFileSnapshots:
    """Snapshot tests for read_file tool using full ACPSession flow."""

    async def test_read_file_basic(
        self,
        mock_env: MockExecutionEnvironment,
        recording_client: RecordingACPClient,
        mock_acp_agent: Any,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test basic file read produces expected notifications."""
        # Setup file content
        await mock_env.set_file_content("/test/hello.txt", "Hello, World!")

        # Create manifest with model configured to call read_file
        manifest = create_test_manifest(
            tool_args={"read_file": {"path": "/test/hello.txt"}},
            call_tools=["read_file"],
        )

        # Create session and process prompt
        session = await create_session(manifest, recording_client, mock_env, mock_acp_agent)
        recording_client.clear_wire_messages()
        content_blocks = [create_text_content_block("Read the file")]
        await session.process_prompt(content_blocks)

        # Get tool call messages only
        messages = recording_client.get_tool_call_messages()
        assert messages == snapshot

    async def test_read_file_with_line_range(
        self,
        mock_env: MockExecutionEnvironment,
        recording_client: RecordingACPClient,
        mock_acp_agent: Any,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test file read with line/limit produces expected notifications."""
        # Setup file content
        content = "\n".join(f"Line {i}" for i in range(1, 11))
        await mock_env.set_file_content("/test/lines.txt", content)

        # Create manifest with model configured to call read_file with line range
        manifest = create_test_manifest(
            tool_args={"read_file": {"path": "/test/lines.txt", "line": 3, "limit": 2}},
            call_tools=["read_file"],
        )

        # Create session and process prompt
        session = await create_session(manifest, recording_client, mock_env, mock_acp_agent)
        recording_client.clear_wire_messages()
        content_blocks = [create_text_content_block("Read lines from file")]
        await session.process_prompt(content_blocks)

        # Get tool call messages only
        messages = recording_client.get_tool_call_messages()
        assert messages == snapshot


class TestWriteFileSnapshots:
    """Snapshot tests for write_file tool using full ACPSession flow."""

    async def test_write_file_new(
        self,
        mock_env: MockExecutionEnvironment,
        recording_client: RecordingACPClient,
        mock_acp_agent: Any,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test writing a new file produces expected notifications."""
        # Create manifest with model configured to call write_file
        manifest = create_test_manifest(
            tool_args={
                "write_file": {
                    "path": "/test/new_file.txt",
                    "content": "New content here",
                }
            },
            call_tools=["write_file"],
        )

        # Create session and process prompt
        session = await create_session(manifest, recording_client, mock_env, mock_acp_agent)
        recording_client.clear_wire_messages()
        content_blocks = [create_text_content_block("Write a file")]
        await session.process_prompt(content_blocks)

        # Get tool call messages only
        messages = recording_client.get_tool_call_messages()
        assert messages == snapshot

    async def test_write_file_overwrite(
        self,
        mock_env: MockExecutionEnvironment,
        recording_client: RecordingACPClient,
        mock_acp_agent: Any,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test overwriting existing file produces expected notifications."""
        # Setup existing file
        await mock_env.set_file_content("/test/existing.txt", "Old content")

        # Create manifest with model configured to call write_file with overwrite
        manifest = create_test_manifest(
            tool_args={
                "write_file": {
                    "path": "/test/existing.txt",
                    "content": "Updated content",
                    "overwrite": True,
                }
            },
            call_tools=["write_file"],
        )

        # Create session and process prompt
        session = await create_session(manifest, recording_client, mock_env, mock_acp_agent)
        recording_client.clear_wire_messages()
        content_blocks = [create_text_content_block("Overwrite the file")]
        await session.process_prompt(content_blocks)

        # Get tool call messages only
        messages = recording_client.get_tool_call_messages()
        assert messages == snapshot


if __name__ == "__main__":
    pytest.main(["-v", __file__])
