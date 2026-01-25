"""Test fixtures for OpenCode server tests.

Provides fixtures for testing the OpenCode server API, including:
- Mock agent pools and agents
- Server state management
- FastAPI test client setup
- Temporary directory management for git-enabled tests
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
import pytest

from agentpool.utils.time_utils import now_ms
from agentpool_server.opencode_server.dependencies import get_state
from agentpool_server.opencode_server.models import Session
from agentpool_server.opencode_server.models.common import TimeCreatedUpdated
from agentpool_server.opencode_server.routes import file_router, session_router
from agentpool_server.opencode_server.state import ServerState


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


# =============================================================================
# Temporary Directory Fixtures (similar to OpenCode's tmpdir)
# =============================================================================


@pytest.fixture
def tmp_project_dir() -> Iterator[Path]:
    """Create a temporary directory for testing.

    Yields the path to a temporary directory that is cleaned up after the test.
    """
    with tempfile.TemporaryDirectory(prefix="opencode-test-") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tmp_git_dir(tmp_project_dir: Path) -> Path:
    """Create a temporary directory with git initialized.

    Creates a git repository with an initial empty commit.
    """
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_project_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_project_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_project_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "Initial commit"],
        cwd=tmp_project_dir,
        check=True,
        capture_output=True,
    )
    return tmp_project_dir


# =============================================================================
# Mock Agent Pool Fixtures
# =============================================================================


@pytest.fixture
def mock_session_store() -> Mock:
    """Create a mock session store."""
    store = AsyncMock()
    store.list_sessions = AsyncMock(return_value=[])
    store.load = AsyncMock(return_value=None)
    store.save = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_sessions_manager(mock_session_store: Mock) -> Mock:
    """Create a mock sessions manager."""
    manager = Mock()
    manager.store = mock_session_store
    return manager


@pytest.fixture
def mock_storage() -> Mock:
    """Create a mock storage backend."""
    storage = AsyncMock()
    storage.filter_messages = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_file_ops() -> Mock:
    """Create a mock file operations tracker."""
    file_ops = Mock()
    file_ops.changes = []
    file_ops.reverted_changes = []
    file_ops.get_changes_since_message = Mock(return_value=[])
    file_ops.get_revert_operations = Mock(return_value=[])
    file_ops.get_unrevert_operations = Mock(return_value=[])
    file_ops.remove_changes_since_message = Mock()
    file_ops.restore_reverted_changes = Mock()
    return file_ops


@pytest.fixture
def mock_todos() -> Mock:
    """Create a mock todo tracker."""
    todos = Mock()
    todos.entries = []
    return todos


@pytest.fixture
def mock_manifest() -> Mock:
    """Create a mock manifest."""
    manifest = Mock()
    manifest.config_file_path = "/tmp/test-pool"
    return manifest


@pytest.fixture
def mock_pool(
    mock_sessions_manager: Mock,
    mock_storage: Mock,
    mock_file_ops: Mock,
    mock_todos: Mock,
    mock_manifest: Mock,
) -> Mock:
    """Create a mock agent pool with all required components."""
    pool = Mock()
    pool.sessions = mock_sessions_manager
    pool.storage = mock_storage
    pool.file_ops = mock_file_ops
    pool.todos = mock_todos
    pool.manifest = mock_manifest
    return pool


@pytest.fixture
def mock_env(tmp_project_dir: Path) -> Mock:
    """Create a mock agent environment.

    Uses a real AsyncLocalFileSystem for proper path traversal testing.
    """
    from upathtools.filesystems import AsyncLocalFileSystem

    env = Mock()
    # Use real async filesystem for proper path handling
    fs = AsyncLocalFileSystem()
    env.get_fs = Mock(return_value=fs)
    env.cwd = str(tmp_project_dir)
    env.execute_command = AsyncMock(
        return_value=Mock(success=True, result="command output", error=None)
    )
    return env


@pytest.fixture
def mock_agent(mock_env: Mock, mock_pool: Mock) -> Mock:
    """Create a mock agent for testing."""
    agent = Mock()
    agent.name = "test-agent"
    agent.env = mock_env
    agent._input_provider = None
    agent.run = AsyncMock(return_value=Mock(data="test response"))
    agent.agent_pool = mock_pool  # Agent carries its pool
    # Session management methods (used by session routes)
    agent.list_sessions = AsyncMock(return_value=[])
    agent.load_session = AsyncMock(return_value=None)
    return agent


# =============================================================================
# Server State Fixtures
# =============================================================================


@pytest.fixture
def server_state(
    tmp_project_dir: Path,
    mock_agent: Mock,
) -> ServerState:
    """Create a server state for testing."""
    return ServerState(working_dir=str(tmp_project_dir), agent=mock_agent)


# =============================================================================
# FastAPI Test Client Fixtures
# =============================================================================


@pytest.fixture
def app(server_state: ServerState) -> FastAPI:
    """Create a FastAPI app with all routes for testing."""
    app = FastAPI()
    app.include_router(session_router)
    app.include_router(file_router)
    app.dependency_overrides[get_state] = lambda: server_state
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a synchronous test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# =============================================================================
# Event Capture Fixtures
# =============================================================================


class EventCapture:
    """Helper class to capture broadcasted events."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    async def capture(self, event: Any) -> None:
        """Capture an event."""
        self.events.append(event)
        await self._queue.put(event)

    def get_events_by_type(self, event_type: str) -> list[Any]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.type == event_type]

    def clear(self) -> None:
        """Clear captured events."""
        self.events.clear()


@pytest.fixture
def event_capture(server_state: ServerState) -> EventCapture:
    """Create an event capture and hook it into the server state."""
    capture = EventCapture()
    # Patch the broadcast_event method to capture events
    original_broadcast = server_state.broadcast_event

    async def capturing_broadcast(event: Any) -> None:
        await capture.capture(event)
        await original_broadcast(event)

    server_state.broadcast_event = capturing_broadcast  # type: ignore[method-assign]
    return capture


# =============================================================================
# Session Factory Fixtures
# =============================================================================


@pytest.fixture
def session_factory(tmp_project_dir: Path):
    """Factory for creating test sessions."""

    def create_session(
        session_id: str = "test-session-001",
        title: str = "Test Session",
        project_id: str = "default",
    ) -> Session:
        now = now_ms()
        return Session(
            id=session_id,
            project_id=project_id,
            directory=str(tmp_project_dir),
            title=title,
            version="1",
            time=TimeCreatedUpdated(created=now, updated=now),
        )

    return create_session
